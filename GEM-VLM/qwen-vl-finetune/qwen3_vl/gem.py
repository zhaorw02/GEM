from dataclasses import dataclass
# from typing import Any, Callable, Optional, Union
from typing import Optional, List, Dict, Union, Tuple, Any

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.modeling_outputs import  ModelOutput

from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, PretrainedConfig

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


from diffusers import AutoencoderDC
from .sana_transformer import SanaTransformer2DModel
from diffusers.models.normalization import RMSNorm
import numpy as np
import PIL
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm


from torch.nn.utils.rnn import pad_sequence


def numpy_to_pil(images: np.ndarray):
    """
    Convert a NumPy array of shape (batch, height, width, channels) to a list of PIL Images.
    """
    pil_images = []
    for img in images:
        img_uint8 = (img * 255).round().astype("uint8")
        if img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[..., 0]
        pil_images.append(img_uint8)
    return pil_images

def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids

@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen3VL causal language model (or autoregressive) outputs.
    """
)
class GEMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    dit_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None



class GEMForConditionalGeneration(Qwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    
    def __init__(self, config):
        super().__init__(config)

        self.sana = SanaTransformer2DModel.from_pretrained("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
        self.vae = AutoencoderDC.from_pretrained("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", subfolder="vae", torch_dtype=torch.bfloat16)
        self.vae.eval()
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", subfolder="scheduler")
        self.diffusion_connector = nn.Sequential(
                nn.Linear(self.config.text_config.hidden_size, 2304),
                nn.GELU(approximate="tanh"),
                nn.Linear(2304, 2304),
                RMSNorm(2304, eps=1e-5, elementwise_affine=True),
            )
        self.vae.enable_slicing()
        
        self.post_init()

      
    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask
        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        input_images: Optional[torch.FloatTensor] = None,
        weight: float = 0.1,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, GEMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        dit_loss = None
        if input_images is not None:
           
            with torch.no_grad():
                latents = self.vae.encode(input_images).latent.detach()
            
            if "shift_factor" in self.vae.config and self.vae.config.shift_factor is not None:
                latents = latents - self.vae.config.shift_factor
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents, device=latents.device)
            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=latents.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            

            starts = (input_ids[0] == self.config.vision_start_token_id).nonzero()[:, 0].tolist()
            ends   = (input_ids[0] == self.config.vision_end_token_id).nonzero()[:, 0].tolist()  

            selected_hidden_states = [] 
            encoder_attention_mask = []                      
            for s_idx, e_idx in zip(starts, ends):
                start = s_idx + 1
                end = e_idx
                hidden_states_filter = hidden_states[0, start:end, :]
            
                selected_hidden_states.append(hidden_states_filter) 
                encoder_attention_mask.append(
                    torch.ones(hidden_states_filter.shape[0], device=hidden_states_filter.device, dtype=torch.long)
                )

           
            selected_hidden_states = pad_sequence(
                selected_hidden_states,
                batch_first=True
            )  # [B, T_max, H]

            encoder_attention_mask = pad_sequence(
                encoder_attention_mask,
                batch_first=True
            )  
            
         
            diffusion_pred = self.sana(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=self.diffusion_connector(self.mask_drop(selected_hidden_states)),
                encoder_attention_mask=encoder_attention_mask, #None,
            ).sample
           
            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            dit_loss = torch.mean(
                (weighting.float() * (diffusion_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            dit_loss = dit_loss.mean()
        
        
        if dit_loss is None:
            loss = lm_loss
        else:  
            loss = lm_loss + weight * dit_loss
        
        return GEMOutput(
            loss=loss,
            lm_loss=lm_loss,
            dit_loss=dit_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.model.rope_deltas,
            
        )
    
    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.vae is not None:
            latents = latents / self.vae.config.scaling_factor
            if "shift_factor" in self.vae.config and self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            samples = self.vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    @torch.no_grad()
    def reconstruct(self, input_images):
        latents = self.vae.encode(input_images).latent.detach()
        samples = self.vae.decode(latents).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples
    
    @torch.no_grad()
    def generate_images(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        guidance_scale: float = 5.0,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 30,
        return_tensor=False,
   
        
        **kwargs,
    ):
        

        # breakpoint()
        with torch.no_grad():
            
            outs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outs[0]  
       

        starts = (input_ids[0] == self.config.vision_start_token_id).nonzero()[:, 0].tolist()
        ends   = (input_ids[0] == self.config.vision_end_token_id).nonzero()[:, 0].tolist()  

        pred_latent = []                       
        for s_idx, e_idx in zip(starts, ends):
            start = s_idx + 1
            end = e_idx
            hidden_states_filter = hidden_states[0, start:end :]
            
            pred_latent.append(hidden_states_filter)
        pred_latent = torch.stack(pred_latent, dim=0)  
          
        
        img_hidden_states_null = torch.zeros_like(pred_latent)
        pred_latent = torch.cat([img_hidden_states_null, pred_latent], 0)
        ## sample images from here
        device = next(self.parameters()).device
       

        bsz = 1
        
        vae_compression_factor = 32  
        latent_h = height // vae_compression_factor
        latent_w = width // vae_compression_factor
        
        latent_channels = self.sana.config.in_channels
        latents = randn_tensor(
            shape=(bsz * len(starts), latent_channels, latent_h, latent_w),
            generator=None,
            device=device,
            dtype=torch.bfloat16,
        )

        # set step values
        if isinstance(self.noise_scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.noise_scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            self.noise_scheduler.set_timesteps(num_inference_steps)


        # Convert to float32 before saving
        
        for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling images"):

            latent_model_input = torch.cat([latents] * 2)
           
            latent_model_input = latent_model_input.to(pred_latent.dtype)

            if hasattr(self.noise_scheduler.timesteps, "scale_model_input"):
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            # predict noise model_output
            with torch.no_grad():
                noise_pred = self.sana(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=self.diffusion_connector(pred_latent),
                    timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latents.device),
                    encoder_attention_mask=None
                ).sample


            noise_pred_uncond, noise_pred = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        samples = self.decode_latents(latents.to(self.vae.dtype) if self.vae is not None else latents, return_tensor=return_tensor)      


        return samples
    
        
        
        
        
        
        