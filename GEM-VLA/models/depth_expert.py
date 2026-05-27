"""Depth Expert module for joint VLM+RDT+Depth training.

Wraps SANA transformer, VAE (frozen), diffusion connector, and noise scheduler
into a single nn.Module that computes depth prediction loss given VLM hidden states
and ground-truth depth images.

Architecture: VLM hidden → connector → SANA transformer for depth flow-matching
"""

import torch
import torch.nn as nn
import numpy as np

from diffusers import AutoencoderDC
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
from torch.nn.utils.rnn import pad_sequence


class DepthExpert(nn.Module):
    """Depth prediction expert using SANA diffusion conditioned on VLM hidden states.

    Architecture:
        VLM hidden states (vision tokens) -> diffusion_connector -> SANA transformer
        Depth target image -> VAE encoder (frozen) -> latent -> add noise -> SANA input
        Loss: flow-matching MSE on predicted vs true velocity
    """

    def __init__(
        self,
        vlm_hidden_size: int = 2048,
        sana_pretrained_path: str = None,
        dtype=torch.bfloat16,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.dtype = dtype

        # SANA transformer for depth prediction
        self.sana = SanaTransformer2DModel.from_pretrained(
            sana_pretrained_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        # VAE for encoding depth images to latent space (frozen)
        self.vae = AutoencoderDC.from_pretrained(
            sana_pretrained_path,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self.vae.eval()
        self.vae.enable_slicing()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Noise scheduler for flow matching
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            sana_pretrained_path,
            subfolder="scheduler",
        )

        # Connector: project VLM hidden states to SANA cross-attention dimension
        self.diffusion_connector = nn.Sequential(
            nn.Linear(vlm_hidden_size, 2304),
            nn.GELU(approximate="tanh"),
            nn.Linear(2304, 2304),
            RMSNorm(2304, eps=1e-5, elementwise_affine=True),
        )

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps
        ]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents):
        """Randomly zero out entire batch elements for classifier-free guidance."""
        if self.drop_prob <= 0 or not self.training:
            return latents
        mask = torch.bernoulli(
            torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype)
            + self.drop_prob
        )
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # flip: 0 -> 1 (keep), 1 -> 0 (drop)
        return latents * mask

    def extract_vision_hidden_states(
        self, hidden_states, input_ids, vision_start_token_id, vision_end_token_id
    ):
        """Extract hidden states corresponding to vision tokens for each image.

        Args:
            hidden_states: (B, seq_len, hidden_size) - last-layer VLM output
            input_ids: (B, seq_len) - input token IDs
            vision_start_token_id: int
            vision_end_token_id: int

        Returns:
            selected_hidden_states: (num_images, max_vis_len, hidden_size) padded
            encoder_attention_mask: (num_images, max_vis_len) padded mask
        """
        selected_list = []
        mask_list = []

        # Process each sample in the batch
        for b in range(input_ids.shape[0]):
            starts = (input_ids[b] == vision_start_token_id).nonzero()[:, 0].tolist()
            ends = (input_ids[b] == vision_end_token_id).nonzero()[:, 0].tolist()

            for s_idx, e_idx in zip(starts, ends):
                h = hidden_states[b, s_idx + 1 : e_idx, :]
                selected_list.append(h)
                mask_list.append(
                    torch.ones(h.shape[0], device=h.device, dtype=torch.long)
                )

        if not selected_list:
            return None, None

        selected_hidden_states = pad_sequence(selected_list, batch_first=True)
        encoder_attention_mask = pad_sequence(mask_list, batch_first=True)

        return selected_hidden_states, encoder_attention_mask

    def compute_loss(
        self,
        hidden_states,
        input_ids,
        depth_images,
        vision_start_token_id,
        vision_end_token_id,
    ):
        """Compute depth prediction loss.

        Args:
            hidden_states: (B, seq_len, hidden_size) - VLM last-layer hidden states
            input_ids: (B, seq_len) - token IDs for vision token extraction
            depth_images: (num_images, 3, H, W) - ground-truth depth images, normalized to [-1, 1]
            vision_start_token_id: int - token ID marking vision block start
            vision_end_token_id: int - token ID marking vision block end

        Returns:
            dit_loss: scalar tensor
        """
        # 1. Encode depth images to latent space
        with torch.no_grad():
            latents = self.vae.encode(depth_images.to(self.dtype)).latent.detach()

        if (
            "shift_factor" in self.vae.config
            and self.vae.config.shift_factor is not None
        ):
            latents = latents - self.vae.config.shift_factor
        latents = latents * self.vae.config.scaling_factor

        # 2. Sample noise and timesteps (flow matching)
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
        sigmas = self.get_sigmas(
            timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
        )
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        # 3. Extract vision-token hidden states from VLM output
        selected_hidden_states, encoder_attention_mask = (
            self.extract_vision_hidden_states(
                hidden_states, input_ids, vision_start_token_id, vision_end_token_id
            )
        )

        if selected_hidden_states is None:
            raise RuntimeError(
                "No vision tokens found in input_ids. "
                "Joint training requires images in every sample."
            )

        # Ensure num_images matches between depth images and extracted vision spans
        num_depth = depth_images.shape[0]
        num_vis = selected_hidden_states.shape[0]
        if num_depth != num_vis:
            n = min(num_depth, num_vis)
            noisy_latents = noisy_latents[:n]
            latents = latents[:n]
            noise = noise[:n]
            timesteps = timesteps[:n]
            sigmas = sigmas[:n]
            selected_hidden_states = selected_hidden_states[:n]
            encoder_attention_mask = encoder_attention_mask[:n]

        # 4. SANA forward: predict velocity
        diffusion_pred = self.sana(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=self.diffusion_connector(
                self.mask_drop(selected_hidden_states)
            ),
            encoder_attention_mask=encoder_attention_mask,
        ).sample

        # 5. Compute flow-matching loss
        target = noise - latents
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme, sigmas=sigmas
        )
        dit_loss = torch.mean(
            (
                weighting.float()
                * (diffusion_pred.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            1,
        )
        dit_loss = dit_loss.mean()

        return dit_loss

    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)
