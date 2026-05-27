from typing import Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt_runner import RDTRunner


class RDTDistillRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin,
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    """
    Runner class for diffusion distillation with regression loss
    """
    def __init__(
        self,
        rdt_config: dict,
        dtype = torch.bfloat16
    ):
        """
        Args:
            rdt_config: a dictionary containing the configuration for the RDT model.
            dtype: the data type for the model, default is torch.bfloat16.
        """
        super(RDTDistillRunner, self).__init__()

        # Initialize the three networks from the pretrained denoiser
        self.real_score = RDTRunner(**rdt_config)
        self.generator = RDTRunner(**rdt_config)
        # Remove the gradient computation for the real score model
        self.real_score.requires_grad_(False)

        self.loss_dict = defaultdict(float)
        self.dtype = dtype

    def initialize_from_pretrained_denoiser(
        self,
        pretrained_denoiser: RDTRunner,
    ):
        """
        Initialize the runner from a pre-trained denoiser network.

        Args:
            pretrained_denoiser: a pre-trained RDT denoiser network.
        """
        if not isinstance(pretrained_denoiser, RDTRunner):
            raise TypeError(
                f"Expected pretrained_denoiser to be an instance of RDTRunner, "
                f"but got {type(pretrained_denoiser)}"
            )

        self.real_score.load_state_dict(pretrained_denoiser.state_dict())
        self.generator.load_state_dict(pretrained_denoiser.state_dict())
        self.real_score.requires_grad_(False)

    # ========= Inference  ============
    @torch.no_grad()
    def predict_action(self, 
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None) :
        return self.predict_action_wg(lang_tokens, lang_kv_cache, lang_attn_mask, img_tokens, state_tokens, noisy_action)
    
    # _wg means with gradient
    def predict_action_wg(
        self,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None,
    ):
        '''
        Args:
            lang_tokens: (batch_size, lang_len, lang_token_dim)
            lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
                which should be True-False bool tensor.
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, state_history_len, hidden_size),
                state conditional data, if available.
            noisy_action: (batch_size, horizon, action_dim), optional,
                if provided, will be used as the initial noisy action for sampling.
                If not provided, a random noise will be generated.

        Returns:
            (batch_size, horizon, action_dim), predicted action sequence
        '''
        batch_size = state_tokens.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype
        zero_timesteps = torch.zeros(
            (batch_size,), dtype=dtype, device=device
        )

        # Sample a noise if not provided
        if noisy_action is None:
            noisy_action = torch.randn(
                size=(batch_size,
                    self.generator.pred_horizon,
                    self.generator.action_dim),
                dtype=dtype, device=device)

        # Generate action w.r.t. noise
        v_pred = self.generator.predict_velocity(
            lang_tokens=lang_tokens,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            noisy_action=noisy_action,
            timesteps=zero_timesteps
        )
        action_pred = noisy_action + v_pred

        return action_pred

    # ========= Train  ============
    def compute_loss(
            self,
            img_tokens: torch.Tensor,
            state_tokens: torch.Tensor,
            action_gt: torch.Tensor,
            lang_kv_cache: Optional[torch.Tensor] = None,
            lang_tokens: Optional[torch.Tensor] = None,
            lang_attn_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        '''
        Args:
            lang_tokens: (batch_size, lang_len, lang_token_dim)
            lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
                which should be True-False bool tensor.
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, 1, state_dim)
            action_gt: (batch_size, horizon, action_dim), ground-truth actions for supervision

        Returns:
            loss_value, a scalar tensor
        '''
        batch_size = action_gt.shape[0]
        dtype = action_gt.dtype
        device = action_gt.device
        zero_timesteps = torch.zeros(
            (batch_size,), dtype=dtype, device=device
        )
        condition = {
            "lang_tokens": lang_tokens,
            "lang_kv_cache": lang_kv_cache,
            "lang_attn_mask": lang_attn_mask,
            "img_tokens": img_tokens,
            "state_tokens": state_tokens
        }
        
        # Sample by the real score model
        noise = torch.randn(
            action_gt.shape, dtype=dtype, device=device
        )
        # Generate reference action
        action_real = self.real_score.predict_action(
            **condition,
            noisy_action=noise
        ).detach()

        # Generate distilled action
        v_generated = self.generator.predict_velocity(
            **condition,
            noisy_action=noise,
            timesteps=zero_timesteps
        )
        action_generated = noise + v_generated
        
        loss = F.mse_loss(
            action_generated, action_real, reduction='mean'
        )
        
        # Update the loss dictionary
        self.loss_dict['loss'] = loss.item()
        self.loss_dict['action_real_gt_loss'] = F.mse_loss(
            action_gt, action_real, reduction='mean'
        ).item()
        
        # Regression loss
        return loss

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
    
    def get_loss_dict(self) -> Dict[str, float]:
        """
        Returns the loss dictionary containing the losses computed during training.
        """
        return self.loss_dict
