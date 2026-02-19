from dataclasses import dataclass
from typing import Optional
import numpy as np

import torch
from torch import nn

from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast

from transformers.models.llama.modeling_llama import can_return_tuple
from transformers.models.llama.modeling_llama import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3PreTrainedModel


class MRM(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        
        # R(x,y) = 2 * mu_d^T Sigma_d^{-1} f_theta(x,y)
        # where mu_d = E[chosen - rejected], Sigma_d = Cov(chosen - rejected)
        hidden_size = config.hidden_size
        
        self.register_buffer('mu_chosen', torch.zeros(hidden_size))   # E[chosen_i]
        self.register_buffer('mu_rejected', torch.zeros(hidden_size))   # E[rejected_i]
        self.register_buffer('mu_d', torch.zeros(hidden_size))   # E[d_i]
        self.register_buffer('sigma_chosen', torch.eye(hidden_size))   # Σ_chosen
        self.register_buffer('sigma_chosen_inv', torch.eye(hidden_size))   # Σ_chosen^{-1}
        self.register_buffer('sigma_rejected', torch.eye(hidden_size))   # Σ_rejected
        self.register_buffer('sigma_rejected_inv', torch.eye(hidden_size))   # Σ_rejected^{-1}
        self.register_buffer('sigma_d', torch.eye(hidden_size))   # Σ_d
        self.register_buffer('sigma_d_inv', torch.eye(hidden_size))   # Σ_d^{-1}
        
        self.use_gda_reward = True
        self.const_odds = 1.0
        self.const_mahalanobis = 0.001

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_gda_params(self, mu_chosen, mu_rejected, mu_d, sigma_chosen, sigma_chosen_inv, sigma_rejected, sigma_rejected_inv, sigma_d, sigma_d_inv):
        """
        Set difference-based GDA parameters.

        Args:
            mu_d:      mean of difference vectors E[chosen - rejected]  (shape: [hidden_size])
            sigma_chosen: covariance of chosen vectors Σ_chosen (shape: [hidden_size, hidden_size])
            sigma_chosen_inv: inverse covariance of chosen vectors Σ_chosen^{-1} (shape: [hidden_size, hidden_size])
            sigma_rejected: covariance of rejected vectors Σ_rejected (shape: [hidden_size, hidden_size])
            sigma_rejected_inv: inverse covariance of rejected vectors Σ_rejected^{-1} (shape: [hidden_size, hidden_size])
            sigma_d: covariance of difference vectors Σ_d (shape: [hidden_size, hidden_size])
            sigma_d_inv: inverse covariance of difference vectors Σ_d^{-1} (shape: [hidden_size, hidden_size])
        """
        device = self.mu_chosen.device
        dtype = self.mu_d.dtype

        self.mu_chosen.copy_(torch.tensor(mu_chosen, dtype=dtype, device=device))
        self.mu_rejected.copy_(torch.tensor(mu_rejected, dtype=dtype, device=device))
        self.mu_d.copy_(torch.tensor(mu_d, dtype=dtype, device=device))
        
        self.sigma_chosen.copy_(torch.tensor(sigma_chosen, dtype=dtype, device=device))
        self.sigma_chosen_inv.copy_(torch.tensor(sigma_chosen_inv, dtype=dtype, device=device))
        
        self.sigma_rejected.copy_(torch.tensor(sigma_rejected, dtype=dtype, device=device))
        self.sigma_rejected_inv.copy_(torch.tensor(sigma_rejected_inv, dtype=dtype, device=device))
        
        self.sigma_d.copy_(torch.tensor(sigma_d, dtype=dtype, device=device))
        self.sigma_d_inv.copy_(torch.tensor(sigma_d_inv, dtype=dtype, device=device))
        
        self.use_gda_reward = True
    
    def compute_gda_reward(self, features):
        """
        Difference-based GDA reward:
        R(x,y) = 2 * mu_d^T Sigma_d^{-1} f_theta(x,y)
        
        Args:
            features: hidden representation (shape: [batch_size, hidden_size])
        
        Returns:
            rewards: (shape: [batch_size])
        """
        device = features.device
        mu_chosen = self.mu_chosen.to(device)
        mu_rejected = self.mu_rejected.to(device)
        mu_d = self.mu_d.to(device)
        
        sigma_chosen_inv = self.sigma_chosen_inv.to(device)
        sigma_rejected_inv = self.sigma_rejected_inv.to(device)
        sigma_d_inv = self.sigma_d_inv.to(device)
        
        # sigma_chosen = self.sigma_chosen.to(device)
        # sigma_rejected = self.sigma_rejected.to(device)
        # sigma_d = self.sigma_d.to(device)

        # Odds reward: r = 2 * Σ_d^{-1} μ_d
        odds_reward = 2.0 * sigma_d_inv @ mu_d  # [hidden_size]
        odds_rewards = features @ odds_reward       # [batch_size]

        # Mahalanobis reward: r = - 0.5 * (f - μ_chosen)^T Σ_chosen^{-1} (f - μ_chosen) + 0.5 * (f - μ_rejected)^T Σ_rejected^{-1} (f - μ_rejected)
        diff_chosen = features - mu_chosen  # [batch_size, hidden_size]
        diff_rejected = features - mu_rejected  # [batch_size, hidden_size]
        mahalanobis_reward = -0.5 * np.sum(diff_chosen @ sigma_chosen_inv * diff_chosen, axis=1) + 0.5 * np.sum(diff_rejected @ sigma_rejected_inv * diff_rejected, axis=1)  # [batch_size]
        
        rewards = self.const_odds * odds_rewards + self.const_mahalanobis * mahalanobis_reward

        return rewards

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)
        
        if self.use_gda_reward:
            # Extract hidden features at the last non-pad token position
            # hidden_states shape: [batch_size, seq_len, hidden_size]
            pooled_features = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_non_pad_token]
            
            # Compute GDA-based reward: R(x,y) = (mu_+ - mu_-)^T Sigma^{-1} f_theta(x,y) + b
            final_logits = self.compute_gda_reward(pooled_features)
        else:
            # Use standard score layer output
            final_logits = pooled_logits

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )