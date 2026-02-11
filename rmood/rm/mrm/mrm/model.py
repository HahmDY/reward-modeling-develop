from dataclasses import dataclass
from typing import Optional

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
        
        # R(x,y) = (mu_+ - mu_-)^T Sigma^{-1} f_theta(x,y) + b
        hidden_size = config.hidden_size
        self.register_buffer('mu_pos', torch.zeros(hidden_size))  # μ_+
        self.register_buffer('mu_neg', torch.zeros(hidden_size))  # μ_-
        self.register_buffer('sigma_inv', torch.eye(hidden_size))  # Σ^{-1}
        self.register_buffer('bias', torch.tensor(0.0))  # b
        self.use_gda_reward = True

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_gda_params(self, mu_pos, mu_neg, sigma_inv, bias=None):
        """
        GDA 파라미터 설정
        
        Args:
            mu_pos: positive 샘플들의 평균 벡터 (shape: [hidden_size])
            mu_neg: negative 샘플들의 평균 벡터 (shape: [hidden_size])
            sigma_inv: 공분산 행렬의 역행렬 (shape: [hidden_size, hidden_size])
            bias: bias 항 (optional, 자동 계산 가능)
        """
        device = self.mu_pos.device
        dtype = self.mu_pos.dtype
        
        self.mu_pos = torch.tensor(mu_pos, dtype=dtype, device=device)
        self.mu_neg = torch.tensor(mu_neg, dtype=dtype, device=device)
        self.sigma_inv = torch.tensor(sigma_inv, dtype=dtype, device=device)
        
        if bias is None:
            # b = 0.5 * (mu_-^T Sigma^{-1} mu_- - mu_+^T Sigma^{-1} mu_+)
            bias = 0.5 * (
                self.mu_neg @ self.sigma_inv @ self.mu_neg -
                self.mu_pos @ self.sigma_inv @ self.mu_pos
            )
        
        self.bias = torch.tensor(bias, dtype=dtype, device=device)
        self.use_gda_reward = True
    
    def compute_gda_reward(self, features):
        """
        GDA based reward calculation
        R(x,y) = (mu_+ - mu_-)^T Sigma^{-1} f_theta(x,y) + b
        
        Args:
            features: model's hidden representation (shape: [batch_size, hidden_size])
        
        Returns:
            rewards: (shape: [batch_size])
        """
        # (mu_+ - mu_-)^T Sigma^{-1}
        mu_diff = self.mu_pos - self.mu_neg  # [hidden_size]
        weight = mu_diff @ self.sigma_inv  # [hidden_size]
        
        # weight^T f_theta(x,y) + b
        rewards = features @ weight + self.bias  # [batch_size]
        
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