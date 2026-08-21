"""
SHIKSHA-MoE: Shared Hybrid Inference for Knowledge in STEM Hinglish ASR

Architecture components:
  - HybridExpertFFN: Small expert block (shared and routed)
  - TopKRouter: Softmax-based Top-K expert routing
  - HybridMoELayer: Shared expert (always-on) + sparse routed experts
  - HybridMoEEncoderLayer: Full encoder layer with MoE FFN
  - HybridMoSEEncoderWrapper: Wraps Whisper encoder with MoE layers

Paper: "SHIKSHA-MoE: Preventing Semantic Feature Dissociation in
       Technical Hinglish ASR via Shared-Expert Sparsity"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import BaseModelOutput


class HybridExpertFFN(nn.Module):
    """Small expert FFN block used for both shared and routed experts."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TopKRouter(nn.Module):
    """Softmax-based Top-K expert router."""

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-9
        )
        return router_probs, top_k_indices, top_k_weights


class HybridMoELayer(nn.Module):
    """
    Hybrid MoE FFN layer:
      1. Shared Expert (always active) — anchors linguistic structure
      2. Routed Experts (sparse, Top-K) — specialize for domain features

    Active neurons per token: shared_dim + top_k * expert_dim
    """

    def __init__(
        self, d_model, num_experts=4, top_k=2, expert_dim=256,
        shared_dim=256, dropout=0.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.expert_dim = expert_dim
        self.shared_dim = shared_dim

        # shared_dim=0 disables the anchor entirely, which is how the
        # no-shared-expert control is run. Previously shared_dim=0 built an
        # nn.Linear(d_model, 0) whose forward returned only fc2's bias, i.e. a
        # constant offset rather than nothing.
        self.shared_expert = (HybridExpertFFN(d_model, shared_dim, dropout)
                              if shared_dim > 0 else None)
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            HybridExpertFFN(d_model, expert_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        shared_output = (self.shared_expert(x_flat) if self.shared_expert is not None
                         else torch.zeros_like(x_flat))

        router_probs, top_k_indices, top_k_weights = self.router(x)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)

        expert_output_sum = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices_flat == expert_idx).any(dim=-1)
            if expert_mask.sum() > 0:
                expert_input = x_flat[expert_mask]
                out = self.experts[expert_idx](expert_input)
                weight_mask = (
                    top_k_indices_flat[expert_mask] == expert_idx
                )
                weights = (
                    top_k_weights_flat[expert_mask] * weight_mask.float()
                ).sum(dim=-1, keepdim=True)
                expert_output_sum[expert_mask] += weights * out

        final_output = shared_output + expert_output_sum
        return final_output.view(batch_size, seq_len, d_model), router_probs


class HybridMoEEncoderLayer(nn.Module):
    """Whisper encoder layer with MoE FFN replacing the dense FFN."""

    def __init__(self, config, num_experts=4, top_k=2, expert_dim=256,
                 shared_dim=256):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim, config.encoder_attention_heads,
            dropout=config.attention_dropout, batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.moe_ffn = HybridMoELayer(
            self.embed_dim, num_experts, top_k, expert_dim, shared_dim,
            config.activation_dropout,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=None, need_weights=False,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, router_probs = self.moe_ffn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, router_probs


class HybridMoSEEncoderWrapper(nn.Module):
    """Wraps the Whisper encoder, replacing dense layers with MoE layers."""

    def __init__(self, original_encoder, moe_layers):
        super().__init__()
        self.conv1 = original_encoder.conv1
        self.conv2 = original_encoder.conv2
        self.embed_positions = original_encoder.embed_positions
        self.moe_layers = moe_layers
        self.layer_norm = original_encoder.layer_norm
        self.dropout_layer = nn.Dropout(original_encoder.dropout)
        self.router_probs_list = []

    def forward(self, input_features, attention_mask=None, **kwargs):
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[:inputs_embeds.shape[1]]
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.dropout_layer(hidden_states)

        self.router_probs_list = []
        for moe_layer in self.moe_layers:
            hidden_states, router_probs = moe_layer(
                hidden_states, attention_mask
            )
            self.router_probs_list.append(router_probs)

        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


def compute_load_balancing_loss(router_probs, top_k=2):
    """Auxiliary load-balancing loss to prevent expert collapse."""
    num_experts = router_probs.shape[-1]
    expert_probs = router_probs.mean(dim=[0, 1])
    return num_experts * (expert_probs * expert_probs).sum()


def create_shiksha_moe(
    base_model_name="openai/whisper-tiny",
    num_experts=4,
    top_k=2,
    num_encoder_layers=4,
    num_decoder_layers=4,
    expert_dim=256,
    shared_dim=256,
):
    """
    Create a SHIKSHA-MoE model from a pre-trained Whisper checkpoint.

    Performs deterministic weight initialization over the rows of W1:
      - Shared expert: rows [0, shared_dim)
      - Routed experts: rows [shared_dim + i*expert_dim, shared_dim + (i+1)*expert_dim)
      - Rows at or above shared_dim + num_experts*expert_dim are DISCARDED.
        At the paper's defaults (shared_dim=256, num_experts=4, expert_dim=256)
        that is rows [1280, 1536), i.e. 256 of the 1536 pre-trained rows.

    Args:
        base_model_name: HuggingFace model ID (e.g. "openai/whisper-tiny")
        num_experts: Number of routed experts per encoder layer
        top_k: Number of experts activated per token
        num_encoder_layers: Encoder depth
        num_decoder_layers: Decoder depth
        expert_dim: Hidden dimension per routed expert
        shared_dim: Hidden dimension for the shared expert

    Returns:
        model: WhisperForConditionalGeneration with MoE encoder
        moe_encoder: HybridMoSEEncoderWrapper instance
    """
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name
    )
    config = base_model.config
    config.encoder_layers = num_encoder_layers
    config.decoder_layers = num_decoder_layers
    model = WhisperForConditionalGeneration(config)

    model.model.encoder.conv1.load_state_dict(
        base_model.model.encoder.conv1.state_dict()
    )
    model.model.encoder.conv2.load_state_dict(
        base_model.model.encoder.conv2.state_dict()
    )
    model.model.encoder.embed_positions.load_state_dict(
        base_model.model.encoder.embed_positions.state_dict()
    )
    model.model.decoder.load_state_dict(
        base_model.model.decoder.state_dict(), strict=False
    )
    if hasattr(base_model, "proj_out") and base_model.proj_out is not None:
        model.proj_out.load_state_dict(base_model.proj_out.state_dict())

    moe_encoder_layers = nn.ModuleList()
    base_encoder_layers = list(base_model.model.encoder.layers)

    for idx in range(num_encoder_layers):
        moe_layer = HybridMoEEncoderLayer(
            config, num_experts, top_k, expert_dim, shared_dim
        )
        base_layer = base_encoder_layers[idx]

        # Copy self-attention weights (Whisper uses separate q/k/v projections;
        # nn.MultiheadAttention uses fused in_proj)
        moe_layer.self_attn.in_proj_weight.data.copy_(torch.cat([
            base_layer.self_attn.q_proj.weight,
            base_layer.self_attn.k_proj.weight,
            base_layer.self_attn.v_proj.weight,
        ], dim=0))

        q_bias = (base_layer.self_attn.q_proj.bias
                  if base_layer.self_attn.q_proj.bias is not None
                  else torch.zeros(config.d_model))
        k_bias = (base_layer.self_attn.k_proj.bias
                  if base_layer.self_attn.k_proj.bias is not None
                  else torch.zeros(config.d_model))
        v_bias = (base_layer.self_attn.v_proj.bias
                  if base_layer.self_attn.v_proj.bias is not None
                  else torch.zeros(config.d_model))
        moe_layer.self_attn.in_proj_bias.data.copy_(
            torch.cat([q_bias, k_bias, v_bias], dim=0)
        )

        moe_layer.self_attn.out_proj.load_state_dict(
            base_layer.self_attn.out_proj.state_dict()
        )
        moe_layer.self_attn_layer_norm.load_state_dict(
            base_layer.self_attn_layer_norm.state_dict()
        )
        moe_layer.final_layer_norm.load_state_dict(
            base_layer.final_layer_norm.state_dict()
        )

        original_fc1 = base_layer.fc1.weight
        original_fc1_bias = base_layer.fc1.bias
        original_fc2 = base_layer.fc2.weight
        original_fc2_bias = base_layer.fc2.bias

        if shared_dim > 0:
            with torch.no_grad():
                moe_layer.moe_ffn.shared_expert.fc1.weight.copy_(
                    original_fc1[:shared_dim, :]
                )
                moe_layer.moe_ffn.shared_expert.fc1.bias.copy_(
                    original_fc1_bias[:shared_dim]
                )
                moe_layer.moe_ffn.shared_expert.fc2.weight.copy_(
                    original_fc2[:, :shared_dim]
                )
                moe_layer.moe_ffn.shared_expert.fc2.bias.copy_(original_fc2_bias)
        else:
            # No anchor: the first routed expert inherits W_2's output bias so the
            # summed paths still reproduce the dense bias exactly once.
            with torch.no_grad():
                moe_layer.moe_ffn.experts[0].fc2.bias.copy_(original_fc2_bias)

        start_neuron = shared_dim
        for i, expert in enumerate(moe_layer.moe_ffn.experts):
            idx_start = start_neuron + (i * expert_dim)
            idx_end = idx_start + expert_dim
            with torch.no_grad():
                expert.fc1.weight.copy_(original_fc1[idx_start:idx_end, :])
                expert.fc1.bias.copy_(original_fc1_bias[idx_start:idx_end])
                expert.fc2.weight.copy_(original_fc2[:, idx_start:idx_end])
                expert.fc2.bias.zero_()

        moe_encoder_layers.append(moe_layer)

    moe_encoder = HybridMoSEEncoderWrapper(
        model.model.encoder, moe_encoder_layers
    )
    model.model.encoder = moe_encoder

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None

    return model, moe_encoder
