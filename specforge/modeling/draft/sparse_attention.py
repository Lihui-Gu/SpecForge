from typing import List, Optional, Tuple
import math
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from specforge.kernel import act_quant, fp8_gemm, fp8_index
from specforge.utils import print_with_rank
from specforge.modeling.utils import precompute_freqs_cis, apply_rotary_emb, repeat_kv

class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        return F.layer_norm(hidden_states, (self.dim,), self.weight.to(torch.float32), self.bias.to(torch.float32), self.eps).to(input_dtype)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
        scale_fmt (Optional[str]): The format of scaling factors.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    assert bias is None
    if weight.dtype != torch.float8_e4m3fn:
        return F.linear(x, weight)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        return fp8_gemm(x, scale, weight, weight.scale)


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        with torch.no_grad():
            # 使用Kaiming初始化
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class Indexer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim: int = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.num_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.wq_b = Linear(config.head_dim, self.num_heads * self.head_dim)
        self.wk = Linear(config.head_dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(config.head_dim, self.num_heads, dtype=torch.get_default_dtype())
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = None
        self.block_size = config.index_block_size
        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)

    def forward(
        self, 
        query_states: torch.Tensor, 
        key_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]],
        past_key_values: Optional[Cache],
        mask: Optional[torch.Tensor],
        use_cache: bool = False
    ):
        bsz, _, seq_len, _ = query_states.shape
        past_seen_tokens = (
            past_key_values.get_seq_length(layer_idx=1) if past_key_values is not None else 0
        )
        end_pos = past_seen_tokens + seq_len

        _, _, q_len, kv_len = mask.shape
        q = self.wq_b(query_states)
        q = rearrange(q, 'b n s (h d) -> (b n) s h d', n=self.num_attention_heads, d=self.head_dim)
        
        k = repeat_kv(key_states, self.num_key_value_groups)
        k = rearrange(k, 'b n s d -> (b n) s d', n=self.num_attention_heads)
        k = self.wk(k)
        k = self.k_norm(k) # (bsz, sql, index_head_num)

        q_fp8, q_scale = act_quant(q, self.block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, self.block_size, self.scale_fmt)

        if use_cache and past_key_values is not None:
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + q_len, device=query_states.device
            )
            cache_kwargs = {"cache_position": cache_position}
            k_fp8_cache, _ = past_key_values.update(
                k_fp8,
                torch.empty_like(k_fp8),
                layer_idx=1,  # TODO: support multiple layers
                cache_kwargs=cache_kwargs,
            )
            k_scale_cache, _ = past_key_values.update(
                k_scale,
                torch.empty_like(k_scale),
                layer_idx=2,  # TODO: support multiple layers
                cache_kwargs=cache_kwargs,
            )
        else:
            k_fp8_cache, k_scale_cache = k_fp8, k_scale

        weights = self.weights_proj(query_states) * self.num_heads ** -0.5
        weights = rearrange(weights, 'b n s d -> (b n) s d', n=self.num_attention_heads)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        
        index_score = fp8_index(q_fp8.contiguous(), weights, k_fp8_cache.contiguous(), k_scale_cache.contiguous()) # (bsz, q_len, kv_len)
        index_score = rearrange(index_score, '(b n) s d -> b n s d', n=self.num_attention_heads)

        if mask is not None: # (bsz, 1, q_len, kv_len)
            index_score += mask

        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
            
        index_mask = torch.full(
            (bsz, self.num_attention_heads, q_len, kv_len),
            float("-inf"),
            device=query_states.device
        ).scatter_(-1, topk_indices, 0)

        return index_mask, topk_indices