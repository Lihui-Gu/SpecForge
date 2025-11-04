import torch
import torch.nn as nn
from specforge.kernel import act_quant, fp8_gemm, fp8_index
from specforge.utils import print_with_rank, precompute_freqs_cis, apply_rotary_emb

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
        self.num_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.hidden_size
        self.wq_b = Linear(self.q_lora_rank, self.num_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(self.dim, self.num_heads, dtype=torch.get_default_dtype())
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = None
        self.block_size = config.index_block_size
        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)

        # self.register_buffer("k_cache", torch.zeros(config.max_batch_size, config.max_seq_len, config.index_head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        # self.register_buffer("k_scale_cache", torch.zeros(config.max_batch_size, config.max_seq_len, config.index_head_dim // config.index_block_size, dtype=torch.float32), persistent=False)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_emb: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ):
        bsz, seq_len, _ = hidden_states.shape
        start_pos = 0
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]

        bsz, seqlen, _ = hidden_states.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(input_emb)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, self.block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, self.block_size, self.scale_fmt)

        # self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        # self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(hidden_states) * self.num_heads ** -0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale

        index_score = fp8_index(q_fp8.contiguous(), weights, k_fp8.contiguous(), k_scale.contiguous())
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        # topk_indices_ = topk_indices.clone()
        # dist.broadcast(topk_indices_, src=0)
        # assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        index_mask = torch.full((bsz, seq_len, seq_len), float("-inf"), device=hidden_states.device).scatter_(-1, topk_indices, 0).unsqueeze(1) # （bsz, 1, sql_len, sql_len）
        return index_mask