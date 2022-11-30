import math

import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, einsum
from torch.quantization import quantize_dynamic

from lamda_pytorch.config.config import CFG

# residual wrapper

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# pre-normalization wrapper

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# gated-GELU activation function

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
        
class SquaredRelu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.relu(x) ** 2

# feedforward layer with gated-GELU activation function

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            SquaredRelu(),
            nn.Dropout(dropout), # optional dropout
            nn.Linear(inner_dim * 2, dim)
        )

    def forward(self, x):
        return self.net(x)

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)

    def forward(self, x):
        h, device = self.heads, x.device

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        # T5 Relative Positional Bias
        sim = self.rel_pos_bias(sim)

        # Causal Mask
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn) # Optional dropout

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0., quantized = False):
        super().__init__()
        self.layers = nn.ModuleList([])
   
        attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout)
        ffn = FeedForward(dim = dim, dropout = dropout)
        #if quantized = True, apply dynamic quantization to Linears in both attention and FFN
        if quantized:
            attn = quantize_dynamic(attn, {nn.Linear}, dtype=torch.qint8)
            ffn = quantize_dynamic(ffn, {nn.Linear}, dtype=torch.qint8)
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, attn)),
                Residual(PreNorm(dim, ffn))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# LaMDA Model

class LaMDA(nn.Module):
    def __init__(self, *, num_tokens, dim, depth, dim_head, heads, quantized_transformer = False):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(dim, depth, dim_head, heads, quantized = quantized_transformer)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        x = self.token_emb(x)
        x = self.transformer(x)
        logits = self.to_logits(x)
        return logits

def lamda_model(quantized_logits = False, quantized_transformer = False): # note: quantized logits does not automatically guarantee a quantized transformer, quantized lamda refers to quantizing the linear layer in LaMDA.to_logits and that only
    model = LaMDA(
        num_tokens = CFG.num_tokens,
        dim = CFG.dim,
        depth = CFG.depth,
        dim_head = CFG.dim_head,
        heads = CFG.heads,
        quantized_transformer = quantized_transformer
    )
    if quantized_logits:
        quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        del model # save on memory
        return quantized_model
    else: # technically this doesn't need an else statement but I feel it looks nicer like this
        return model

if __name__ == "__main__":

    lamda_base = lamda_model()

    # lamda = AutoregressiveWrapper(lamda_base, max_seq_len = 2048)

    tokens = torch.randint(0, lamda_base.num_tokens, (1, lamda_base.dim)) # mock token data

    logits = lamda_base(tokens)
    print(logits.shape)

    n_params_torch = sum(
        p.numel() for p in lamda_base.parameters() if p.requires_grad
    )

    print(f"Number of parameters in torch model: {n_params_torch}")