import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import nn, init
import einops
from collections import OrderedDict
from functools import partial

class QuickGELU(nn.Module):
    def forward(self, cx, x):
        return x * jax.nn.sigmoid(1.702 * x)

class NamedSeq(nn.Module):
    def __init__(self, items):
        self.items = OrderedDict(items)

    def self_named_modules(self):
        for (k, v) in self.items.items():
            yield k, v

    def forward(self, cx, x):
        for mod in self.items.values():
            x = mod(cx, x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.in_proj_weight = init.glorot_normal(d_model * 3, d_model)
        self.in_proj_bias = init.zeros(d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, cx, x):
        # x : n k c
        qkv = jnp.einsum('nkc,ic->nki', x, cx[self.in_proj_weight]) + cx[self.in_proj_bias]
        q, k, v = qkv.rearrange('n k (p h c) -> p n h k c', p=3, h=self.n_head)
        qk = jnp.einsum('nhqc,nhkc->nhqk', q, k)
        qk = jax.nn.softmax(qk / jnp.sqrt(q.shape[-1]), axis=-1)
        out = jnp.einsum('nhqk,nhkc->nhqc', qk, v)
        out = out.rearrange('n h k c -> n k (h c)')
        out = self.out_proj(cx, out)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = NamedSeq([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ])
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, cx, x):
        x = x + self.attn(cx, self.ln_1(cx, x))
        x = x + self.mlp(cx, self.ln_2(cx, x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, cx, x):
        return self.resblocks(cx, x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = init.normal(width, stddev=scale)
        self.positional_embedding = init.normal((input_resolution // patch_size) ** 2 + 1, width, stddev=scale)
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = init.normal(width, output_dim, stddev=scale)

    def forward(self, cx, x):
        x = self.conv1(cx, x)  # n h w c
        x = x.rearrange('n c h w -> n (h w) c') # n hw c
        n, _, c = x.shape
        x = jnp.concatenate([cx[self.class_embedding].broadcast_to([n, 1, c]), x], axis=1)  # n (hw + 1) c
        x = x + cx[self.positional_embedding]
        x = self.ln_pre(cx, x)
        x = self.transformer(cx, x)
        x = self.ln_post(cx, x[:, 0, :])

        if self.proj is not None:
            x = x @ cx[self.proj]

        return x