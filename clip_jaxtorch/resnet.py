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

def causal(cx, qk):
    # qk : nhqk
    [n, h, q, k] = qk.shape
    mask = jnp.full([q,k], -float('inf'))
    mask = jnp.triu(mask,1)
    return qk + mask


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.n_head = n_head
        self.in_proj_weight = init.glorot_normal(d_model * 3, d_model)
        self.in_proj_bias = init.zeros(d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_mask = attn_mask

    def forward(self, cx, x):
        # x : n k c
        qkv = jnp.einsum('nkc,ic->nki', x, cx[self.in_proj_weight]) + cx[self.in_proj_bias]
        q, k, v = qkv.rearrange('n k (p h c) -> p n h k c', p=3, h=self.n_head)
        qk = jnp.einsum('nhqc,nhkc->nhqk', q, k) / jnp.sqrt(q.shape[-1])
        if self.attn_mask is not None:
            qk = self.attn_mask(cx, qk)
        qk = jax.nn.softmax(qk, axis=-1)
        out = jnp.einsum('nhqk,nhkc->nhqc', qk, v)
        out = out.rearrange('n h k c -> n k (h c)')
        out = self.out_proj(cx, out)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head, attn_mask)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = NamedSeq([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ])
        self.ln_2 = nn.LayerNorm(d_model)

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

class BatchNorm2d(nn.Module):
    def __init__(self, dim):
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)

        # Implement the buffers as just frozen weights
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        self.num_batches_tracked = init.zeros()

    def forward(self, cx, x):
        eps = 1e-5
        mean = jax.lax.stop_gradient(cx[self.running_mean])
        std = jax.lax.stop_gradient(jnp.sqrt(cx[self.running_var] + eps))
        x = (x - mean[:,None,None]) / std[:,None,None]
        x = x * cx[self.weight][:,None,None] + cx[self.bias][:,None,None]
        return x

class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm2d(width)
        self.relu3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

@jax.vmap
def gather_bi(x, i):
    return x[i]

class CLIPText(nn.Module):
    def __init__(self, n_dim=512, n_layers=12, n_heads=8, d_out=512):
        super().__init__()
        self.transformer = Transformer(n_dim, n_layers, heads=n_heads, attn_mask=causal)
        self.token_embedding = nn.Embedding(49408, n_dim)
        self.ln_final = nn.LayerNorm(n_dim)
        self.positional_embedding = init.normal(77, n_dim)
        self.text_projection = init.normal(n_dim, d_out)
        self.logit_scale = init.ones()

    def encode_text(self, cx, text):
        x = self.token_embedding(cx, text)  # [batch_size, n_ctx, d_model]

        x = x + cx[self.positional_embedding]
        x = self.transformer(cx, x)
        x = self.ln_final(cx, x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = gather_bi(x,jnp.argmax(text,axis=1)) @ cx[self.text_projection]

        return x

class VITB32(CLIPText):
    def __init__(self):
        super().__init__(512, 12, 8, 512)
        self.visual = VisionTransformer(224, 32, 768, 12, heads=12, output_dim=512)

class VITL14(CLIPText):
    def __init__(self):
        super().__init__(768, 12, 768//64, 768)
        self.visual = VisionTransformer(224, 14, 1024, 24, heads=1024//64, output_dim=768)
