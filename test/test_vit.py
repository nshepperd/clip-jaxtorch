import sys

import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import nn, init, PRNG, Context
import einops
import numpy as np
from jax.tree_util import tree_map
import torch
import clip as torch_clip
import pytest

from clip_jaxtorch import vit
from clip_jaxtorch import clip as jax_clip

def fromjax(x):
    return tree_map(lambda x: torch.tensor(np.array(x)), x)
def fromtorch(x):
    return tree_map(lambda x: jnp.array(x.cpu().numpy()), x)

def norm1(x):
    return x / x.square().sum(axis=-1,keepdims=True).sqrt()

@pytest.mark.parametrize('model_name', ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
@pytest.mark.parametrize('flash', [False, True])
@torch.no_grad()
def test_vit(model_name, flash):
    vit.use_flash_attention = flash

    rng = PRNG(jax.random.PRNGKey(1))

    clip_model, _ = torch_clip.load(model_name, device='cpu')
    jax_model, params = jax_clip.load(model_name)

    if flash:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    params = tree_map(lambda x: x.astype(dtype), params)

    size = jax_model.visual.input_resolution

    if flash:
        # Error with float16 is a little higher
        tol = dict(atol=0.007, rtol=0.003)
    else:
        tol = dict(atol=0.002, rtol=0.001)

    image = jax.random.normal(rng.split(), [1, 3, size, size])
    out_jax = jax_model.visual(Context(params, None), image.astype(dtype))
    out_torch = fromtorch(clip_model.visual(fromjax(image)))
    assert jnp.allclose(out_torch, out_jax, **tol), ((out_torch-out_jax).abs().max(), out_torch.abs().max())

    image = jnp.zeros([1, 3, size, size])
    out_jax = jax_model.visual(Context(params, None), image.astype(dtype))
    out_torch = fromtorch(clip_model.visual(fromjax(image)))
    assert jnp.allclose(out_torch, out_jax, **tol), ((out_torch-out_jax).abs().max(), out_torch.abs().max())

    text = fromtorch(torch_clip.tokenize("hello world"))
    out_jax = jax_model.encode_text(Context(params, None), text)
    out_torch = fromtorch(clip_model.encode_text(fromjax(text)))
    assert jnp.allclose(out_torch, out_jax, **tol), ((out_torch-out_jax).abs().max(), ((out_torch-out_jax).abs().max()/out_torch.abs().max()))

if __name__ == '__main__':
    key = jax.random.PRNGKey(1)
    model_name = 'ViT-B/32'
    # clip_model, _ = torch_clip.load(model_name, device='cpu')
    jax_model, params_f32 = jax_clip.load(model_name)
    params_f32 = tree_map(lambda x: x.astype(jnp.float32), params_f32)
    params_f16 = tree_map(lambda x: x.astype(jnp.float16), params_f32)

    from functools import partial, wraps

    def capture(name, fwd):
        @wraps(fwd)
        def forward(cx, *args, **kwargs):
            out = fwd(cx, *args, **kwargs)
            cx.tmp['out'][name] = out
            return out
        return forward

    for mod in jax_model.modules():
        mod.forward = capture(mod.name, mod.forward)

    vit.use_flash_attention = True

    @jax.jit
    def compare(params_f32, params_f16, text):
        out32 = {}
        out16 = {}

        cx = Context(params_f32, None)
        cx.tmp['out'] = out32
        jax_model.encode_text(cx, text)#image.astype(jnp.float32))
        cx = Context(params_f16, None)
        cx.tmp['out'] = out16
        jax_model.encode_text(cx, text)#image.astype(jnp.float16))

        rtol = {}
        for name in out32.keys():
            rtol[name] = (out16[name] - out32[name]).max()/out32[name].max()
        return rtol

    size = jax_model.visual.input_resolution
    text = fromtorch(torch_clip.tokenize("hello world"))
    # image = jax.random.normal(key, [1, 3, size, size])
    rtol = compare(params_f32, params_f16, text)
    for mod in jax_model.modules():
        if mod.name in rtol.keys():
            print(mod.name, rtol[mod.name])
