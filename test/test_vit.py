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
@torch.no_grad()
def test_vit(model_name):
    rng = PRNG(jax.random.PRNGKey(1))

    clip_model, _ = torch_clip.load(model_name, device='cpu')
    jax_model, params = jax_clip.load(model_name)
    params = tree_map(lambda x: x.astype(jnp.float32), params)

    size = jax_model.visual.input_resolution

    image = jax.random.normal(rng.split(), [1, 3, size, size])
    out_jax = jax_model.visual(Context(params, None), image)
    out_torch = fromtorch(clip_model.visual(fromjax(image)))
    assert jnp.allclose(out_torch, out_jax, atol=0.002, rtol=0.001)

    image = jnp.zeros([1, 3, size, size])
    out_jax = jax_model.visual(Context(params, None), image)
    out_torch = fromtorch(clip_model.visual(fromjax(image)))
    assert jnp.allclose(out_torch, out_jax, atol=0.002, rtol=0.001)

    text = fromtorch(torch_clip.tokenize("hello world"))
    out_jax = jax_model.encode_text(Context(params, None), text)
    out_torch = fromtorch(clip_model.encode_text(fromjax(text)))
    assert jnp.allclose(out_torch, out_jax, atol=0.002, rtol=0.001)
