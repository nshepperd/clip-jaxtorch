import sys
sys.path.append('../CLIP')


import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import nn, init, PRNG, Context
import einops
import numpy as np
from jax.tree_util import tree_map
import torch
import clip

from clip_jax_simple import vit

import numpy.testing._private.utils as nputil

def fromjax(x):
    return tree_map(lambda x: torch.tensor(np.array(x)).cuda(), x)
def fromtorch(x):
    return tree_map(lambda x: jnp.array(x.cpu().numpy()), x)

def norm1(x):
    return x / x.square().sum(axis=-1,keepdims=True).sqrt()

@torch.no_grad()
def test_vit():
    rng = PRNG(jax.random.PRNGKey(1))

    clip_model, _ = clip.load('ViT-B/32')

    jax_model = vit.VisionTransformer(224, 32, 768, 12, heads=12, output_dim=512)
    params = jax_model.init_weights(rng.split())

    jax_model.load_state_dict(params, tree_map(lambda x: jnp.array(x.cpu().numpy()).astype(jnp.float32), clip_model.visual.state_dict()))

    all_good = True
    for (k, v) in clip_model.visual.named_parameters():
        if k in params and params[k].shape == v.shape:
            # print(f'{k} ok, {v.shape}')
            pass
        elif k in params:
            print(f'{k} -> {v.shape} vs {params[k].shape}')
            all_good = False
        else:
            print(f'{k} missing!')
            all_good = False
    assert all_good

    image = jax.random.normal(rng.split(), [1, 3, 224, 224])
    out_jax = norm1(jax_model(Context(params, None), image))
    out_torch = norm1(fromtorch(clip_model.visual(fromjax(image).half())))
    print((out_jax * out_torch).sum())
    print(jnp.sort((out_jax - out_torch).abs(), axis=1))
    assert jnp.allclose(out_torch, out_jax, atol=0.001, rtol=0.001)

    image = jnp.zeros([1, 3, 224, 224])
    out_jax = norm1(jax_model(Context(params, None), image))
    out_torch = norm1(fromtorch(clip_model.visual(fromjax(image).half())))
    print((out_jax * out_torch).sum())
    print(jnp.sort((out_jax - out_torch).abs(), axis=1))
    assert jnp.allclose(out_torch, out_jax, atol=0.001, rtol=0.001)