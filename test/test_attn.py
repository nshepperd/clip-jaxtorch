import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import nn, init, PRNG, Context
import einops
import torch
import numpy as np
from jax.tree_util import tree_map

from clip_jax_simple import vit

import numpy.testing._private.utils as nputil

def fromjax(x):
    return tree_map(lambda x: torch.tensor(np.array(x)), x)
def fromtorch(x):
    return tree_map(lambda x: jnp.array(x), x)

@torch.no_grad()
def test_attn():
    n_batch = 3
    d_model = 8
    n_head = 2

    rng = PRNG(jax.random.PRNGKey(1))
    jax_mod = vit.MultiheadAttention(d_model, n_head)
    params = jax_mod.init_weights(rng.split())

    torch_mod = torch.nn.MultiheadAttention(d_model, n_head, batch_first=True)
    torch_mod.load_state_dict(fromjax(jax_mod.state_dict(params)))

    x = jax.random.normal(rng.split(), [n_batch, 4, d_model])

    def f(params, x):
        cx = Context(params, None)
        return jax_mod(cx, x)

    out = f(params, x)
    y = fromjax(x)
    out_base = fromtorch(torch_mod(y, y, y, need_weights=False, attn_mask=None))[0]
    assert jnp.allclose(out, out_base, atol=1e-5)