import os
import jax
import jax.numpy as jnp
import jaxtorch
import clip
import torch

from . import vit

_MODELS = {
    "ViT-B/32": vit.VITB32,
    "ViT-B/16": vit.VITB16,
    "ViT-L/14": vit.VITL14,
    "ViT-L/14@336px": vit.VITL14_336,
}

def load(name: str, download_root: str = None):
    if name not in _MODELS:
        raise RuntimeError(f"Model {name} not found; available models = {_MODELS.keys()}")

    model_path = clip.clip._download(clip.clip._MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))

    model = _MODELS[name]()
    model.name_everything_()

    try:
        # loading JIT archive
        torch_model = torch.jit.load(model_path, map_location="cpu").eval()
        params = {k:jnp.array(v.cpu().numpy()) for (k,v) in torch_model.state_dict().items()}
    except RuntimeError:
        params = jaxtorch.pt.load(model_path)
    return model, params

tokenize = clip.tokenize
