import sys
sys.path.append('../CLIP')

import clip

a, b = clip.load('ViT-B/32')
print(a.visual)

print(a.visual.transformer.resblocks[0].attn.__dict__)

for name, par in a.visual.transformer.resblocks[0].attn.named_parameters():
    print(name, par.shape)