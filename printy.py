import sys
sys.path.append('../CLIP')

import clip

a, b = clip.load('ViT-L/14')
print(a)

# print(a.visual.transformer.resblocks[0].attn.__dict__)

for name, par in a.named_parameters(): #.visual.transformer.resblocks[0].attn.named_parameters():
    print(name, par.shape)