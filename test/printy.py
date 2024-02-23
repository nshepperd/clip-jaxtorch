import sys
sys.path.append('../CLIP')

import clip

# a, b = clip.load('RN50x4')
# print(a)

# # print(a.visual.transformer.resblocks[0].attn.__dict__)

# for name, par in a.named_parameters(): #.visual.transformer.resblocks[0].attn.named_parameters():
#     print(name, par.shape)

# print('Buffers:')
# for name, par in a.named_buffers():
#     print(name, par.shape)


a, b = clip.load('ViT-L/14@336px')
print('visual:n_layers=', len(a.visual.transformer.resblocks))
print('visual:n_heads=', a.visual.transformer.resblocks[0].attn.num_heads)
print('visual:patch_size=', a.visual.conv1)
print('visual:width=', a.visual.proj.shape[0])
print('visual:output_dim=', a.visual.proj.shape[1])

print('d_out', a.text_projection.shape[1])
print('n_dim', a.text_projection.shape[0])
print('n_layers', len(a.transformer.resblocks))
print('n_heads', a.transformer.resblocks[0].attn.num_heads)
# print(a.visual.transformer.resblocks[0].attn.named_parameters())
# for name, par in a.named_parameters():
#     print(name, par.shape)
