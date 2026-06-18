import os
import time
import numpy as np
import torch
from models.EDAFNet import EDAFNet
import pynvml
from thop import profile

img_size = (1000, 1504)
in_chans = 18
window_size = 8
embed_dim = 72
num_heads = [4]
depths = [5]
mlp_ratio = 2

model = EDAFNet(
    img_size=img_size,
    in_chans=in_chans,
    window_size=window_size,
    img_range=1.,
    drop_path_rate=0.1,
    depths=depths,
    embed_dim=embed_dim,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio
).cuda().half().eval()

# Dummy input para EDAFNet (1 batch, 18 canais, 1000x1504)
img = torch.randn(1, in_chans, img_size[0], img_size[1]).cuda()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpuName = pynvml.nvmlDeviceGetName(handle)
print(gpuName)

with torch.no_grad():
    for i in range(10):
        out = model(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(100):
        out = model(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('Time: {:.3f}s'.format((time.time() - time_stamp) / 100))

flops, params = profile(model, inputs=(img,), verbose=False)
print('FLOPs: {:.3f}T, Params: {:.2f}k'.format(flops / 1e12, params / 1e3))
