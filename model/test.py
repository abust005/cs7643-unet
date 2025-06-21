from unet import UNet
import torch
from time import time
x = torch.randn(8, 3, 572, 572, dtype=torch.float32)

cpu_net = UNet().to(dtype=torch.float32)
cuda_net = UNet().to(device='cuda', dtype=torch.float32)

start_time = time()
y1 = cpu_net(x)
end_time = time()

cpu_time = end_time - start_time

start_time = time()
y2 = cuda_net(x.to(device='cuda'))
end_time = time()

cuda_time = end_time - start_time

print(f'CPU Time: {cpu_time}s')
print(f'CUDA Time: {cuda_time}s')