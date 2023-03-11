import torch
from torch.backends import cudnn

print(torch.cuda.is_available())

print(cudnn.is_available())

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())