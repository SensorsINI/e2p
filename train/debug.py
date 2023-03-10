"""
 @Time    : 19.03.22 18:19
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : debug.py
 @Function:
 
"""
import torch

x = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])
print(x)
print(x.shape)

y = x.unsqueeze(0).unsqueeze(0)
print(y)
print(y.shape)

z = y.view(1, 1, 12)
print(z)
print(z.shape)

w = z.view(1, 1, 4, 3)
print(w)
print(w.shape)