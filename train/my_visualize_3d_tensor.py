"""
 @Time    : 31.10.22 09:48
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_visualize_3d_tensor.py
 @Function:
 
"""
import matplotlib.pyplot as plt
import numpy as np

# axes = [16, 16, 16] # change to 64
# traj = np.random.choice([-1,1], axes)
# print(traj)
traj = np.random.rand(32, 24, 96)
print(traj)

# alpha = 0.9
# colors = np.empty(axes + [4], dtype=np.float32)
# colors[traj==1] = [1, 0, 0, alpha]  # red
# colors[traj==-1] = [0, 0, 1, alpha]  # blue

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(traj, facecolors=None, edgecolors='black')
ax.voxels(traj, edgecolors='black')
plt.show()
