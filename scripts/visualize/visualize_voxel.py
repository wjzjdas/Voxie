#python scripts/visualize_voxel.py data/processed/voxels/test__cow.pt <-- Change this 

import torch
import matplotlib.pyplot as plt
import sys

vox = torch.load(sys.argv[1]).numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x, y, z = vox.nonzero()
ax.scatter(x, y, z, s=2)

ax.set_box_aspect([1,1,1])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(0,32)
ax.set_ylim(0,32)
ax.set_zlim(0,32)

plt.show()

