import torch

x = torch.load("data/processed/voxels/test__armadillo.pt")
print(x.shape)
print(x.sum())

import glob; print(len(glob.glob("data/processed/voxels/*.pt")))