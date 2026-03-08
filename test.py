import torch, glob

for f in glob.glob("data/processed/voxels/*.pt")[:10]:
    vox = torch.load(f)
    coords = vox.nonzero()

    size = coords.max(0).values - coords.min(0).values
    print(f, size.tolist())