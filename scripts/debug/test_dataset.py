#python -m scripts.debug.test_dataset

from torch.utils.data import DataLoader
from voxie.data.voxel_dataset import VoxelDataset

train_dataset = VoxelDataset(
    index_path="data/processed_pointcloud/index.jsonl",
    split="train"
)

print("train size:", len(train_dataset))
sample = train_dataset[0]
print(sample["id"])
print(sample["category"])
print(sample["category_idx"])
print(sample["voxel"].shape)
print(sample["voxel"].dtype)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

batch = next(iter(train_loader))
print(batch["voxel"].shape)       # [8, 1, 32, 32, 32]
print(batch["category_idx"].shape)  # [8]