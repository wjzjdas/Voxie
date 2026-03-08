#RUN WITH: python -m scripts.test_baseline

from voxie.data.voxel_dataset import VoxelDataset
from voxie.baselines.random_retrieval import RandomRetrievalBaseline

dataset = VoxelDataset(
    index_path="data/processed_pointcloud/index.jsonl"
)

baseline = RandomRetrievalBaseline(dataset, split="train", seed=42)

sample = baseline.sample_by_category("Knife")

print("baseline sample id:", sample["id"])
print("category:", sample["category"])
print("category idx:", sample["category_idx"])
print("voxel shape:", sample["voxel"].shape)
print("occupied voxels:", int(sample["voxel"].sum().item()))