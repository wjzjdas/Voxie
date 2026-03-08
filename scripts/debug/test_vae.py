#RUN AS: python -m scripts.test_vae

import torch
from torch.utils.data import DataLoader

from voxie.data.voxel_dataset import VoxelDataset
from voxie.models.vae3d import VoxelVAE3D, vae_loss


def main():
    dataset = VoxelDataset(
        index_path="data/processed_pointcloud/index.jsonl",
        split="train",
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    x = batch["voxel"]              # [B, 1, 32, 32, 32]
    category_idx = batch["category_idx"]  # [B]

    num_categories = len(dataset.category_to_idx)

    model = VoxelVAE3D(
        num_categories=num_categories,
        latent_dim=128,
        category_embed_dim=32,
    )

    logits, mu, logvar = model(x, category_idx)

    print("input shape:", x.shape)
    print("logits shape:", logits.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

    loss, stats = vae_loss(logits, x, mu, logvar, beta=1.0)
    print("loss stats:", stats)


if __name__ == "__main__":
    main()