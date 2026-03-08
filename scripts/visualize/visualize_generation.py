import argparse
import random

import torch
import matplotlib.pyplot as plt

from voxie.data.voxel_dataset import VoxelDataset
from voxie.models.vae3d import VoxelVAE3D
from voxie.baselines.random_retrieval import RandomRetrievalBaseline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None, help="Category to visualize, e.g. Chair")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voxel threshold")
    parser.add_argument("--sample_mode", type=str, choices=["fixed", "random"], default="fixed")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plot_voxel(ax, vox, title):
    coords = vox.nonzero()
    if coords.numel() == 0:
        ax.set_title(title + " (empty)")
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        return

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.scatter(x, y, z, s=2)
    ax.set_title(title)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    ax.set_box_aspect([1, 1, 1])


def choose_sample(dataset, category=None, seed=42):
    if category is None:
        return dataset[0]

    matching = [i for i, r in enumerate(dataset.records) if r["category"] == category]
    if not matching:
        raise ValueError(f"No test samples found for category '{category}'")

    rng = random.Random(seed)
    return dataset[rng.choice(matching)]


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VoxelDataset(
        index_path="data/processed_pointcloud/index.jsonl",
        split="train",
    )
    test_dataset = VoxelDataset(
        index_path="data/processed_pointcloud/index.jsonl",
        split="test",
        category_to_idx=train_dataset.category_to_idx,
    )

    baseline = RandomRetrievalBaseline(train_dataset, split="train", seed=args.seed)

    model = VoxelVAE3D(
        num_categories=len(train_dataset.category_to_idx),
        latent_dim=128,
        category_embed_dim=32,
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/vae_best.pt", map_location=device))
    model.eval()

    sample = choose_sample(test_dataset, category=args.category, seed=args.seed)

    x = sample["voxel"].unsqueeze(0).to(device)
    category_idx = sample["category_idx"].unsqueeze(0).to(device)

    with torch.no_grad():
        # Stable reconstruction: decode from mu, not sampled z
        mu, logvar = model.encode(x)
        recon_logits = model.decode(mu, category_idx)
        recon = (torch.sigmoid(recon_logits) >= args.threshold).float()[0, 0].cpu()

        # Random generation
        if args.sample_mode == "fixed":
            torch.manual_seed(args.seed)
        z = torch.randn(1, model.latent_dim, device=device)
        gen_logits = model.decode(z, category_idx)
        gen = (torch.sigmoid(gen_logits) >= args.threshold).float()[0, 0].cpu()

    baseline_sample = baseline.sample_by_category(sample["category"])
    baseline_vox = baseline_sample["voxel"][0].cpu()
    gt = sample["voxel"][0].cpu()

    print("Category:", sample["category"])
    print("GT occupied:", int(gt.sum().item()))
    print("Baseline occupied:", int(baseline_vox.sum().item()))
    print("Recon occupied:", int(recon.sum().item()))
    print("Gen occupied:", int(gen.sum().item()))

    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(141, projection="3d")
    ax2 = fig.add_subplot(142, projection="3d")
    ax3 = fig.add_subplot(143, projection="3d")
    ax4 = fig.add_subplot(144, projection="3d")

    plot_voxel(ax1, gt, f"Ground Truth ({sample['category']})")
    plot_voxel(ax2, baseline_vox, "Baseline Retrieval")
    plot_voxel(ax3, recon, "VAE Reconstruction")
    plot_voxel(ax4, gen, f"VAE Generation ({args.sample_mode})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()