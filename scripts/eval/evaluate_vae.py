#python -m scripts.evaluate_vae

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

from voxie.data.voxel_dataset import VoxelDataset
from voxie.models.vae3d import VoxelVAE3D

THRESHOLD = 0.60

def voxel_iou(a, b):
    a = a.bool()
    b = b.bool()
    intersection = (a & b).sum().item()
    union = (a | b).sum().item()
    return 0.0 if union == 0 else intersection / union


def main():
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = VoxelVAE3D(
        num_categories=len(train_dataset.category_to_idx),
        latent_dim=128,
        category_embed_dim=32,
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/vae_best.pt", map_location=device))
    model.eval()

    ious = []
    category_ious = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            x = batch["voxel"].to(device)
            category_idx = batch["category_idx"].to(device)
            categories = batch["category"]

            logits, _, _ = model(x, category_idx)
            probs = torch.sigmoid(logits)
            preds = (torch.sigmoid(logits) >= THRESHOLD).float()

            for pred, target, category in zip(preds, x, categories):
                iou = voxel_iou(pred.cpu(), target.cpu())
                ious.append(iou)
                category_ious[category].append(iou)

    ious = np.array(ious)

    print("\n===== VAE Test Performance =====")
    print("Samples evaluated:", len(ious))
    print("Mean IoU:", np.mean(ious))
    print("Std IoU:", np.std(ious))
    print("Min IoU:", np.min(ious))
    print("Max IoU:", np.max(ious))

    print("\n===== Per-Category IoU =====")
    for category in sorted(category_ious.keys()):
        cat_ious = np.array(category_ious[category])
        print(f"{category:12s} N={len(cat_ious):4d} Mean IoU={np.mean(cat_ious):.4f}")


if __name__ == "__main__":
    main()