import numpy as np
import torch
from torch.utils.data import DataLoader

from voxie.data.voxel_dataset import VoxelDataset
from voxie.models.vae3d import VoxelVAE3D


def voxel_iou(a, b):
    a = a.bool()
    b = b.bool()
    intersection = (a & b).sum().item()
    union = (a | b).sum().item()
    return 0.0 if union == 0 else intersection / union


def evaluate_threshold(model, loader, device, threshold):
    ious = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["voxel"].to(device)
            category_idx = batch["category_idx"].to(device)

            logits, _, _ = model(x, category_idx)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for pred, target in zip(preds, x):
                ious.append(voxel_iou(pred.cpu(), target.cpu()))

    return float(np.mean(ious))


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

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    best_threshold = None
    best_iou = -1.0

    for th in thresholds:
        mean_iou = evaluate_threshold(model, test_loader, device, th)
        print(f"threshold={th:.2f} mean_iou={mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_threshold = th

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best mean IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()