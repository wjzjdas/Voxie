import torch
import numpy as np
from collections import defaultdict

from voxie.data.voxel_dataset import VoxelDataset
from voxie.baselines.random_retrieval import RandomRetrievalBaseline


def voxel_iou(a, b):
    """
    Compute IoU between two voxel grids.
    """
    a = a.bool()
    b = b.bool()

    intersection = (a & b).sum().item()
    union = (a | b).sum().item()

    if union == 0:
        return 0.0

    return intersection / union


def main():

    dataset = VoxelDataset(
        index_path="data/processed_pointcloud/index.jsonl"
    )

    baseline = RandomRetrievalBaseline(dataset, split="train", seed=42)

    # collect test indices
    test_indices = [
        i for i, r in enumerate(dataset.records)
        if r["split"] == "test"
    ]

    print("Test samples:", len(test_indices))

    ious = []
    category_ious = defaultdict(list)

    for idx in test_indices:

        gt = dataset[idx]

        pred = baseline.sample_by_category(gt["category"])

        iou = voxel_iou(pred["voxel"], gt["voxel"])

        ious.append(iou)
        category_ious[gt["category"]].append(iou)

    ious = np.array(ious)

    print("\n===== Overall Baseline Performance =====")
    print("Samples evaluated:", len(ious))
    print("Mean IoU:", np.mean(ious))
    print("Std IoU:", np.std(ious))
    print("Min IoU:", np.min(ious))
    print("Max IoU:", np.max(ious))

    print("\n===== Per-Category IoU =====")

    for category in sorted(category_ious.keys()):

        cat_ious = np.array(category_ious[category])

        print(
            f"{category:12s} "
            f"N={len(cat_ious):4d} "
            f"Mean IoU={np.mean(cat_ious):.4f}"
        )


if __name__ == "__main__":
    main()