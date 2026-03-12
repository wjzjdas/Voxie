'''
python -m scripts.preprocess.preprocess_pointclouds --dataset_root data/Shapenetcore_benchmark --output_dir data/processed_pointcloud
'''


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ShapeNet point clouds into 32^3 voxel tensors."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root folder of ShapeNetcore_benchmark",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder for voxel .pt files and index.jsonl",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Voxel grid resolution",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Optional directory containing train_split.json / val_split.json / test_split.json. "
             "If omitted, the script first checks dataset_root directly.",
    )
    return parser.parse_args()


def load_split_file(split_path: Path) -> list:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pointcloud_to_voxel(points: np.ndarray, resolution: int = 32) -> np.ndarray:
    """
    Convert Nx3 point cloud to binary occupancy grid [R, R, R]
    using isotropic scaling to preserve aspect ratio.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {points.shape}")

    points = points.astype(np.float32)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = (mins + maxs) / 2.0
    points = points - center

    extent = maxs - mins
    max_extent = float(extent.max())
    if max_extent < 1e-8:
        max_extent = 1.0

    # Scale uniformly so aspect ratio is preserved
    points = points / max_extent

    # Move into [0, 1]
    points = points + 0.5
    points = np.clip(points, 0.0, 1.0)

    idx = np.floor(points * (resolution - 1)).astype(np.int32)
    idx = np.clip(idx, 0, resolution - 1)

    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid


def resolve_split_files(dataset_root: Path, splits_dir_arg: Optional[str]) -> dict[str, Path]:
    """
    Resolve where train/val/test split files are stored.

    Priority:
    1. --splits_dir if provided
    2. dataset_root directly
    3. dataset_root / train_test_split
    """
    filenames = {
        "train": "train_split.json",
        "val": "val_split.json",
        "test": "test_split.json",
    }

    if splits_dir_arg is not None:
        base = Path(splits_dir_arg).resolve()
        return {split: base / name for split, name in filenames.items()}

    direct = {split: dataset_root / name for split, name in filenames.items()}
    nested = {split: dataset_root / "train_test_split" / name for split, name in filenames.items()}

    direct_exists = any(path.exists() for path in direct.values())
    nested_exists = any(path.exists() for path in nested.values())

    if direct_exists:
        print(f"[INFO] Using split files directly under dataset root: {dataset_root}")
        return direct

    if nested_exists:
        print(f"[INFO] Using split files under: {dataset_root / 'train_test_split'}")
        return nested

    print("[WARN] Could not find split files in either of these locations:")
    print(f"       1) {dataset_root}")
    print(f"       2) {dataset_root / 'train_test_split'}")
    return direct


def process_entry(
    entry: list,
    dataset_root: Path,
    voxel_dir: Path,
    split_name: str,
    resolution: int,
) -> Optional[dict]:
    """
    Example entry:
    [8, "Lamp", "03636649/points/9db87bf898efd448cbde89e0c48a01bf.npy", ...]
    """
    if len(entry) < 3:
        return None

    model_idx = entry[0]
    category = entry[1]
    rel_points_path = entry[2]

    points_path = dataset_root / rel_points_path
    if not points_path.exists():
        print(f"[SKIP] Missing point cloud: {points_path}")
        return None

    try:
        points = np.load(points_path)
    except Exception as exc:
        print(f"[SKIP] Failed to load {points_path}: {exc}")
        return None

    try:
        grid = pointcloud_to_voxel(points, resolution=resolution)
    except Exception as exc:
        print(f"[SKIP] Failed to voxelize {points_path}: {exc}")
        return None

    occupied = int(grid.sum())
    if occupied == 0:
        print(f"[SKIP] Empty voxel grid: {points_path}")
        return None

    stem = points_path.stem
    model_id = f"{category}__{stem}"
    out_path = voxel_dir / f"{model_id}.pt"

    tensor = torch.from_numpy(grid)  # uint8
    torch.save(tensor, out_path)

    record = {
        "id": model_id,
        "model_idx": model_idx,
        "category": category,
        "source_points_path": str(rel_points_path).replace("\\", "/"),
        "voxel_path": str(out_path.relative_to(voxel_dir.parent)).replace("\\", "/"),
        "split": split_name,
        "resolution": resolution,
        "occupied_voxels": occupied,
    }
    return record


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    voxel_dir = output_dir / "voxels"
    index_path = output_dir / "index.jsonl"

    voxel_dir.mkdir(parents=True, exist_ok=True)

    split_files = resolve_split_files(dataset_root, args.splits_dir)

    records = []

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            print(f"[WARN] Split file not found: {split_path}")
            continue

        entries = load_split_file(split_path)
        print(f"{split_name}: found {len(entries)} entries")

        for i, entry in enumerate(entries, start=1):
            if i % 500 == 0:
                print(f"[{split_name}] processed {i}/{len(entries)}")

            rec = process_entry(
                entry=entry,
                dataset_root=dataset_root,
                voxel_dir=voxel_dir,
                split_name=split_name,
                resolution=args.resolution,
            )
            if rec is not None:
                records.append(rec)

    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone. Saved {len(records)} voxel tensors.")
    print(f"Voxel dir: {voxel_dir}")
    print(f"Index: {index_path}")
    print(
        f'python -c "print(sum(1 for _ in open(r\'{index_path}\', encoding=\'utf-8\')))"'
    )


if __name__ == "__main__":
    main()