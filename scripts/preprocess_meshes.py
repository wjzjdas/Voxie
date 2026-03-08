#!/usr/bin/env python3
"""
Preprocess 3D meshes into fixed 32x32x32 voxel grids.

Outputs:
  - one .pt file per mesh
  - one index.jsonl metadata file

Example:
  python scripts/preprocess_meshes.py \
      --input_dir data/raw \
      --output_dir data/processed \
      --resolution 32
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import trimesh


SUPPORTED_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess meshes into voxel tensors.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root folder containing raw meshes.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save processed tensors + index.")
    parser.add_argument("--resolution", type=int, default=32, help="Voxel grid resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--category_mode",
        type=str,
        choices=["parent", "topdir"],
        default="parent",
        help=(
            "How to infer category from file path. "
            "'parent' = immediate parent folder, "
            "'topdir' = first directory under input_dir."
        ),
    )
    parser.add_argument(
        "--rotate_y_deg",
        type=float,
        default=0.0,
        help="Optional fixed random rotation range around Y axis, e.g. 15 => sample in [-15, 15]. Usually keep 0 for base preprocessing.",
    )
    parser.add_argument(
        "--min_faces",
        type=int,
        default=4,
        help="Skip meshes with fewer than this many faces after loading.",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=["uint8", "bool", "float32"],
        default="uint8",
        help="Tensor dtype for saved voxel grids.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_mesh_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def infer_category(mesh_path: Path, input_dir: Path, mode: str) -> str:
    rel = mesh_path.relative_to(input_dir)

    if mode == "parent":
        return mesh_path.parent.name

    # mode == "topdir"
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return mesh_path.parent.name


def load_as_trimesh(mesh_path: Path) -> Optional[trimesh.Trimesh]:
    """
    Load file as a single trimesh.Trimesh.
    If the file is a Scene, concatenate all mesh geometries.
    """
    try:
        loaded = trimesh.load(mesh_path, force="scene")
    except Exception as exc:
        print(f"[WARN] Failed to load {mesh_path}: {exc}")
        return None

    try:
        if isinstance(loaded, trimesh.Scene):
            geometries = []
            for geom in loaded.geometry.values():
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                    geometries.append(geom)
            if not geometries:
                return None
            mesh = trimesh.util.concatenate(geometries)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            return None
    except Exception as exc:
        print(f"[WARN] Failed to combine scene {mesh_path}: {exc}")
        return None

    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None

    # Basic cleanup
    try:
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.process(validate=True)
    except Exception:
        pass

    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None

    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, rotate_y_deg: float = 0.0) -> Optional[trimesh.Trimesh]:
    """
    Normalize mesh to fit inside [-0.5, 0.5]^3, centered at origin.
    Optionally apply small random Y rotation before scaling.
    """
    mesh = mesh.copy()

    if rotate_y_deg > 0:
        angle_deg = np.random.uniform(-rotate_y_deg, rotate_y_deg)
        angle_rad = np.deg2rad(angle_deg)
        rot = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
        mesh.apply_transform(rot)

    bounds = mesh.bounds
    if bounds is None or np.any(~np.isfinite(bounds)):
        return None

    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    longest = float(np.max(extents))

    if longest <= 0 or not np.isfinite(longest):
        return None

    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / longest)

    # After scaling, longest dimension is 1.0 and centered near origin
    return mesh


def voxelize_mesh(mesh: trimesh.Trimesh, resolution: int) -> Optional[np.ndarray]:
    """
    Convert normalized mesh in [-0.5, 0.5]^3 to binary occupancy grid [R, R, R].

    Uses trimesh voxelization, then maps occupied voxel centers into a fixed dense grid.
    """
    pitch = 1.0 / resolution

    try:
        vox = mesh.voxelized(pitch=pitch)
        # Fill interior where possible
        try:
            vox = vox.fill()
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARN] Voxelization failed: {exc}")
        return None

    try:
        points = np.asarray(vox.points, dtype=np.float32)
    except Exception:
        return None

    if points.size == 0:
        return None

    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    # Map world-space occupied voxel centers into [0, resolution-1]
    idx = np.floor((points + 0.5) / pitch).astype(np.int32)
    idx = np.clip(idx, 0, resolution - 1)

    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid


def convert_dtype(arr: np.ndarray, save_dtype: str) -> torch.Tensor:
    if save_dtype == "bool":
        return torch.from_numpy(arr.astype(bool))
    if save_dtype == "float32":
        return torch.from_numpy(arr.astype(np.float32))
    return torch.from_numpy(arr.astype(np.uint8))


def assign_splits(records: list[dict], seed: int, val_ratio: float, test_ratio: float) -> list[dict]:
    """
    Deterministic per-category split.
    """
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = defaultdict(list)

    for rec in records:
        by_category[rec["category"]].append(rec)

    out = []
    for category, items in by_category.items():
        items = items[:]
        rng.shuffle(items)

        n = len(items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        # Avoid consuming all data in tiny categories
        if n >= 3:
            if n_test == 0 and test_ratio > 0:
                n_test = 1
            if n_val == 0 and val_ratio > 0 and n - n_test >= 2:
                n_val = 1
            if n_test + n_val >= n:
                n_val = max(0, n_val - 1)

        test_cut = n_test
        val_cut = n_test + n_val

        for i, item in enumerate(items):
            item = dict(item)
            if i < test_cut:
                item["split"] = "test"
            elif i < val_cut:
                item["split"] = "val"
            else:
                item["split"] = "train"
            out.append(item)

    return sorted(out, key=lambda x: x["id"])


def process_one_mesh(
    mesh_path: Path,
    input_dir: Path,
    voxel_dir: Path,
    resolution: int,
    category_mode: str,
    rotate_y_deg: float,
    min_faces: int,
    save_dtype: str,
) -> Optional[dict]:
    category = infer_category(mesh_path, input_dir, category_mode)
    mesh_id = mesh_path.relative_to(input_dir).with_suffix("")
    mesh_id_str = "__".join(mesh_id.parts)

    mesh = load_as_trimesh(mesh_path)
    if mesh is None:
        print(f"[SKIP] Could not load mesh: {mesh_path}")
        return None

    if len(mesh.faces) < min_faces:
        print(f"[SKIP] Too few faces: {mesh_path}")
        return None

    mesh = normalize_mesh(mesh, rotate_y_deg=rotate_y_deg)
    if mesh is None:
        print(f"[SKIP] Failed to normalize: {mesh_path}")
        return None

    grid = voxelize_mesh(mesh, resolution=resolution)
    if grid is None or grid.sum() == 0:
        print(f"[SKIP] Empty voxel grid: {mesh_path}")
        return None

    tensor = convert_dtype(grid, save_dtype=save_dtype)

    out_path = voxel_dir / f"{mesh_id_str}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_path)

    record = {
        "id": mesh_id_str,
        "source_path": str(mesh_path.relative_to(input_dir)).replace("\\", "/"),
        "voxel_path": str(out_path.relative_to(voxel_dir.parent)).replace("\\", "/"),
        "category": category,
        "resolution": resolution,
        "occupied_voxels": int(grid.sum()),
    }
    return record


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    voxel_dir = output_dir / "voxels"
    index_path = output_dir / "index.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = find_mesh_files(input_dir)
    print(f"Found {len(mesh_files)} mesh files.")

    records = []
    for i, mesh_path in enumerate(mesh_files, start=1):
        print(f"[{i}/{len(mesh_files)}] Processing {mesh_path}")
        rec = process_one_mesh(
            mesh_path=mesh_path,
            input_dir=input_dir,
            voxel_dir=voxel_dir,
            resolution=args.resolution,
            category_mode=args.category_mode,
            rotate_y_deg=args.rotate_y_deg,
            min_faces=args.min_faces,
            save_dtype=args.save_dtype,
        )
        if rec is not None:
            records.append(rec)

    records = assign_splits(
        records=records,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("\nDone.")
    print(f"Saved {len(records)} processed voxel tensors to: {voxel_dir}")
    print(f"Index written to: {index_path}")

    # concise verification snippet, per your preference
    print("\nVerification:")
    print(f'python -c "import json; print(sum(1 for _ in open(r\'{index_path}\', encoding=\'utf-8\')))"')


if __name__ == "__main__":
    main()