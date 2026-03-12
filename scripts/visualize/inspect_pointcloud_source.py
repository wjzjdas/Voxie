'''
How to use it
Case 1: inspect only the original .npy
python -m scripts.visualize.inspect_pointcloud_source --npy path/to/model.npy
Case 2: inspect .npy + .seg
python -m scripts.visualize.inspect_pointcloud_source --npy path/to/model.npy --seg path/to/model.seg
Case 3: compare raw source against processed .pt
python -m scripts.visualize.inspect_pointcloud_source --npy path/to/model.npy --seg path/to/model.seg --pt path/to/model.pt
Case 4: save the visualization to a file
python -m scripts.visualize.inspect_pointcloud_source --npy path/to/model.npy --seg path/to/model.seg --pt
'''

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_seg_file(seg_path: str) -> np.ndarray:
    """
    Load a .seg file. Supports:
    - text labels via np.loadtxt
    - binary numpy labels via np.load (if the file is actually saved that way)
    """
    try:
        labels = np.loadtxt(seg_path).astype(int)
        return labels
    except Exception:
        try:
            labels = np.load(seg_path)
            labels = np.asarray(labels).astype(int)
            return labels
        except Exception as e:
            raise RuntimeError(f"Failed to load seg file '{seg_path}': {e}")


def load_points_file(npy_path: str) -> np.ndarray:
    """
    Load a .npy point cloud file.
    Expected shape is usually (N, 3).
    """
    try:
        points = np.load(npy_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load npy file '{npy_path}': {e}")

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected point cloud shape (N, 3), but got {points.shape} from '{npy_path}'"
        )

    return points


def summarize_array(name: str, arr: np.ndarray) -> None:
    print(f"\n{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

    if arr.size == 0:
        print("  array is empty")
        return

    if arr.ndim == 2 and arr.shape[1] == 3:
        print(f"  min xyz: {arr.min(axis=0)}")
        print(f"  max xyz: {arr.max(axis=0)}")
        print(f"  mean xyz: {arr.mean(axis=0)}")
        print(f"  std xyz:  {arr.std(axis=0)}")
        print(f"  first 5 rows:\n{arr[:5]}")
    else:
        unique_vals = np.unique(arr)
        preview = unique_vals[:20]
        print(f"  unique values (first 20): {preview}")
        if len(unique_vals) > 20:
            print(f"  total unique values: {len(unique_vals)}")
        print(f"  first 20 entries: {arr[:20]}")


def verify_point_label_match(points: np.ndarray, labels: np.ndarray) -> None:
    print("\nConsistency checks:")
    print(f"  points.shape[0] == labels.shape[0]: {points.shape[0] == labels.shape[0]}")
    print(f"  points.ndim == 2 and points.shape[1] == 3: {points.ndim == 2 and points.shape[1] == 3}")


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    """
    Make 3D plot axes have equal scale so the shape is not visually distorted.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2

    max_range = max(
        (x.max() - x.min()),
        (y.max() - y.min()),
        (z.max() - z.min()),
    ) / 2

    if max_range == 0:
        max_range = 1.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


def plot_point_cloud(ax, points: np.ndarray, title: str, labels: np.ndarray = None, point_size: int = 4) -> None:
    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=point_size)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax, points)


def tensor_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj

def try_extract_points_from_pt(pt_obj):
    """
    Try to extract plottable 3D coordinates from a loaded .pt file.

    Supported cases:
    - tensor of shape (N, 3): interpreted as point cloud
    - tensor of shape (R, R, R): interpreted as voxel grid, converted to occupied voxel coords
    - dict containing point-cloud-like keys
    - dict containing voxel-grid-like keys
    """
    point_candidate_keys = ["points", "pointcloud", "point_cloud", "xyz"]
    voxel_candidate_keys = ["voxels", "voxel", "grid", "occupancy", "occupancy_grid"]

    if isinstance(pt_obj, torch.Tensor):
        arr = tensor_to_numpy(pt_obj)
        arr = np.asarray(arr)

        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr, "tensor_pointcloud", "pointcloud"

        if arr.ndim == 3:
            coords = np.argwhere(arr > 0)
            return coords, "tensor_voxel_grid", "voxel"

        return None, "tensor_unsupported_shape", None

    if isinstance(pt_obj, dict):
        for key in point_candidate_keys:
            if key in pt_obj:
                arr = tensor_to_numpy(pt_obj[key])
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr, f"dict['{key}']", "pointcloud"

        for key in voxel_candidate_keys:
            if key in pt_obj:
                arr = tensor_to_numpy(pt_obj[key])
                arr = np.asarray(arr)
                if arr.ndim == 3:
                    coords = np.argwhere(arr > 0)
                    return coords, f"dict['{key}']", "voxel"

        return None, "dict_no_supported_key", None

    return None, f"unsupported_type_{type(pt_obj)}", None

def inspect_pt_file(pt_path: str):
    print(f"\nLoading PT file: {pt_path}")
    try:
        pt_obj = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"  Failed to load PT file: {e}")
        return None, None

    print(f"  Loaded PT object type: {type(pt_obj)}")

    if isinstance(pt_obj, dict):
        print("  Dict keys:")
        for k, v in pt_obj.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                try:
                    shape = np.asarray(v).shape
                    print(f"    {k}: type={type(v)}, shape={shape}")
                except Exception:
                    print(f"    {k}: type={type(v)}")

    elif isinstance(pt_obj, torch.Tensor):
        print(f"  Tensor shape: {tuple(pt_obj.shape)}, dtype: {pt_obj.dtype}")

    coords, source, data_kind = try_extract_points_from_pt(pt_obj)

    if coords is not None:
        summarize_array(f"processed data from {source}", coords)

        if data_kind == "voxel":
            print(f"  interpreted as voxel grid")
            print(f"  occupied voxels: {coords.shape[0]}")
        elif data_kind == "pointcloud":
            print(f"  interpreted as point cloud")
            print(f"  number of points: {coords.shape[0]}")

        return coords, data_kind

    print("  Could not confidently extract plottable 3D coordinates from this PT file.")
    return None, None

def build_figure(
    raw_points,
    seg_labels=None,
    processed_points=None,
    processed_kind=None,
    point_size=4,
    title_prefix=""
) -> plt.Figure:
    num_plots = 2 if processed_points is None else 3
    fig = plt.figure(figsize=(6 * num_plots, 6))

    ax1 = fig.add_subplot(1, num_plots, 1, projection="3d")
    plot_point_cloud(ax1, raw_points, f"{title_prefix}Raw NPY Point Cloud", point_size=point_size)

    ax2 = fig.add_subplot(1, num_plots, 2, projection="3d")
    if seg_labels is not None and len(seg_labels) == len(raw_points):
        plot_point_cloud(
            ax2,
            raw_points,
            f"{title_prefix}Raw NPY + SEG Labels",
            labels=seg_labels,
            point_size=point_size,
        )
    else:
        plot_point_cloud(
            ax2,
            raw_points,
            f"{title_prefix}Raw NPY (No Valid SEG)",
            point_size=point_size,
        )

    if processed_points is not None:
        ax3 = fig.add_subplot(1, num_plots, 3, projection="3d")

        if processed_kind == "voxel":
            plot_point_cloud(
                ax3,
                processed_points,
                f"{title_prefix}Processed PT Voxel Occupancy",
                point_size=point_size,
            )
        else:
            plot_point_cloud(
                ax3,
                processed_points,
                f"{title_prefix}Processed PT Point Cloud",
                point_size=point_size,
            )

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw .npy point clouds, .seg labels, and optionally processed .pt point clouds."
    )
    parser.add_argument(
        "--npy",
        type=str,
        required=True,
        help="Path to the source .npy file containing point cloud data.",
    )
    parser.add_argument(
        "--seg",
        type=str,
        default=None,
        help="Path to the corresponding .seg file.",
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="Optional path to the processed .pt file for comparison.",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=4,
        help="Scatter point size for visualization.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure instead of only showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the plot window.",
    )

    args = parser.parse_args()

    npy_path = args.npy
    seg_path = args.seg
    pt_path = args.pt

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"NPY file not found: {npy_path}")

    raw_points = load_points_file(npy_path)
    summarize_array("raw points", raw_points)

    seg_labels = None
    if seg_path is not None:
        if not os.path.exists(seg_path):
            print(f"\nWarning: SEG file not found: {seg_path}")
        else:
            seg_labels = load_seg_file(seg_path)
            summarize_array("seg labels", seg_labels)
            verify_point_label_match(raw_points, seg_labels)

            if len(seg_labels) != len(raw_points):
                print(
                    "\nWarning: SEG labels length does not match number of points. "
                    "SEG coloring will be skipped."
                )
                seg_labels = None
                
    processed_points = None
    processed_kind = None

    if pt_path is not None:
        if not os.path.exists(pt_path):
            print(f"\nWarning: PT file not found: {pt_path}")
        else:
            processed_points, processed_kind = inspect_pt_file(pt_path)

            if processed_points is not None:
                print("\nRaw vs Processed comparison:")
                print(f"  raw shape:       {raw_points.shape}")
                print(f"  processed shape: {processed_points.shape}")
                print(f"  raw min xyz:       {raw_points.min(axis=0)}")
                print(f"  raw max xyz:       {raw_points.max(axis=0)}")
                print(f"  processed min xyz: {processed_points.min(axis=0)}")
                print(f"  processed max xyz: {processed_points.max(axis=0)}")
                print(f"  processed kind:    {processed_kind}")

    title_prefix = ""
    sample_name = Path(npy_path).stem
    if sample_name:
        title_prefix = f"[{sample_name}] "

    fig = build_figure(
        raw_points=raw_points,
        seg_labels=seg_labels,
        processed_points=processed_points,
        processed_kind=processed_kind,
        point_size=args.point_size,
        title_prefix=title_prefix,
    )

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to: {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()