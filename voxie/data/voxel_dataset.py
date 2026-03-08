from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class VoxelDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        split: Optional[str] = None,
        category_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.root_dir = self.index_path.parent

        with open(self.index_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        if split is not None:
            records = [r for r in records if r["split"] == split]

        self.records = records

        categories = sorted({r["category"] for r in self.records})
        if category_to_idx is None:
            self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        else:
            self.category_to_idx = category_to_idx

        self.idx_to_category = {i: c for c, i in self.category_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        voxel_path = self.root_dir / record["voxel_path"]
        voxel = torch.load(voxel_path)

        # convert to float for neural net training
        voxel = voxel.float()

        # make sure shape is [1, 32, 32, 32]
        if voxel.ndim == 3:
            voxel = voxel.unsqueeze(0)

        category = record["category"]
        category_idx = self.category_to_idx[category]

        return {
            "voxel": voxel,
            "category": category,
            "category_idx": torch.tensor(category_idx, dtype=torch.long),
            "id": record["id"],
        }