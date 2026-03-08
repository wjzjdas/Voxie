from __future__ import annotations

import random
from collections import defaultdict

#Fix random seed for consistent baseline evaluation
class RandomRetrievalBaseline: 
    def __init__(self, dataset, split: str = "train", seed: int = 42) -> None:
        self.dataset = dataset
        self.rng = random.Random(seed)

        self.by_category = defaultdict(list)

        for i, record in enumerate(dataset.records):
            if record["split"] == split:
                self.by_category[record["category"]].append(i)

        self.categories = sorted(self.by_category.keys())

    def sample_by_category(self, category: str):
        if category not in self.by_category or len(self.by_category[category]) == 0:
            raise ValueError(f"No samples found for category '{category}'")

        idx = self.rng.choice(self.by_category[category])
        return self.dataset[idx]

    def sample_by_category_idx(self, category_idx: int):
        category = self.dataset.idx_to_category[int(category_idx)]
        return self.sample_by_category(category)