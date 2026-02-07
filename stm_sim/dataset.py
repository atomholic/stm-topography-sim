import json
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .simulator import generate_sample
from .utils import ensure_rng


class STMDataset:
    def __init__(self, n_samples: int, cfg: SimulationConfig, seed: int | None = None):
        self.n_samples = n_samples
        self.cfg = cfg
        self.seed = seed if seed is not None else cfg.seed
        if self.seed is None:
            self.seed = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        seed = int(self.seed) + int(idx)
        return generate_sample(self.cfg, seed=seed)


def generate_dataset(output_dir: str | Path, n_samples: int, cfg: SimulationConfig, seed: int | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = ensure_rng(seed if seed is not None else cfg.seed)

    for i in range(n_samples):
        sample_seed = int(rng.integers(0, 2**32 - 1))
        image, mask, metadata = generate_sample(cfg, seed=sample_seed)

        np.save(output_dir / f"image_{i:04d}.npy", image)
        np.save(output_dir / f"mask_{i:04d}.npy", mask)
        with open(output_dir / f"meta_{i:04d}.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
