from __future__ import annotations

import random
from dataclasses import dataclass

import jax
import numpy as np
import torch


@dataclass
class Seed:
    seed: int

    def split(self) -> "Seed":
        return Seed(self.seed + 1)

    def to_key(self) -> jax.random.PRNGKey:
        return jax.random.PRNGKey(self.seed)


def seed_everything(seed: int) -> Seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Seed(seed)
