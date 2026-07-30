from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def ensure_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def array_to_torch(x, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(ensure_numpy(x), dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
