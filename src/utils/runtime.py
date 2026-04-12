import random

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    resolved_device = device or get_device()
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif (
        resolved_device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
