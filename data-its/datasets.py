from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DatasetName = Literal["cifar10", "fashionmnist", "mnist"]
_MEAN_STD: dict[DatasetName, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "fashionmnist": ((0.5,), (0.5,)),
    "mnist": ((0.1307,), (0.3081,)),
}


@dataclass
class DatasetConfig:
    name: DatasetName = "cifar10"
    root: str = "./data/tensors"
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.05
    download: bool = True
    augment: bool = True
    normalize: bool = True
    # If set, randomly sample this many examples from the training set.
    # Useful for fast smoke-test runs and hyperparameter sweep pilots.
    subset_size: Optional[int] = None


def _build_transforms(config: DatasetConfig) -> transforms.Compose:
    norm_mean, norm_std = _MEAN_STD[config.name]

    train_tfms: list[transforms.Transform] = []
    if config.augment:
        if config.name == "cifar10":
            train_tfms.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
    train_tfms.append(transforms.ToTensor())
    if config.normalize:
        train_tfms.append(transforms.Normalize(norm_mean, norm_std))
    test_tfms = [transforms.ToTensor()]
    if config.normalize:
        test_tfms.append(transforms.Normalize(norm_mean, norm_std))
    return transforms.Compose(train_tfms), transforms.Compose(test_tfms)


def _load_dataset(config: DatasetConfig, train: bool, tfm: transforms.Compose):
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    if config.name == "cifar10":
        return datasets.CIFAR10(root=str(root), train=train, download=config.download, transform=tfm)
    if config.name == "fashionmnist":
        return datasets.FashionMNIST(root=str(root), train=train, download=config.download, transform=tfm)
    if config.name == "mnist":
        return datasets.MNIST(root=str(root), train=train, download=config.download, transform=tfm)
    raise ValueError(f"Unsupported dataset: {config.name}")


def build_dataset(config: Optional[DatasetConfig] = None) -> dict[str, DataLoader]:
    cfg = config or DatasetConfig()
    train_tfm, test_tfm = _build_transforms(cfg)
    full_train = _load_dataset(cfg, train=True, tfm=train_tfm)
    test = _load_dataset(cfg, train=False, tfm=test_tfm)

    if cfg.subset_size is not None:
        subset_n = min(cfg.subset_size, len(full_train))
        full_train, _ = random_split(full_train, [subset_n, len(full_train) - subset_n])

    if cfg.val_split > 0:
        val_size = max(1, int(len(full_train) * cfg.val_split))
        train_size = len(full_train) - val_size
        train_set, val_set = random_split(full_train, [train_size, val_size])
    else:
        train_set, val_set = full_train, None

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True),
        "test": DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True),
    }
    if val_set is not None:
        loaders["val"] = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    return loaders


def get_dataset_stats(name: DatasetName) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return _MEAN_STD[name]
