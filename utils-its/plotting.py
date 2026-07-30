from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(steps: Sequence[float], losses: Sequence[float], title: str = "Loss", save_path: str | None = None) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, label="loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_tradeoff(x: Sequence[float], y: Sequence[float], xlabel: str, ylabel: str, title: str = "Trade-off", save_path: str | None = None) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()
