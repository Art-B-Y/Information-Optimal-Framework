from __future__ import annotations

"""
Log parsing and plotting helpers to reconstruct metrics and figures from local runs.

The parsers rely only on text logs produced via `configure_logging` and printed dicts
from evaluation scripts. They do not require wandb.
"""

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt



# --------- Data containers --------- #


@dataclass
class ParsedRun:
    run_id: str
    path: Path
    records: List[dict]


# --------- Log discovery --------- #


def discover_logs(
    roots: Sequence[str | Path],
    patterns: Sequence[str] = ("*.log", "*.out", "*.txt"),
) -> List[Path]:
    """Recursively discover candidate log files under the given roots."""
    logs: List[Path] = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            logs.append(root_path)
            continue
        if not root_path.exists():
            continue
        for pat in patterns:
            logs.extend(root_path.rglob(pat))
    # Remove duplicates while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in logs:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


# --------- Regex helpers --------- #


_FLOAT = r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?"

# Controlled image training
_RE_CONTROLLED = re.compile(
    rf"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+loss=(?P<loss>{_FLOAT})\s+"
    rf"dsm=(?P<dsm>{_FLOAT})\s+control=(?P<control>{_FLOAT})\s+"
    rf"path_kl=(?P<path_kl>{_FLOAT})\s+jar=(?P<jar>{_FLOAT})"
)

# Score training (baseline)
_RE_SCORE_STEP = re.compile(
    rf"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+loss=(?P<loss>{_FLOAT})"
)
_RE_SCORE_EPOCH = re.compile(
    rf"epoch=(?P<epoch>\d+)\s+avg_loss=(?P<avg_loss>{_FLOAT})"
)

# Toy training (run_training)
_RE_TOY_STEP = re.compile(
    rf"step=(?P<step>\d+)\s+loss=(?P<loss>{_FLOAT})\s+quality=(?P<quality>{_FLOAT})\s+"
    rf"control=(?P<control>{_FLOAT})\s+entropy=(?P<entropy>{_FLOAT})\s+"
    rf"left=(?P<left>{_FLOAT})\s+right=(?P<right>{_FLOAT})\s+spread=(?P<spread>{_FLOAT})\s+bal=(?P<bal>{_FLOAT})"
)
_RE_TOY_DIAG = re.compile(
    rf"diagnostics\s+step=(?P<step>\d+)\s+balance=(?P<bal>{_FLOAT})\s+pen=(?P<pen>{_FLOAT})\s+final_abs=(?P<final_abs>{_FLOAT})"
)

# Eval line
_RE_EVAL = re.compile(r"Evaluation results:\s*(\{.*\})")


# --------- Parsers --------- #


def _parse_float_dict(match_dict: dict) -> dict:
    out = {}
    for k, v in match_dict.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def parse_controlled_training_log(path: Path) -> ParsedRun:
    records: List[dict] = []
    for line in path.read_text().splitlines():
        m = _RE_CONTROLLED.search(line)
        if m:
            rec = _parse_float_dict(m.groupdict())
            records.append(rec)
    run_id = path.parent.name or path.stem
    return ParsedRun(run_id=run_id, path=path, records=records)


def parse_score_training_log(path: Path) -> ParsedRun:
    records: List[dict] = []
    for line in path.read_text().splitlines():
        m_step = _RE_SCORE_STEP.search(line)
        if m_step:
            rec = _parse_float_dict(m_step.groupdict())
            rec["kind"] = "step"
            records.append(rec)
        m_epoch = _RE_SCORE_EPOCH.search(line)
        if m_epoch:
            rec = _parse_float_dict(m_epoch.groupdict())
            rec["kind"] = "epoch"
            records.append(rec)
    run_id = path.parent.name or path.stem
    return ParsedRun(run_id=run_id, path=path, records=records)


def parse_toy_training_log(path: Path) -> ParsedRun:
    records: List[dict] = []
    for line in path.read_text().splitlines():
        m_step = _RE_TOY_STEP.search(line)
        if m_step:
            rec = _parse_float_dict(m_step.groupdict())
            rec["kind"] = "step"
            records.append(rec)
        m_diag = _RE_TOY_DIAG.search(line)
        if m_diag:
            rec = _parse_float_dict(m_diag.groupdict())
            rec["kind"] = "diagnostic"
            records.append(rec)
    run_id = path.parent.name or path.stem
    return ParsedRun(run_id=run_id, path=path, records=records)


def parse_eval_log(path: Path) -> ParsedRun:
    records: List[dict] = []
    for line in path.read_text().splitlines():
        m = _RE_EVAL.search(line)
        if m:
            try:
                rec = ast.literal_eval(m.group(1))
                if isinstance(rec, dict):
                    records.append(rec)
            except Exception:
                try:
                    rec = json.loads(m.group(1))
                    if isinstance(rec, dict):
                        records.append(rec)
                except Exception:
                    continue
    run_id = path.parent.name or path.stem
    return ParsedRun(run_id=run_id, path=path, records=records)


# --------- CSV / JSON helpers --------- #


def save_records_csv(records: Iterable[dict], path: Path) -> None:
    import csv

    records = list(records)
    if not records:
        return
    keys = sorted({k for r in records for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def save_records_json(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(list(records), f, indent=2)


# --------- Plotting utilities --------- #


def _maybe_use_seaborn():
    try:
        import seaborn as sns  # noqa: F401
    except ModuleNotFoundError:
        return


def plot_time_series(records: List[dict], x_key: str, y_keys: Sequence[str], title: str, out_path: Path) -> None:
    _maybe_use_seaborn()
    plt.figure(figsize=(7, 4))
    for y in y_keys:
        xs = [r.get(x_key) for r in records if x_key in r and y in r]
        ys = [r.get(y) for r in records if x_key in r and y in r]
        if xs and ys:
            plt.plot(xs, ys, label=y)
    plt.xlabel(x_key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(records: List[dict], x_key: str, y_key: str, title: str, out_path: Path) -> None:
    _maybe_use_seaborn()
    xs = [r.get(x_key) for r in records if x_key in r and y_key in r]
    ys = [r.get(y_key) for r in records if x_key in r and y_key in r]
    plt.figure(figsize=(5, 4))
    plt.scatter(xs, ys, alpha=0.7)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histogram(values: Sequence[float], title: str, out_path: Path, bins: int = 50) -> None:
    _maybe_use_seaborn()
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------- Sample grid reconstruction --------- #


def save_sample_grid(
    tensor_path: Path,
    output_path: Path,
    dataset_name: str,
    nrow: Optional[int] = None,
) -> None:
    """
    Load a saved tensor of samples (e.g., final_samples.pt) and write a PNG grid.
    """
    import torch
    from torchvision.utils import make_grid, save_image
    from its.data.datasets import get_dataset_stats

    tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict) and "samples" in tensor:
        tensor = tensor["samples"]
    if not torch.is_tensor(tensor):
        raise ValueError(f"Unsupported tensor payload in {tensor_path}")
    if nrow is None:
        nrow = int(len(tensor) ** 0.5) or 1

    mean, std = get_dataset_stats(dataset_name)
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    denorm = tensor * std_t + mean_t
    grid = make_grid(denorm, nrow=nrow)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)
