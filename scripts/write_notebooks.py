"""One-time script that writes clean ITS notebooks using the current src/its.* API."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def cell(source: str, kind: str = "code") -> dict:
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def write_nb(path: Path, cells: list) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Written {path}")


# ------------------------------------------------------------------
# 01_baseline.ipynb
# ------------------------------------------------------------------
cells_01 = [
    cell("# ITS — Baseline Score-Based Sampler\n\nUses the `src/its.*` API.\n\n## Contents\n1. Load score-model checkpoint\n2. Baseline SDE sampling\n3. DDPM / DDIM sampling\n4. Visualise sample grids\n5. Quick FID/IS evaluation", "markdown"),
    cell(
        "import sys\nfrom pathlib import Path\n\nROOT = Path('.').resolve().parent\nif str(ROOT / 'src') not in sys.path:\n    sys.path.insert(0, str(ROOT / 'src'))\nprint('Project root:', ROOT)"
    ),
    cell(
        "import torch\nfrom its.models import ScoreUNetConfig, build_score_model\nfrom its.sde import ScoreSDEConfig, ScoreSDESimulator\n\nDEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\nprint('Device:', DEVICE)"
    ),
    cell("## 1. Load checkpoint", "markdown"),
    cell(
        "CKPT_DIR = ROOT / 'checkpoints' / 'score'\ncheckpoints = sorted(CKPT_DIR.glob('score_epoch_*.pt'))\nCKPT_PATH = checkpoints[-1] if checkpoints else None\nprint('Latest checkpoint:', CKPT_PATH)"
    ),
    cell(
        "if CKPT_PATH:\n    state = torch.load(CKPT_PATH, map_location='cpu')\n    if 'model_config' in state:\n        model_cfg = ScoreUNetConfig(**state['model_config'])\n        print('Config from checkpoint:', model_cfg)\n    else:\n        # Fallback for the epoch-50 checkpoint trained with base_channels=128.\n        model_cfg = ScoreUNetConfig(in_channels=3, base_channels=128, channel_mults=(1, 2, 2, 4))\n        print('WARNING: using fallback config (no model_config in checkpoint)')\n    score_model = build_score_model(model_cfg)\n    score_model.load_state_dict(state['model'])\n    score_model.to(DEVICE).eval()\n    print('Parameters:', sum(p.numel() for p in score_model.parameters()))\nelse:\n    model_cfg = ScoreUNetConfig(in_channels=3, base_channels=32)\n    score_model = build_score_model(model_cfg).to(DEVICE).eval()\n    print('No checkpoint — using random weights.')"
    ),
    cell("## 2. Baseline SDE sampling", "markdown"),
    cell(
        "sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=20.0, num_steps=200,\n                        sigma_min=0.01, sigma_max=1.0, control_weight=0.0)\nsimulator = ScoreSDESimulator(score_model, sde_cfg)\nBATCH = 16\nwith torch.no_grad():\n    samples, stats = simulator.sample(\n        shape=(BATCH, model_cfg.in_channels, 32, 32),\n        device=torch.device(DEVICE), return_stats=True,\n    )\nprint('Shape:', samples.shape, '| Stats:', stats)"
    ),
    cell(
        "import matplotlib.pyplot as plt\nfrom torchvision.utils import make_grid\n\ngrid = make_grid(samples.cpu().clamp(-1, 1) * 0.5 + 0.5, nrow=4)\nplt.figure(figsize=(8, 8))\nplt.imshow(grid.permute(1, 2, 0).numpy())\nplt.axis('off')\nplt.title('Baseline SDE samples')\nplt.show()"
    ),
    cell("## 3. DDPM / DDIM sampling\n\nThe `DDPMSampler` internally calls `score_to_eps(score, sigma)` before the update rule.", "markdown"),
    cell(
        "from its.samplers.ddpm import DDPMSampler, DDPMSamplerConfig\n\nddpm_cfg = DDPMSamplerConfig(num_steps=100, use_ddim=False)\nddpm = DDPMSampler(score_model, ddpm_cfg)\nwith torch.no_grad():\n    ddpm_samples, ddpm_stats = ddpm.sample((BATCH, model_cfg.in_channels, 32, 32), torch.device(DEVICE))\nprint('DDPM NFE:', ddpm_stats['nfe_total'])"
    ),
    cell(
        "ddim_cfg = DDPMSamplerConfig(num_steps=50, use_ddim=True, ddim_eta=0.0)\nddim = DDPMSampler(score_model, ddim_cfg)\nwith torch.no_grad():\n    ddim_samples, ddim_stats = ddim.sample((BATCH, model_cfg.in_channels, 32, 32), torch.device(DEVICE))\nprint('DDIM NFE:', ddim_stats['nfe_total'])"
    ),
    cell("## 4. Quick FID/IS evaluation (256 samples)", "markdown"),
    cell(
        "from its.eval import EvaluationConfig, evaluate_sampler\nfrom its.eval.evaluator import evaluate_ddpm_baseline\n\neval_cfg = EvaluationConfig(dataset_name='cifar10', num_samples=256, batch_size=64, device=DEVICE)\nprint('Evaluating baseline SDE ...')\nresults = evaluate_sampler(score_model, None, sde_cfg, eval_cfg)\nprint('Baseline SDE:', results)"
    ),
    cell(
        "print('Evaluating DDPM ...')\nddpm_results = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline='ddpm')\nprint('DDPM:', ddpm_results)"
    ),
]

write_nb(NB_DIR / "01_baseline.ipynb", cells_01)


# ------------------------------------------------------------------
# 02_controlled.ipynb
# ------------------------------------------------------------------
cells_02 = [
    cell("# ITS — Controlled Score-Based Sampler\n\nUses the `src/its.*` API.\n\n## Contents\n1. Load combined controlled checkpoint\n2. Controlled SDE sampling\n3. Visualise samples and compare to baseline\n4. Inspect thermodynamic diagnostics (path KL, Jarzynski, entropy)\n5. Quick benchmark: controlled vs. DDPM vs. DDIM", "markdown"),
    cell(
        "import sys\nfrom pathlib import Path\n\nROOT = Path('.').resolve().parent\nif str(ROOT / 'src') not in sys.path:\n    sys.path.insert(0, str(ROOT / 'src'))\nprint('Project root:', ROOT)"
    ),
    cell(
        "import torch\nfrom its.models import ScoreUNetConfig, build_score_model\nfrom its.controllers import ControlConfig, build_control_policy, ConvControlConfig, build_conv_control_policy\nfrom its.sde import ScoreSDEConfig, ScoreSDESimulator\n\nDEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\nprint('Device:', DEVICE)"
    ),
    cell("## 1. Load controlled checkpoint", "markdown"),
    cell(
        "CKPT_DIR = ROOT / 'checkpoints' / 'controlled'\nckpts = sorted(CKPT_DIR.glob('controlled_epoch_*.pt')) if CKPT_DIR.exists() else []\nCKPT_PATH = ckpts[-1] if ckpts else None\nif not CKPT_PATH:\n    last = CKPT_DIR / 'controlled_last.pt' if CKPT_DIR.exists() else None\n    CKPT_PATH = last if last and last.is_file() else None\nprint('Checkpoint:', CKPT_PATH)"
    ),
    cell(
        "if CKPT_PATH:\n    state = torch.load(CKPT_PATH, map_location='cpu')\n    print('Checkpoint keys:', list(state.keys()))\n    # Rebuild score model from saved config.\n    model_cfg = ScoreUNetConfig(**state['score_config']) if 'score_config' in state else ScoreUNetConfig(in_channels=1, base_channels=32, channel_mults=(1,2,2))\n    score_model = build_score_model(model_cfg)\n    score_model.load_state_dict(state.get('score_state_dict', state.get('model')))\n    score_model.to(DEVICE).eval()\n    print('Score model loaded.')\n    # Rebuild control policy from saved config.\n    ctrl_cfg = ControlConfig(**state['control_config']) if 'control_config' in state else ControlConfig(state_dim=784, hidden_dim=128)\n    control_policy = build_control_policy(ctrl_cfg, device=torch.device(DEVICE))\n    control_policy.load_state_dict(state.get('control_state_dict', state.get('control')))\n    control_policy.eval()\n    print('Control policy loaded.')\nelse:\n    print('No controlled checkpoint found. Run scripts/train_controlled_score.py first.')\n    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=32, channel_mults=(1,2,2))\n    score_model = build_score_model(model_cfg).to(DEVICE).eval()\n    ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=64)\n    control_policy = build_control_policy(ctrl_cfg, device=torch.device(DEVICE))\n    print('Using uninitialised models — metrics will be meaningless.')"
    ),
    cell("## 2. Controlled SDE sampling", "markdown"),
    cell(
        "sde_cfg = ScoreSDEConfig(\n    beta_min=0.1, beta_max=5.0, num_steps=50,\n    sigma_min=0.01, sigma_max=1.0, control_weight=1.0,\n    corrector_steps=1, corrector_step_size=0.01,\n)\nsimulator = ScoreSDESimulator(score_model, sde_cfg)\nBATCH = 16\nwith torch.no_grad():\n    ctrl_samples, ctrl_stats = simulator.sample(\n        shape=(BATCH, model_cfg.in_channels, 28, 28),\n        device=torch.device(DEVICE),\n        control=control_policy,\n        return_stats=True,\n    )\nprint('Shape:', ctrl_samples.shape)\nprint('Stats:', ctrl_stats)"
    ),
    cell(
        "import matplotlib.pyplot as plt\nfrom torchvision.utils import make_grid\n\ngrid = make_grid(ctrl_samples.cpu().clamp(-1, 1) * 0.5 + 0.5, nrow=4)\nplt.figure(figsize=(8, 8))\nplt.imshow(grid.permute(1, 2, 0).squeeze().numpy(), cmap='gray' if model_cfg.in_channels == 1 else None)\nplt.axis('off')\nplt.title('Controlled SDE samples')\nplt.show()"
    ),
    cell("## 3. Thermodynamic diagnostics", "markdown"),
    cell(
        "from its.physics.entropy import entropy_production_estimate\nfrom its.physics.fluctuation import jarzynski_work_estimate\nimport numpy as np\n\nif 'control_energy' in ctrl_stats:\n    print(f\"Control energy:  {ctrl_stats['control_energy']:.4f}\")\nif 'path_kl' in ctrl_stats:\n    print(f\"Path KL:         {ctrl_stats['path_kl']:.4f}\")\nif 'jarzynski' in ctrl_stats:\n    print(f\"Jarzynski dF:    {ctrl_stats['jarzynski']:.4f}\")"
    ),
    cell("## 4. Quick benchmark (256 samples)", "markdown"),
    cell(
        "from its.eval import EvaluationConfig, evaluate_sampler\nfrom its.eval.evaluator import evaluate_ddpm_baseline\n\neval_cfg = EvaluationConfig(dataset_name='fashionmnist', num_samples=256, batch_size=64, device=DEVICE)\n\nprint('Evaluating controlled SDE ...')\nctrl_results = evaluate_sampler(score_model, control_policy, sde_cfg, eval_cfg)\nprint('Controlled SDE:', ctrl_results)\n\nprint('Evaluating DDPM ...')\nddpm_results = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline='ddpm')\nprint('DDPM:', ddpm_results)"
    ),
    cell(
        "# Side-by-side summary table\nimport pandas as pd\n\nrows = [\n    {'sampler': 'controlled_sde', **ctrl_results},\n    {'sampler': 'ddpm', **ddpm_results},\n]\ndf = pd.DataFrame(rows).set_index('sampler')\nprint(df.to_string())"
    ),
]

write_nb(NB_DIR / "02_controlled.ipynb", cells_02)

print("Done.")
