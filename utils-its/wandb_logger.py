from __future__ import annotations

from typing import Any, Dict, Optional


class WandbLogger:
    def __init__(self, enabled: bool = False, project: str = "its", run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = enabled
        self.run = None
        if self.enabled:
            import wandb

            self.run = wandb.init(project=project, name=run_name, config=config, reinit=True)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.log(data, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
