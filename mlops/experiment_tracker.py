"""
Experiment Tracking
MLflow and Weights & Biases integration for v1.2.
Falls back gracefully if neither is installed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracking interface.
    Supports: MLflow | Weights & Biases | local JSON fallback

    Parameters
    ----------
    backend : 'mlflow' | 'wandb' | 'local'
    experiment_name : str
    run_name : str | None
    config : dict  hyperparameters / metadata to log
    """

    def __init__(
        self,
        backend: str = "local",
        experiment_name: str = "pbt_trading",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        local_dir: str = "experiments/runs",
    ):
        self.backend = backend
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.config = config or {}
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._run = None
        self._metrics_buffer: List[Dict] = []
        self._local_path = self.local_dir / f"{self.run_name}.jsonl"

        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "mlflow":
            self._init_mlflow()
        elif self.backend == "wandb":
            self._init_wandb()
        else:
            logger.info("ExperimentTracker: local backend → %s", self._local_path)

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
            if self.config:
                mlflow.log_params(self.config)
            self._mlflow = mlflow
            logger.info("MLflow run started: %s", self._run.info.run_id)
        except ImportError:
            logger.warning("MLflow not installed; falling back to local")
            self.backend = "local"
        except Exception as exc:
            logger.error("MLflow init failed: %s; falling back to local", exc)
            self.backend = "local"

    def _init_wandb(self) -> None:
        try:
            import wandb
            self._run = wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                config=self.config,
            )
            self._wandb = wandb
            logger.info("W&B run started: %s", self._run.name)
        except ImportError:
            logger.warning("wandb not installed; falling back to local")
            self.backend = "local"
        except Exception as exc:
            logger.error("W&B init failed: %s; falling back to local", exc)
            self.backend = "local"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        record = {"step": step, "timestamp": time.time(), **metrics}
        self._metrics_buffer.append(record)

        if self.backend == "mlflow" and self._run:
            try:
                float_metrics = {k: float(v) for k, v in metrics.items()
                                 if isinstance(v, (int, float))}
                self._mlflow.log_metrics(float_metrics, step=step)
            except Exception as exc:
                logger.debug("MLflow log error: %s", exc)

        elif self.backend == "wandb" and self._run:
            try:
                self._wandb.log(metrics, step=step)
            except Exception as exc:
                logger.debug("W&B log error: %s", exc)

        # Always write locally
        with self._local_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_generation(self, generation: int, metrics: Dict[str, Any]) -> None:
        self.log({"generation": generation, **metrics}, step=generation)

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        if self.backend == "mlflow" and self._run:
            try:
                self._mlflow.log_artifact(path)
            except Exception as exc:
                logger.debug("MLflow artifact error: %s", exc)
        elif self.backend == "wandb" and self._run:
            try:
                self._wandb.save(path)
            except Exception as exc:
                logger.debug("W&B artifact error: %s", exc)

    def log_table(self, key: str, data: List[Dict]) -> None:
        if self.backend == "wandb" and self._run:
            try:
                import pandas as pd
                self._wandb.log({key: self._wandb.Table(dataframe=pd.DataFrame(data))})
            except Exception:
                pass
        # Also write to local
        table_path = self.local_dir / f"{self.run_name}_{key}.json"
        table_path.write_text(json.dumps(data, indent=2, default=str))

    # ------------------------------------------------------------------
    # Summary & finish
    # ------------------------------------------------------------------

    def summary(self, metrics: Dict[str, Any]) -> None:
        if self.backend == "mlflow" and self._run:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._mlflow.log_metric(f"summary/{k}", v)
            except Exception:
                pass
        elif self.backend == "wandb" and self._run:
            try:
                self._wandb.run.summary.update(metrics)
            except Exception:
                pass
        summary_path = self.local_dir / f"{self.run_name}_summary.json"
        summary_path.write_text(json.dumps(metrics, indent=2, default=str))

    def finish(self) -> None:
        if self.backend == "mlflow" and self._run:
            try:
                self._mlflow.end_run()
            except Exception:
                pass
        elif self.backend == "wandb" and self._run:
            try:
                self._wandb.finish()
            except Exception:
                pass
        logger.info("ExperimentTracker finished: %d records logged", len(self._metrics_buffer))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
