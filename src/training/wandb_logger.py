"""Weights & Biases logging integration for MarkGPT training.

Provides utilities for logging training metrics, model checkpoints, and
sample generations to Weights & Biases for experiment tracking and analysis.

References:
    - Smith, L. N. (2018). A disciplined approach to neural network training.
      https://arxiv.org/abs/1803.09820
    - Weights & Biases: https://docs.wandb.ai/
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class WandbLogger:
    """Weights & Biases experiment logger for training metrics and checkpoints.
    
    Attributes:
        enabled (bool): Whether wandb logging is active
        project (str): W&B project name
        entity (str): W&B entity (team/username)
    """

    def __init__(
        self,
        project: str = "markgpt",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """Initialize Weights & Biases logger.
        
        Args:
            project: Weights & Biases project name
            entity: W&B entity (team or username)
            name: Run name for this experiment
            config: Configuration dictionary to log (hyperparameters, model config)
            enabled: Whether to actually log to W&B or use no-op mode
        """
        self.enabled = enabled
        self.project = project
        self.entity = entity

        if enabled:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=config or {},
                    reinit=True,
                )
            except ImportError:
                logger.warning("wandb not installed, logging disabled")
                self.enabled = False
        else:
            self.wandb = None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        if not self.enabled:
            return

        self.wandb.log(metrics, step=step)

    def log_model_gradients(
        self, model: nn.Module, step: int, log_freq: int = 100
    ) -> None:
        """Log model gradient statistics.
        
        Args:
            model: Model to log gradients for
            step: Current training step
            log_freq: Only log every log_freq steps
        """
        if not self.enabled or step % log_freq != 0:
            return

        gradient_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_dict[f"grad/{name}"] = self.wandb.Histogram(
                    param.grad.cpu().detach()
                )
                gradient_dict[f"grad_norm/{name}"] = param.grad.norm().item()

        self.wandb.log(gradient_dict, step=step)

    def log_sample_generation(
        self, text: str, step: int, prefix: str = "sample"
    ) -> None:
        """Log generated text sample.
        
        Args:
            text: Generated text to log
            step: Current training step
            prefix: Prefix for the log entry
        """
        if not self.enabled:
            return

        self.wandb.log({f"{prefix}/generation": text}, step=step)

    def log_checkpoint(self, path: Path, step: int) -> None:
        """Log model checkpoint to W&B artifacts.
        
        Args:
            path: Path to checkpoint file
            step: Current training step
        """
        if not self.enabled:
            return

        artifact = self.wandb.Artifact(
            f"checkpoint-step-{step}", type="model"
        )
        artifact.add_file(str(path))
        self.wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled and self.wandb is not None:
            self.wandb.finish()
