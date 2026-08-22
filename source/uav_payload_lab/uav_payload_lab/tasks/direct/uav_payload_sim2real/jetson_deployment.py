"""Pure-PyTorch modules and strict checkpoint loaders for Jetson deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


HISTORY_LEN = 50
PROPRIO_DIM = 21
Z_SLOW_DIM = 2
Z_FAST_DIM = 3
Z_DIM = Z_SLOW_DIM + Z_FAST_DIM
ACTOR_INPUT_DIM = PROPRIO_DIM + Z_DIM
ACTION_DIM = 4
NORMALIZER_EPS = 1.0e-2


class CNNContextEncoder(nn.Module):
    """History encoder with the exact architecture used by Phase-II training."""

    def __init__(self, input_dim: int, history_len: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * self.history_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            expected = (self.history_len, self.input_dim)
            if history.ndim != 3 or tuple(history.shape[1:]) != expected:
                raise ValueError(
                    "Expected history shape [B, "
                    f"{self.history_len}, {self.input_dim}], got {tuple(history.shape)}"
                )
        channels_first = history.permute(0, 2, 1)
        return self.mlp(self.cnn(channels_first))


class NormalizedActor(nn.Module):
    """Actor MLP with the training-time empirical normalization embedded."""

    def __init__(
        self,
        actor: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = NORMALIZER_EPS,
    ):
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_mean", mean.detach().clone().reshape(1, -1))
        self.register_buffer("obs_std", std.detach().clone().reshape(1, -1))
        self.eps = float(eps)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if actor_input.ndim != 2 or actor_input.shape[-1] != self.obs_mean.shape[-1]:
                raise ValueError(
                    "Expected actor input shape [B, "
                    f"{self.obs_mean.shape[-1]}], got {tuple(actor_input.shape)}"
                )
        normalized = (actor_input - self.obs_mean) / (self.obs_std + self.eps)
        return self.actor(normalized)


@dataclass(frozen=True)
class DeploymentModels:
    slow_encoder: CNNContextEncoder
    fast_encoder: CNNContextEncoder
    actor: NormalizedActor


def validate_student_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject a checkpoint that cannot reproduce the selected split-hard run."""

    expected = {
        "student_context_mode": "split",
        "teacher_context_mode": "split_hard",
        "input_dim": PROPRIO_DIM,
        "history_len": HISTORY_LEN,
        "z_dim": Z_DIM,
        "z_slow_dim": Z_SLOW_DIM,
        "z_fast_dim": Z_FAST_DIM,
    }
    missing = [key for key in (*expected, "state_dict") if key not in checkpoint]
    if missing:
        raise ValueError(f"Student checkpoint is missing required fields: {missing}")
    for key, value in expected.items():
        if checkpoint[key] != value:
            raise ValueError(
                f"Student checkpoint {key} must be {value!r}, got {checkpoint[key]!r}; "
                "this exporter only accepts the split_hard deployment lineage."
            )


def _branch_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    branch = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not branch:
        raise ValueError(f"Student state_dict has no parameters with prefix {prefix!r}")
    return branch


def _build_actor(model_state: Mapping[str, torch.Tensor]) -> NormalizedActor:
    linear_indices = sorted(
        int(key.split(".")[1])
        for key, value in model_state.items()
        if key.startswith("actor.")
        and key.endswith(".weight")
        and isinstance(value, torch.Tensor)
        and value.ndim == 2
    )
    if linear_indices != [0, 2, 4]:
        raise ValueError(
            f"Expected Actor linear module indices [0, 2, 4], got {linear_indices}"
        )

    modules: list[nn.Module] = []
    for position, module_index in enumerate(linear_indices):
        weight = model_state[f"actor.{module_index}.weight"]
        bias = model_state[f"actor.{module_index}.bias"]
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        modules.append(layer)
        if position < len(linear_indices) - 1:
            modules.append(nn.ELU())

    actor = nn.Sequential(*modules)
    if actor[0].in_features != ACTOR_INPUT_DIM or actor[-1].out_features != ACTION_DIM:
        raise ValueError(
            "Teacher Actor shape mismatch: expected "
            f"{ACTOR_INPUT_DIM}->...->{ACTION_DIM}, got "
            f"{actor[0].in_features}->...->{actor[-1].out_features}"
        )

    mean_key = "actor_obs_normalizer._mean"
    std_key = "actor_obs_normalizer._std"
    if mean_key not in model_state or std_key not in model_state:
        raise ValueError("Teacher checkpoint is missing Actor observation normalization")
    return NormalizedActor(actor, model_state[mean_key], model_state[std_key])


def load_deployment_models(
    teacher_checkpoint_path: str | Path,
    student_checkpoint_path: str | Path,
) -> tuple[DeploymentModels, dict[str, Any]]:
    """Load strict CPU/eval deployment models and their audited lineage."""

    teacher_path = Path(teacher_checkpoint_path).resolve()
    student_path = Path(student_checkpoint_path).resolve()
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    student_checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)

    validate_student_checkpoint(student_checkpoint)
    if "model_state_dict" not in teacher_checkpoint:
        raise ValueError("Teacher checkpoint is missing model_state_dict")

    slow_encoder = CNNContextEncoder(PROPRIO_DIM, HISTORY_LEN, Z_SLOW_DIM)
    fast_encoder = CNNContextEncoder(PROPRIO_DIM, HISTORY_LEN, Z_FAST_DIM)
    student_state = student_checkpoint["state_dict"]
    slow_encoder.load_state_dict(
        _branch_state_dict(student_state, "slow_encoder."), strict=True
    )
    fast_encoder.load_state_dict(
        _branch_state_dict(student_state, "fast_encoder."), strict=True
    )
    actor = _build_actor(teacher_checkpoint["model_state_dict"])

    slow_encoder.eval()
    fast_encoder.eval()
    actor.eval()
    models = DeploymentModels(slow_encoder, fast_encoder, actor)
    metadata = {
        "teacher_checkpoint": str(teacher_path),
        "student_checkpoint": str(student_path),
        "teacher_iteration": int(teacher_checkpoint.get("iter", -1)),
        "student_epoch": int(student_checkpoint.get("epoch", -1)),
        "student_best_val": float(student_checkpoint.get("best_val", float("nan"))),
        "seed": int(student_checkpoint.get("seed", -1)),
        "teacher_context_mode": student_checkpoint["teacher_context_mode"],
        "student_context_mode": student_checkpoint["student_context_mode"],
        "input_dim": int(student_checkpoint["input_dim"]),
        "history_len": int(student_checkpoint["history_len"]),
        "z_dim": int(student_checkpoint["z_dim"]),
        "z_slow_dim": int(student_checkpoint["z_slow_dim"]),
        "z_fast_dim": int(student_checkpoint["z_fast_dim"]),
        "normalizer_eps": NORMALIZER_EPS,
    }
    return models, metadata
