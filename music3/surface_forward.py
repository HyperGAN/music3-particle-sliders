"""Call the loaded Music 3 module forward.

``MiniMaxMusic3Transformer1DModel.forward(hidden_states, timestep,
encoder_hidden_states)`` and ``MiniMaxMusic3ConditionEncoder.forward`` live on
the diffusers classes this product loads. ``conceptmod/textsliders`` calls
those methods and then applies an nmse or mse slider loss. This module does
not copy that loss, ``LoRANetwork``, or the autoregressive cache that builds
the forward's inputs.

``--dummy`` does not import this module. A live run calls it only when the
checkout already holds the tensors the forward requires. Missing tensors fail
closed. This module does not invent latents and does not download weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from music3.defaults import BASE_MODEL_ID


@dataclass(frozen=True)
class TransformerSurface:
    latents: torch.Tensor
    timestep: torch.Tensor
    neutral: torch.Tensor
    positive: torch.Tensor


@dataclass(frozen=True)
class EncoderSurface:
    neutral: torch.Tensor
    positive: torch.Tensor


def _surface_root(model_dir: Path, kind: str) -> Path:
    if kind == "transformer":
        return model_dir / "surface" / "transformer"
    if kind == "condition_encoder":
        return model_dir / "surface" / "encoder"
    raise RuntimeError(f"no model-surface forward for host kind {kind!r}")


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"{path} is not a local tensor this forward can read "
            f"({type(exc).__name__}: {exc}). "
            f"This command does not download {BASE_MODEL_ID}."
        ) from exc
    if not isinstance(payload, torch.Tensor):
        raise RuntimeError(
            f"{path} must be a tensor for the Music 3 module forward. "
            f"Got {type(payload).__name__}. "
            "conceptmod cache dicts are not read here."
        )
    return payload


def _rows(tensor: torch.Tensor, n_rows: int, *, name: str, row_rank: int) -> torch.Tensor:
    if tensor.ndim == row_rank:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != row_rank + 1:
        raise RuntimeError(
            f"{name} must be rank {row_rank} or {row_rank + 1}, got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[0] == 1 and n_rows != 1:
        tensor = tensor.expand(n_rows, *tensor.shape[1:]).contiguous()
    if tensor.shape[0] != n_rows:
        raise RuntimeError(
            f"{name} has {int(tensor.shape[0])} rows and the prompt card has {n_rows}."
        )
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} has non-finite values. Refusing to call the module forward.")
    return tensor.float()


def _timestep(raw: torch.Tensor | None, n_rows: int) -> torch.Tensor:
    if raw is None:
        return torch.full((n_rows,), 0.5)
    step = raw.detach().float().reshape(-1)
    if step.numel() == 1:
        step = step.expand(n_rows).contiguous()
    if tuple(step.shape) != (n_rows,):
        raise RuntimeError(
            f"timestep must be a scalar or [{n_rows}], got shape {tuple(raw.shape)}."
        )
    if not torch.isfinite(step).all() or bool((step < 0).any()) or bool((step > 1).any()):
        raise RuntimeError("timestep must be finite and inside [0, 1] for the flow transformer.")
    return step


def _missing(kind: str, model_dir: Path, missing: list[Path], context: str) -> RuntimeError:
    lines = "\n".join(f"  missing {path}" for path in missing)
    if kind == "transformer":
        detail = (
            "MiniMaxMusic3Transformer1DModel.forward(hidden_states, timestep, "
            "encoder_hidden_states) is the flow-transformer surface on the loaded "
            "diffusers module. This product calls that forward, then FormulationGame. "
            f"{model_dir} does not contain the local Flow-VAE latents and "
            "frame-aligned encoder_hidden_states that forward requires:\n"
            f"{lines}\n"
            "The script that builds those tensors from a Music 3 checkout is "
            "conceptmod/textsliders/train_lora_music3.py "
            "(_load_ar_pipeline, _encode_condition, _cache_or_encode). "
            "It also runs the nmse slider loss and LoRANetwork, so this repo does not copy it. "
            "This command does not invent latents and does not download "
            f"{BASE_MODEL_ID}."
        )
    else:
        detail = (
            "MiniMaxMusic3ConditionEncoder.forward(hidden_states) is the condition-encoder "
            "surface on the loaded diffusers module. This product calls that forward, "
            "then FormulationGame. "
            f"{model_dir} does not contain the local AR frame hiddens that forward requires:\n"
            f"{lines}\n"
            "The script that builds those tensors is "
            "conceptmod/textsliders/train_encoder_music3.py (cache_real_hiddens). "
            "It also runs the mse slider loop and LoRANetwork, including Conv1d targeting, "
            "so this repo does not copy it. "
            "This command does not invent frame hiddens and does not download "
            f"{BASE_MODEL_ID}."
        )
    return RuntimeError(f"{context} {detail}")


def load_transformer_surface(model_dir: Path, n_rows: int, *, context: str) -> TransformerSurface:
    """Read latents and paired conditions the transformer forward consumes."""
    if int(n_rows) < 1:
        raise ValueError("transformer surface needs at least one prompt row")
    root = _surface_root(model_dir, "transformer")
    required = [root / name for name in ("latents.pt", "neutral.pt", "positive.pt")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise _missing("transformer", model_dir, missing, context)
    latents = _rows(_load_tensor(required[0]), n_rows, name=str(required[0]), row_rank=2)
    neutral = _rows(_load_tensor(required[1]), n_rows, name=str(required[1]), row_rank=2)
    positive = _rows(_load_tensor(required[2]), n_rows, name=str(required[2]), row_rank=2)
    if neutral.shape != positive.shape:
        raise RuntimeError(
            "neutral and positive encoder_hidden_states differ in shape "
            f"({tuple(neutral.shape)} vs {tuple(positive.shape)})."
        )
    if int(latents.shape[-1]) != int(neutral.shape[1]):
        raise RuntimeError(
            "Flow-VAE latent length and encoder_hidden_states length differ "
            f"({int(latents.shape[-1])} vs {int(neutral.shape[1])}). "
            "MiniMaxMusic3Transformer1DModel.forward needs them aligned."
        )
    timestep_path = root / "timestep.pt"
    timestep = _timestep(_load_tensor(timestep_path) if timestep_path.is_file() else None, n_rows)
    return TransformerSurface(latents=latents, timestep=timestep, neutral=neutral, positive=positive)


def load_encoder_surface(model_dir: Path, n_rows: int, *, context: str) -> EncoderSurface:
    """Read paired AR frame hiddens the condition encoder consumes."""
    if int(n_rows) < 1:
        raise ValueError("encoder surface needs at least one prompt row")
    root = _surface_root(model_dir, "condition_encoder")
    required = [root / name for name in ("neutral.pt", "positive.pt")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise _missing("condition_encoder", model_dir, missing, context)
    neutral = _rows(_load_tensor(required[0]), n_rows, name=str(required[0]), row_rank=2)
    positive = _rows(_load_tensor(required[1]), n_rows, name=str(required[1]), row_rank=2)
    if neutral.shape != positive.shape:
        raise RuntimeError(
            "neutral and positive frame hiddens differ in shape "
            f"({tuple(neutral.shape)} vs {tuple(positive.shape)})."
        )
    return EncoderSurface(neutral=neutral, positive=positive)


def _parameter(model: torch.nn.Module) -> torch.nn.Parameter | None:
    try:
        return next(model.parameters())
    except StopIteration:
        return None


def _on_module(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    param = _parameter(model)
    if param is None:
        return tensor
    if tensor.is_floating_point():
        return tensor.to(device=param.device, dtype=param.dtype)
    return tensor.to(device=param.device)


def _forward_error(method: str, exc: BaseException) -> RuntimeError:
    return RuntimeError(
        f"{method} on the loaded module failed ({type(exc).__name__}: {exc}). "
        "The class that implements that method is the diffusers Music 3 module, "
        "which this repo does not vendor. "
        f"This command does not download {BASE_MODEL_ID}."
    )


def transformer_forward(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Call the loaded flow transformer. Returns velocity shaped like the latents."""
    config = getattr(model, "config", None)
    in_channels = getattr(config, "in_channels", None) if config is not None else None
    condition_dim = getattr(config, "condition_dim", None) if config is not None else None
    if in_channels is not None and int(hidden_states.shape[1]) != int(in_channels):
        raise RuntimeError(
            "Flow-VAE latents do not match the loaded transformer "
            f"in_channels ({int(hidden_states.shape[1])} vs {int(in_channels)})."
        )
    if condition_dim is not None and int(encoder_hidden_states.shape[-1]) != int(condition_dim):
        raise RuntimeError(
            "encoder_hidden_states do not match the loaded transformer "
            f"condition_dim ({int(encoder_hidden_states.shape[-1])} vs {int(condition_dim)})."
        )
    if int(hidden_states.shape[-1]) != int(encoder_hidden_states.shape[1]):
        raise RuntimeError(
            "Flow-VAE latent length and encoder_hidden_states length differ "
            f"({int(hidden_states.shape[-1])} vs {int(encoder_hidden_states.shape[1])})."
        )
    if int(hidden_states.shape[0]) != int(encoder_hidden_states.shape[0]):
        raise RuntimeError(
            "latent batch and encoder_hidden_states batch differ "
            f"({int(hidden_states.shape[0])} vs {int(encoder_hidden_states.shape[0])})."
        )
    latents = _on_module(model, hidden_states)
    step = _on_module(model, timestep.reshape(-1))
    cond = _on_module(model, encoder_hidden_states)
    try:
        out = model(latents, step, cond, return_dict=False)
    except RuntimeError:
        raise
    except Exception as exc:
        raise _forward_error(
            "MiniMaxMusic3Transformer1DModel.forward(hidden_states, timestep, encoder_hidden_states)",
            exc,
        ) from exc
    if isinstance(out, tuple):
        if not out or not isinstance(out[0], torch.Tensor):
            raise RuntimeError(
                "MiniMaxMusic3Transformer1DModel.forward returned no velocity tensor."
            )
        return out[0]
    sample = getattr(out, "sample", None)
    if not isinstance(sample, torch.Tensor):
        raise RuntimeError(
            "MiniMaxMusic3Transformer1DModel.forward returned "
            f"{type(out).__name__} without a velocity tensor."
        )
    return sample


def encoder_forward(model: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Call the loaded condition encoder."""
    config = getattr(model, "config", None)
    layers = getattr(config, "num_condition_layers", None) if config is not None else None
    width = getattr(config, "condition_hidden_dim", None) if config is not None else None
    if layers is not None and width is not None:
        expect = int(layers) * int(width)
        if int(hidden_states.shape[-1]) != expect:
            raise RuntimeError(
                "frame hiddens do not match the loaded condition encoder "
                f"(last dim {int(hidden_states.shape[-1])} vs "
                f"num_condition_layers * condition_hidden_dim = {expect})."
            )
    states = _on_module(model, hidden_states)
    try:
        out = model(states)
    except RuntimeError:
        raise
    except Exception as exc:
        raise _forward_error("MiniMaxMusic3ConditionEncoder.forward(hidden_states)", exc) from exc
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(
            "MiniMaxMusic3ConditionEncoder.forward returned "
            f"{type(out).__name__}, not a tensor."
        )
    return out


def pool_velocity(velocity: torch.Tensor) -> torch.Tensor:
    """Mean over latent length. ``[batch, channels, length]`` becomes ``[batch, channels]``."""
    if velocity.ndim != 3:
        raise RuntimeError(
            "MiniMaxMusic3Transformer1DModel.forward must return velocity "
            f"[batch, channels, length], got {tuple(velocity.shape)}."
        )
    return velocity.float().mean(dim=-1)


def pool_condition(encoded: torch.Tensor) -> torch.Tensor:
    """Mean over latent time. ``[batch, length, dim]`` becomes ``[batch, dim]``."""
    if encoded.ndim != 3:
        raise RuntimeError(
            "MiniMaxMusic3ConditionEncoder.forward must return "
            f"[batch, length, dim], got {tuple(encoded.shape)}."
        )
    return encoded.float().mean(dim=1)
