"""Load a local Music 3 checkout and train the host linears ``stamp.bridge()`` is bound to.

``--dummy`` never imports this module's loader. Nothing here downloads Hub
weights. A transformer or condition-encoder checkout can bind, then stops
with a precise error: stepping those hosts needs the pipeline forward that
stays in HyperGAN/particle-sliders under ``conceptmod/textsliders/``.
The language-model host steps ``winning_formulation()`` on the last real
prompt token.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from particle_sliders import FormulationGame

from music3.attach import HostLinearBridge, bind_live_host
from music3.defaults import BASE_MODEL_ID

_AUDIO_START = "<|audio_start|>"
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
_INDEX_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
}
_TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json", "vocab.json")
_LAYOUT = {
    "lm": ("language_model", "language_model", True),
    "transformer": ("transformer", "transformer", False),
    "encoder": ("condition_encoder", "condition_encoder", False),
}


@dataclass
class LoadedHost:
    model: nn.Module
    tokenizer: Any
    kind: str
    model_dir: Path
    subfolder: str
    host_name: str


def assemble_music3_prompt(prompt: str, lyrics: str) -> str:
    """Music 3 chat string ending in ``<|audio_start|>``.

    Caption and lyric cleanup come from the installed diffusers Music 3
    encoders. This repo does not keep a second copy of that cleaner.
    """
    try:
        from diffusers.modular_pipelines.minimax_music3.encoders import (
            _clean_caption,
            _normalize_lyrics,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Live Music 3 prompt assembly needs diffusers "
            "modular_pipelines.minimax_music3.encoders "
            "(_clean_caption and _normalize_lyrics) so the prompt ends at "
            "<|audio_start|>. "
            f"Import failed ({exc}). "
            f"This command does not download {BASE_MODEL_ID}."
        ) from exc
    return (
        "<|im_start|><|caption_start|>"
        f"{_clean_caption(prompt)}<|caption_end|><|lyrics_start|>{_normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


@contextmanager
def _hub_offline():
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _no_download(message: str) -> RuntimeError:
    return RuntimeError(f"{message} This command does not download {BASE_MODEL_ID}.")


def _has_weights(directory: Path) -> bool:
    for path in directory.iterdir():
        if path.name in _INDEX_NAMES and path.stat().st_size > 0:
            return True
        if path.suffix.lower() in _WEIGHT_SUFFIXES and path.is_file() and path.stat().st_size > 0:
            return True
    return False


def _require_checkout(model_dir: Path, host_name: str) -> tuple[Path, str, bool]:
    if host_name not in _LAYOUT:
        raise RuntimeError(f"no live loader for Music 3 host {host_name!r}")
    sub_name, kind, needs_tokenizer = _LAYOUT[host_name]
    sub = model_dir / sub_name
    if not sub.is_dir() or not (sub / "config.json").is_file():
        raise _no_download(
            f"{model_dir} is missing {sub_name}/config.json. "
            f"The {host_name} host binds stamp.bridge() to the linears inside that "
            f"{kind} graph."
        )
    if not _has_weights(sub):
        raise _no_download(
            f"{sub} has config.json but no weight shards "
            "(a nonempty *.safetensors, *.bin, or index json). "
            f"stamp.bridge() cannot bind to {host_name} linears until those weights are local."
        )
    if needs_tokenizer:
        tokenizer = model_dir / "tokenizer"
        if not tokenizer.is_dir() or not any((tokenizer / name).is_file() for name in _TOKENIZER_MARKERS):
            raise _no_download(
                f"{model_dir} is missing tokenizer/ ({', '.join(_TOKENIZER_MARKERS)}). "
                "The language-model host reads prompt states from that tokenizer "
                "before it steps the Qwen3Attention bridges."
            )
    return sub, kind, needs_tokenizer


def _from_pretrained(cls, path: Path, dtype: torch.dtype):
    kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": dtype}
    try:
        return cls.from_pretrained(str(path), **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs.pop("torch_dtype")
        kwargs["dtype"] = dtype
        return cls.from_pretrained(str(path), **kwargs)


def _load_class(host_name: str):
    if host_name == "lm":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM, "transformers"
    if host_name == "transformer":
        from diffusers import MiniMaxMusic3Transformer1DModel

        return MiniMaxMusic3Transformer1DModel, "diffusers"
    try:
        from diffusers.models import MiniMaxMusic3ConditionEncoder
    except ImportError:
        from diffusers import MiniMaxMusic3ConditionEncoder

    return MiniMaxMusic3ConditionEncoder, "diffusers"


def load_live_host(model_dir: Path | str, host: dict) -> LoadedHost:
    """Load one local host graph. Fails closed when the checkout or library is missing."""
    root = Path(model_dir)
    host_name = str(host["name"])
    sub, kind, needs_tokenizer = _require_checkout(root, host_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        with _hub_offline():
            cls, _library = _load_class(host_name)
            model = _from_pretrained(cls, sub, dtype)
            tokenizer = None
            if needs_tokenizer:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    except ImportError as exc:
        library = "transformers" if host_name == "lm" else "diffusers"
        raise _no_download(
            f"Live {kind} training needs {library} to load {sub} and bind "
            f"stamp.bridge() to {host.get('target_replace')} linears. "
            f"Import failed ({exc})."
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise _no_download(
            f"Could not load the {kind} graph from {sub} with local_files_only. "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    return LoadedHost(
        model=model,
        tokenizer=tokenizer,
        kind=kind,
        model_dir=root,
        subfolder=sub.name,
        host_name=host_name,
    )


def _token_id(tokenizer, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        raise RuntimeError(
            f"The local tokenizer cannot map {token}. "
            "Live Qwen3Attention training needs the MiniMax-Music-3 tokenizer "
            "from the checkout."
        )
    try:
        tid = int(convert(token))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Tokenizer returned no id for {token}.") from exc
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        try:
            unknown = int(unk)
        except (TypeError, ValueError):
            unknown = None
        if unknown is not None and tid == unknown:
            raise RuntimeError(
                f"Tokenizer maps {token} to unk id {unknown}. "
                "The checkout tokenizer does not know the Music 3 audio-start token."
            )
    if tid < 0:
        raise RuntimeError(f"Tokenizer returned id {tid} for {token}.")
    return tid


def _tokenize(tokenizer, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    if "attention_mask" in encoded:
        mask = encoded["attention_mask"].to(device)
    else:
        mask = torch.ones_like(ids)
    return ids, mask


def _assert_audio_start(ids: torch.Tensor, mask: torch.Tensor, tokenizer) -> None:
    audio_id = _token_id(tokenizer, _AUDIO_START)
    index = int(mask[0].long().sum().item()) - 1
    if index < 0:
        raise RuntimeError("Music 3 prompt tokenized to an empty mask.")
    last = int(ids[0, index].item())
    if last != audio_id:
        raise RuntimeError(
            "Assembled Music 3 prompt does not end with <|audio_start|>. "
            f"Last token id was {last}; the tokenizer's <|audio_start|> id is {audio_id}. "
            "Refusing to step Qwen3Attention on that prompt."
        )


def _last_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.to(dtype=torch.long).sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, gather.clamp(min=0)).squeeze(1)


def _encode_prompt(loaded: LoadedHost, prompt: str, lyrics: str) -> torch.Tensor:
    if loaded.tokenizer is None:
        raise RuntimeError(
            "Language-model live training needs the checkout tokenizer to read prompt states."
        )
    device = next(loaded.model.parameters()).device
    text = assemble_music3_prompt(prompt, lyrics)
    ids, mask = _tokenize(loaded.tokenizer, text, device)
    _assert_audio_start(ids, mask, loaded.tokenizer)
    outputs = loaded.model(
        input_ids=ids,
        attention_mask=mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError(
            "The loaded language model did not return hidden states, so the "
            "Qwen3Attention bridges have no last-token prompt state to train. "
            f"Output type was {type(outputs).__name__}."
        )
    return _last_hidden(hidden_states[-1], mask).float().reshape(-1)


def _bridge_set(model: nn.Module) -> nn.Module:
    bridge_set = getattr(model, "_music3_stamp_bridges", None)
    if bridge_set is None or not list(bridge_set.bridges):
        raise RuntimeError(
            "Live step expected stamp.bridge() modules on the host. "
            "bind_live_host did not attach any linears."
        )
    return bridge_set


def _project_to_rank(states: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
    flat = states.reshape(states.shape[0], -1)
    return proj(flat)


def run_live(
    stamp,
    declared: dict,
    loaded: LoadedHost,
    bound: list[HostLinearBridge],
    *,
    prompts,
    steps: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Step ``FormulationGame`` on features from the bound host linears.

    Transformer and encoder bindings stay attached and then raise: the missing
    piece is the Music 3 pipeline forward, not the stamp.
    """
    if int(steps) < 1:
        raise ValueError("live steps must be positive")
    if len(prompts) < 2:
        raise ValueError("winning formulation needs at least two prompt rows")
    if not bound:
        raise RuntimeError("live training received no bound host linears")
    if loaded.kind != "language_model":
        shown = ", ".join(bridge.name for bridge in bound[:6])
        classes = sorted({bridge.host_class for bridge in bound})
        extra = "" if len(bound) <= 6 else f" (+{len(bound) - 6} more)"
        raise RuntimeError(
            f"Bound {len(bound)} stamp.bridge() modules to {classes} "
            f"under {loaded.model_dir / loaded.subfolder} ({shown}{extra}). "
            f"The {loaded.kind} graph is loaded. Stepping those linears needs the "
            "MiniMax Music 3 pipeline forward (flow-transformer latents and the "
            "condition encoder). That forward stays in HyperGAN/particle-sliders "
            "under conceptmod/textsliders/. The language-model host steps "
            "FormulationGame on the last <|audio_start|> token through "
            "the Qwen3Attention linears."
        )
    bridge_set = _bridge_set(loaded.model)
    particles = bridge_set.particles
    device = particles.device
    rank = int(stamp.spec["adapter_rank"])
    bridge_set.set_scale(0.0)
    with torch.no_grad():
        neutrals = torch.stack([_encode_prompt(loaded, row.neutral, row.lyrics) for row in prompts])
        positives = torch.stack([_encode_prompt(loaded, row.positive, row.lyrics) for row in prompts])
    if neutrals.shape != positives.shape:
        raise RuntimeError(
            "Neutral and positive prompt states have different shapes "
            f"({tuple(neutrals.shape)} vs {tuple(positives.shape)})."
        )
    proj = nn.Linear(int(neutrals.shape[-1]), rank, bias=False).to(device=device)
    proj.requires_grad_(False)
    with torch.no_grad():
        rank_neutrals = _project_to_rank(neutrals, proj)
        rank_positives = _project_to_rank(positives, proj)
    if torch.allclose(rank_neutrals, rank_positives):
        raise RuntimeError(
            "Neutral and positive last-token states have no paired edit after "
            "projection to adapter_rank, so FormulationGame cannot step the "
            "bound Qwen3Attention linears."
        )
    head = bound[0].bridge
    owned = {id(param) for param in head.parameters()}
    owned.add(id(particles))
    extra = [param for param in bridge_set.parameters() if id(param) not in owned]
    game = FormulationGame(
        stamp,
        declared,
        seed=int(seed),
        device=device,
        bridge=head,
        particles=particles,
        extra_generator=extra,
        critic_targets=rank_positives,
        critic_neutrals=rank_neutrals,
    )
    if not math.isfinite(game.edit_rms) or game.edit_rms <= 0:
        raise RuntimeError(
            "FormulationGame read a non-positive edit RMS from the prompt states, "
            "so it cannot step the bound host linears."
        )
    batch = max(1, min(int(declared["adv_batch"]), len(prompts)))

    def features_for_step(step: int) -> torch.Tensor:
        del step
        bridge_set.set_scale(1.0)
        index = torch.randint(0, len(prompts), (batch,))
        chosen = [prompts[int(item)] for item in index.tolist()]
        states = torch.stack([_encode_prompt(loaded, row.neutral, row.lyrics) for row in chosen])
        return _project_to_rank(states, proj)

    history = []
    for step in range(1, int(steps) + 1):
        row = game.step(step, features_for_step(step))
        history.append(
            {
                "step": int(step),
                "sigma": row["sigma"],
                "d_loss": row["d_loss"],
                "g_loss": row["g_loss"],
                "vic": row["vic"],
                "g_lr": float(declared["g_lr"]),
                "d_lr": float(declared["d_lr"]),
                "particle_lr": float(declared["particle_lr"]),
            }
        )
    built = {
        "bridge": game.bridge,
        "critic": game.critic,
        "regularizer": game.regularizer,
    }
    return history, built
