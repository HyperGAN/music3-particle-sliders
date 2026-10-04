"""Bind ``stamp.bridge()`` onto Music 3 host linears.

Host modules are selected by class name (``Qwen3Attention``,
``MiniMaxMusic3Attention``, ``MiniMaxMusic3TransformerBlock``,
``MiniMaxMusic3Transformer1DModel``, ``MiniMaxMusic3ConditionEncoder``).
Every ``nn.Linear`` inside those modules is a training target. On the
language-model graph those linears are ``q_proj``, ``k_proj``, ``v_proj``,
and ``o_proj``. On the flow-transformer attention block they are ``to_q``,
``to_k``, ``to_v``, and ``to_out``. Full transformer targets also include
feed-forward and ``proj_in`` linears. Convolutions stay frozen.

``--dummy`` does not call this module. A live run and the CPU stand-in test
both call :func:`attach_stamp_bridge`.
"""

from __future__ import annotations

import torch
from torch import nn

# Published projection names. The walker binds every Linear under the host
# class; these strings are the error hint, not a second filter.
_LINEAR_HINTS = {
    "Qwen3Attention": "q_proj, k_proj, v_proj, o_proj",
    "MiniMaxMusic3Attention": "to_q, to_k, to_v, to_out",
    "MiniMaxMusic3TransformerBlock": "attention linears and the block feed-forward",
    "MiniMaxMusic3Transformer1DModel": "root linears such as proj_in",
    "MiniMaxMusic3ConditionEncoder": (
        "nn.Linear children only; the published proj is Conv1d and stays unbound"
    ),
}


class _ParticleRef:
    """Hold the shared particle Parameter without registering it twice."""

    def __init__(self, param: nn.Parameter):
        self.param = param


class HostLinearBridge(nn.Module):
    """Down, ``stamp.bridge()``, up, added to one frozen host linear.

    Scale 0 returns the host linear alone, so the frozen Music 3 forward
    stays exact. The up matrix starts at zero.
    """

    def __init__(self, stamp, linear: nn.Module, particles: _ParticleRef, name: str, host_class: str):
        super().__init__()
        rank = int(stamp.spec["adapter_rank"])
        alpha = float(stamp.spec["adapter_alpha"])
        self.name = name
        self.host_class = host_class
        self.rank_scale = alpha / float(rank)
        self.scale = 1.0
        self.down = nn.Linear(int(linear.in_features), rank, bias=False)
        self.up = nn.Linear(rank, int(linear.out_features), bias=False)
        nn.init.zeros_(self.up.weight)
        self.bridge = stamp.bridge()
        self._particles = particles
        self.org_forward = linear.forward

    def forward(self, x):
        base = self.org_forward(x)
        if self.scale == 0.0:
            return base
        weight = self.down.weight
        features = self.down(x.to(device=weight.device, dtype=weight.dtype))
        routed = self.bridge(features, self._particles.param)
        delta = self.up(routed).to(device=base.device, dtype=base.dtype)
        return base + delta * (self.scale * self.rank_scale)


class StampBridgeSet(nn.Module):
    """One shared particle cloud and one stamp bridge per host linear."""

    def __init__(self, stamp):
        super().__init__()
        parts = int(stamp.spec["parts"])
        dim = int(stamp.spec["particle_dim"])
        self.particles = nn.Parameter(torch.randn(parts, dim) * 0.02)
        self.bridges = nn.ModuleList()
        self.scale = 1.0

    def set_scale(self, scale: float) -> None:
        value = float(scale)
        self.scale = value
        for bridge in self.bridges:
            bridge.scale = value


def _is_host_linear(module: nn.Module) -> bool:
    if getattr(module, "_music3_stamp_bridge", None) is not None:
        return False
    if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
        return False
    return isinstance(module, nn.Linear) or module.__class__.__name__ in {"Linear", "LoRACompatibleLinear"}


def _hint(target_class_names: list[str]) -> str:
    parts = []
    for name in target_class_names:
        detail = _LINEAR_HINTS.get(name)
        parts.append(f"{name} ({detail})" if detail else name)
    return "; ".join(parts)


def matching_host_classes(root: nn.Module, target_class_names: list[str]) -> list[str]:
    wanted = set(target_class_names)
    found: list[str] = []
    for module in root.modules():
        name = module.__class__.__name__
        if name in wanted and name not in found:
            found.append(name)
    return found


def attach_stamp_bridge(root: nn.Module, stamp, target_class_names: list[str]) -> list[HostLinearBridge]:
    """Attach one ``stamp.bridge()`` to each host linear. Returns the bridges.

    Overlapping classes (attention inside a block inside the root model) bind
    each linear once. The returned modules are the same objects ``--live`` trains.
    """
    if not isinstance(root, nn.Module):
        raise TypeError("stamp.bridge() attaches to an nn.Module host graph")
    wanted = [str(name) for name in target_class_names]
    if not wanted:
        raise ValueError("host target list is empty")
    wanted_set = set(wanted)
    matches: list[tuple[str, str, nn.Module]] = []
    seen: set[int] = set()
    for name, module in root.named_modules():
        if module.__class__.__name__ not in wanted_set:
            continue
        host_class = module.__class__.__name__
        for child_name, child in module.named_modules():
            if not _is_host_linear(child):
                continue
            if id(child) in seen:
                continue
            seen.add(id(child))
            qual = ".".join(part for part in (name, child_name) if part)
            matches.append((qual, host_class, child))
    if not matches:
        return []
    bridge_set = StampBridgeSet(stamp)
    particles = _ParticleRef(bridge_set.particles)
    for qual, host_class, linear in matches:
        bridge = HostLinearBridge(stamp, linear, particles, qual, host_class)
        # A normal assignment would register the set as a child and cycle.
        object.__setattr__(bridge, "_stamp_bridge_set", bridge_set)
        bridge_set.bridges.append(bridge)
        linear._music3_stamp_bridge = bridge  # noqa: SLF001
        linear.forward = bridge.forward
    device = matches[0][2].weight.device
    bridge_set.to(device=device)
    root.add_module("_music3_stamp_bridges", bridge_set)
    bridge_set.set_scale(1.0)
    return list(bridge_set.bridges)


def _unbound_convs(root: nn.Module, target_class_names: list[str]) -> list[str]:
    wanted = set(target_class_names)
    found: list[str] = []
    for name, module in root.named_modules():
        if module.__class__.__name__ not in wanted:
            continue
        for child_name, child in module.named_modules():
            if not isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            qual = ".".join(part for part in (name, child_name) if part)
            label = f"{qual} ({child.__class__.__name__})"
            if label not in found:
                found.append(label)
    return found


def bind_live_host(root: nn.Module, stamp, target_class_names: list[str]) -> list[HostLinearBridge]:
    """Attach, or raise a precise error when the loaded graph has no host linears."""
    bound = attach_stamp_bridge(root, stamp, target_class_names)
    if bound:
        return bound
    found = matching_host_classes(root, target_class_names)
    present = sorted({module.__class__.__name__ for module in root.modules()})
    hint = _hint(list(target_class_names))
    if found:
        convs = _unbound_convs(root, list(target_class_names))
        conv_note = ""
        if convs:
            conv_note = f" Unbound convolutions: {convs}. This attach does not retarget convolutions."
            if "MiniMaxMusic3ConditionEncoder" in found:
                conv_note += (
                    " The published MiniMaxMusic3ConditionEncoder.proj is Conv1d, not nn.Linear."
                )
        raise RuntimeError(
            f"Found Music 3 host modules {found} but they contain no nn.Linear "
            f"children. stamp.bridge() binds to those linears ({hint}). "
            f"Classes present: {present}.{conv_note}"
        )
    raise RuntimeError(
        f"Loaded module has none of the Music 3 host classes {list(target_class_names)}. "
        f"stamp.bridge() binds to nn.Linear children of {hint}. "
        f"Classes present: {present}."
    )
