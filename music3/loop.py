"""CPU update loop for Music 3.

The routed adapter, global-mix critic, paired losses, particle VIC, noise
curve, and gradient cap all come from ``winning_formulation()``. This module
only wires a Music 3 prompt-state projection and an optimizer step.
"""

from __future__ import annotations

import torch
from torch import nn


class PromptStateAdapter(nn.Module):
    """One Music 3 projection: down, stamp bridge, up.

    A live run attaches one of these per host linear (Qwen3Attention or
    MiniMaxMusic3Attention). The dummy path uses a single projection so the
    stamp can step on CPU without Hub weights. The up matrix starts at zero,
    so scale 0 matches the frozen input.
    """

    def __init__(self, stamp, hidden: int):
        super().__init__()
        rank = int(stamp.spec["adapter_rank"])
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.up.weight)
        self.bridge = stamp.bridge()
        parts = int(stamp.spec["parts"])
        dim = int(stamp.spec["particle_dim"])
        self.particles = nn.Parameter(torch.randn(parts, dim) * 0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = self.down(hidden_states)
        routed = self.bridge(features, self.particles)
        return hidden_states + self.up(routed)


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"non-finite {name}")


def synthetic_states(rows: int, hidden: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand-in prompt states. Live training replaces these with LM hiddens."""
    generator = torch.Generator().manual_seed(int(seed))
    neutral = torch.randn(rows, hidden, generator=generator)
    positive = neutral + 0.5 * torch.randn(rows, hidden, generator=generator)
    return neutral, positive


def run_dummy(stamp, declared: dict, *, rows: int, steps: int, seed: int, hidden: int) -> tuple[list[dict], dict]:
    if rows < 2:
        raise ValueError("winning formulation needs at least two prompt rows")
    torch.manual_seed(int(seed))
    neutral, positive = synthetic_states(rows, hidden, seed)
    critic = stamp.critic(positive, neutrals=neutral)
    adapter = PromptStateAdapter(stamp, hidden)
    d_loss_fn, g_loss_fn, vic_fn = stamp.losses()
    regularizer = stamp.regularizer()
    built = {
        "bridge": adapter.bridge,
        "critic": critic,
        "regularizer": regularizer,
    }
    vic_weight = float(stamp.spec["vicreg_weight"])
    vic_batch = int(stamp.spec["particle_vic_batch"])
    betas = tuple(float(value) for value in declared["betas"])
    generator_params = [param for name, param in adapter.named_parameters() if name != "particles"]
    g_opt = torch.optim.Adam(generator_params, lr=float(declared["g_lr"]), betas=betas)
    p_opt = torch.optim.Adam([adapter.particles], lr=float(declared["particle_lr"]), betas=betas)
    d_opt = torch.optim.Adam(critic.parameters(), lr=float(declared["d_lr"]), betas=betas)
    batch = max(1, min(int(declared["adv_batch"]), rows))
    logs: list[dict] = []
    for step in range(1, int(steps) + 1):
        index = torch.randint(0, rows, (batch,))
        neu = neutral[index]
        tgt = positive[index]
        sigma = float(stamp.noise_std_at(step - 1, float(critic.edit_rms)))
        real_coords = critic.normalize(tgt).detach()

        adapter.requires_grad_(False)
        critic.requires_grad_(True)
        d_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            pred = adapter(neu)
        fake_coords = critic.normalize(pred).detach()
        epsilon = torch.randn_like(real_coords)
        real_in = (sigma * epsilon).detach()
        fake_in = (real_in + (fake_coords - real_coords)).detach()
        d_adv = d_loss_fn(critic(real_in), critic(fake_in))
        cap, _stats = regularizer.penalty(critic, real_in, fake_in, step=step, collect_stats=False)
        d_total = d_adv + cap
        _finite(d_total, "critic loss")
        d_total.backward()
        d_opt.step()

        adapter.requires_grad_(True)
        critic.requires_grad_(False)
        g_opt.zero_grad(set_to_none=True)
        p_opt.zero_grad(set_to_none=True)
        pred = adapter(neu)
        fake_coords = critic.normalize(pred)
        epsilon = torch.randn_like(real_coords)
        real_in = (sigma * epsilon).detach()
        fake_in = real_in + (fake_coords - real_coords)
        g_adv = g_loss_fn(critic(real_in).detach(), critic(fake_in))
        vic = vic_fn(adapter.particles[:vic_batch])
        g_total = g_adv + vic_weight * vic
        _finite(g_total, "generator loss")
        g_total.backward()
        g_opt.step()
        p_opt.step()
        logs.append(
            {
                "step": step,
                "sigma": sigma,
                "d_adv": float(d_adv.detach()),
                "d_penalty": float(cap.detach()),
                "g_adv": float(g_adv.detach()),
                "vic": float(vic.detach()),
                "g_lr": float(declared["g_lr"]),
                "d_lr": float(declared["d_lr"]),
                "particle_lr": float(declared["particle_lr"]),
            }
        )
    return logs, built
