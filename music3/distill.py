"""Bounded Music3 activation collection for the shared ordinary-LoRA fitter.

This is a post-training approximation, not a GAN objective or a routed export
conversion. It returns CPU factors and provenance; it does not publish weights.
"""
import hashlib
import json
import math
from copy import deepcopy


def distill_lora(adapter, calibration_ids, *, max_rows=128, ridge_fraction=.01):
    """Fit separate q/k/v/o native LoRANetwork factors on explicit token contexts.

    Only the currently admitted tiny CPU cohort is supported. Each target uses
    at most ``max_rows`` token activations, in caller order. The shared fitter
    approximates the routed bottleneck and retains its up matrix. The returned
    native keys can be loaded by shared ``LoRANetwork``; they are not PEFT or
    ComfyUI keys. No teacher weights, forwards, modes, gradients or RNG change.
    """
    from .response_source import CORE_COMMIT, verify_response_core
    verify_response_core()
    import torch
    from particle_sliders.distillation import fit_routed_down
    from .native import ParticleNetwork
    from particle_sliders.pg09 import _hash
    from .response_adapter import Music3ARResponseAdapter
    from particle_sliders.pg09_audio import require_audio_width
    from .lifecycle import _ids, observe_native

    if not isinstance(adapter, Music3ARResponseAdapter) or type(adapter.slider) is not ParticleNetwork:
        raise ValueError("distillation requires the reviewed Music3 routed adapter")
    require_audio_width(adapter.model.config.hidden_size)
    if adapter.model.device.type != "cpu" or not adapter.model_identity.get("fixture"):
        raise ValueError("only the admitted random CPU fixture is qualified for distillation")
    if type(max_rows) is not int or not 1 <= max_rows <= 512:
        raise ValueError("max_rows must be an integer in [1,512]")
    if type(ridge_fraction) not in (int, float) or not math.isfinite(ridge_fraction) or ridge_fraction <= 0:
        raise ValueError("ridge_fraction must be finite and positive")
    if not isinstance(calibration_ids, (list, tuple)) or not 1 <= len(calibration_ids) <= 128:
        raise ValueError("supply 1..128 explicit calibration contexts")
    contexts = tuple(_ids(ids) for ids in calibration_ids)
    for ids in contexts:
        if len(ids) > min(256, adapter.model.config.max_position_embeddings) or max(ids) >= adapter.model.config.vocab_size:
            raise ValueError("complete calibration context exceeds native token budget; no truncation")
    network = adapter.slider
    teacher_sha256 = _hash(network.state_dict())
    captures = {lora.lora_name: [[], [], 0] for lora in network.network.unet_loras}
    handles = []
    state, residuals = {}, {}
    rng = torch.get_rng_state().clone()

    def capture(lora):
        def hook(module, args):
            xs, ys, count = captures[lora.lora_name]
            remaining = max_rows - count
            if remaining <= 0:
                return
            x = args[0].detach().to(lora.lora_down.weight).reshape(-1, lora.lora_down.in_features)[:remaining]
            bridge = network.bridges[lora.lora_name.replace(".", "-")]
            y = bridge(lora.lora_down(x).float(), network.particles)
            xs.append(x.float().cpu().clone())
            ys.append(y.float().cpu().clone())
            captures[lora.lora_name][2] += len(x)
        return hook

    try:
        with torch.no_grad(), observe_native(adapter.model, network):
            for lora in network.network.unet_loras:
                host = getattr(lora.org_forward, "__self__", None)
                if host is None or host.forward is not lora.forward:
                    raise RuntimeError("Music3 routed forward ownership differs")
                handles.append(host.register_forward_pre_hook(capture(lora)))
            for ids in contexts:
                adapter._response(ids, scale=1.)
            for lora in network.network.unet_loras:
                xs, ys, count = captures[lora.lora_name]
                if not count:
                    raise RuntimeError("calibration did not visit target " + lora.lora_name)
                x, y = torch.cat(xs), torch.cat(ys)
                down, ridge = fit_routed_down(x, y, ridge_fraction=ridge_fraction)
                prefix = lora.lora_name
                state[prefix + ".lora_down.weight"] = down.contiguous()
                state[prefix + ".lora_up.weight"] = lora.lora_up.weight.detach().float().cpu().clone()
                state[prefix + ".alpha"] = lora.alpha.detach().cpu().clone()
                residuals[prefix] = dict(rows=count, ridge=float(ridge),
                    calibration_max_abs=float((x @ down.T - y).abs().max()))
    finally:
        for handle in reversed(handles):
            handle.remove()
        torch.set_rng_state(rng)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("ordinary LoRA factors must be finite")
    provenance = dict(format="music3-ordinary-lora-distill-v1", approximation=True,
        native_audio_qualified=False, qualification="tiny random CPU software fixture only",
        core_commit=CORE_COMMIT, teacher_adapter_sha256=teacher_sha256,
        model_identity=deepcopy(adapter.model_identity), factor_sha256=_hash(state),
        calibration_sha256=hashlib.sha256(json.dumps(contexts).encode()).hexdigest(),
        contexts=len(contexts), max_rows=max_rows, ridge_fraction=ridge_fraction,
        trained_teacher_strength=1., strengths=[0., 1.], targets=residuals,
        key_format="native LoRANetwork lora_te; separate q/k/v/o; not PEFT/ComfyUI",
        shared_fitter="particle_sliders.distillation.fit_routed_down")
    return state, provenance
