# Shared PG 0.9 integration

`music3-particle-sliders` owns Music 3 metadata, prompts and CLI routing. The
executable training route delegates to the shared audio worker at commit
`9262a930cd351459e93ac33cc542fa379c12fe52`, which calls
`particle_sliders.ConditionalPG09` from shared core commit
`3e94b13eb5d6cd1ceee3f018f019a68603f35696`. ParticleGAN is exactly 0.9.0.
`shared.lock.json` and `core.lock.json` bind these local integration sources. This repository
does not implement a loss, optimizer, critic update, particle update or
checkpoint loop.

## Response and conditioning contract

The qualified CPU route constructs an actual random Qwen3 causal model
with hidden width 32, freezes its host weights and attaches the shared
Music 3 routed adapter. The existing 128×4 particle bank participates in
the hooked model forward and is owned once by the public optimizer.

Each row binds distinct neutral and positive token bytes and the same
continuation history, when present. The teacher is the frozen positive
final-token hidden state; the student uses the **neutral** tokens with
adapter scale +1. The response remains the native final hidden state,
without projection, truncation, field reshaping or a logits substitute.
The conditional critic receives the frozen neutral response, token
lengths, history length and unipolar adapter sign. Exact token, target and
condition bytes are checkpoint-bound by the shared game.

The generator objective is the public PG 0.9 adversarial loss plus its
public prior regularizer. The critic objective is its public adversarial
loss plus public penalty. No MSE, NMSE, caption loss or auxiliary teacher
objective is added by this product. Recipe construction, learning-rate
schedule, sampled rows, constraints and RNG belong to the shared game.
The bounded CPU route uses the `bispf_optimized` recipe with total budget 4;
`--steps` selects the absolute completion target within that fixed budget.

## Adapter scale, exports and continuation

This route is unipolar: scale 0 bypasses the adapter exactly, and +1 is the
trained treatment. CPU probes accept finite scales in [0,1], including
0.5; an intermediate response does not establish acoustic quality. The
historical energy demo's negative scales are not a PG 0.9 probe contract.

`checkpoint.pt` contains the complete shared training state. Resume must
use that full checkpoint and preserve its recipe horizon, source/runtime
identity, conditioning, targets and RNG. `adapter-export.pt` is the shared
inference export; it is loaded through the public export loader and is
not a conventional native or ComfyUI LoRA file. `qualification.pt` and
`progress.json` retain the shared worker's probes and execution evidence.
Product inference loads the export and writes hidden responses, not tokens,
latents or waveform audio.

## Native-runtime qualification

The pretrained Music 3 LM has hidden width 4096. Unchanged released PG 0.9
rejects flat continuous response widths of 170 or greater. Native planning
therefore requires the exact separately reviewed
`alz-continuous-v1`/`continuous` source and domain profile; the plan does
not authorize or install that profile. It does not replace the native
response with a smaller vector to avoid the domain guard.

Only LM/particle routing has a completed product binding. Transformer and
condition-encoder training fail closed rather than running the former
MSE/NMSE recipes or a synthetic proxy. Native pretrained smoke, exact
fresh-process resume/export replay on that runtime, acoustic rendering and
listening review remain separate gates owned by the coordinator.

The provisional `winning_formulation()` stamp may still supply the shared
routed adapter architecture and historical release metadata. It is not
proof that a CLI executed PG 0.9, and the old product-local synthetic game
is no longer a training route. Research trainers remain in the shared
checkout for their owners; this integration does not edit or dispatch them.
