# Music3 response formulation

Music3 supplies matched neutral and positive token histories to the installed
shared response lifecycle. The frozen positive final-token hidden state is the
teacher. The student uses the matched neutral history with the routed slider
enabled. The critic receives the frozen neutral response, token and history
lengths, and adapter sign.

The product owns the native Qwen3 host binding and audio conditioning. The shared
core owns the adversarial objective, optimizer, prior regularizer, sampled rows,
RNG, checkpoint and RAW export. Both the narrow CPU fixture and the full
4096-coordinate native response use the original 128 × 4 particle bank. Wide
responses require the explicitly configured continuous domain.

Use the [current response and installation guide](NATIVE-CONTINUOUS.md).
Earlier standalone training and probe commands remain available in the
[archived formulation and checkout](https://github.com/HyperGAN/music3-particle-sliders/blob/f23889b4fafd6a63efaff2ceeb7fc4c312f9f5d6/FORMULATION.md).
