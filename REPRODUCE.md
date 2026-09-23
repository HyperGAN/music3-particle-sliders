# Reproduce

CPU smoke does not download Hub weights and does not need a Music 3 checkout.

```bash
python -m pip install -r requirements.txt
python scripts/train_music3.py --dummy
python scripts/infer_music3.py --dummy
python scripts/train_lora_music3.py --help
python scripts/train_lm_slider_music3.py --help
python scripts/train_encoder_music3.py --help
python scripts/train_lora_music3_particle.py --help
pytest
```

`--dummy` runs four winning-formulation steps on synthetic prompt states and
writes `models/music3-slider/<name>_last.json`. The bridge and critic modules
recorded there must be `particle_sliders.reference`. The regularizer module
must be `particlegan`, not a class defined in this repo.

## Live weights

Base model: [MiniMaxAI/MiniMax-Music-3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).

Adapter release:
[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)
(`weights/uni16-fresh-selected-v2/`, `comfyui/uni16-fresh-selected-v2/`,
`samples/uni16-fresh-selected-v2/`).

```bash
python scripts/train_music3.py --live --model_dir /path/to/MiniMax-Music-3
```

That command checks the directory and stops. This skeleton requires the
stamp, then refuses to load the 8B graph. Wiring each host linear to
`stamp.bridge()` is a follow-up once a local pipeline is available. Do not
point the loader at a second copy of the game inside a particle-sliders
checkout.

`--lm_target v9`, `--loss nmse`, and encoder rank 64 are rejected. Those
recipes remain in HyperGAN/particle-sliders `conceptmod/textsliders/`.

## Follow-up in particle-sliders

This task does not open a pull request on HyperGAN/particle-sliders and does
not delete the Music 3 trainers there. A one-line README pointer from that
repo to `https://github.com/HyperGAN/music3-particle-sliders` is still worth
adding:

```markdown
Music 3 product: [music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders).
```
