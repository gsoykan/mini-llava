<div align="center">

# mini-VLM

A **stage-1 LLaVA-style** Vision-Language Model trained on CIFAR-10.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/uv-package%20manager-blueviolet"></a>

</div>

<br>

## What This Is

The smallest code path that still implements the real VLM recipe:

> *project a vision encoder into a frozen LLM's embedding space, and only train the projection.*

```
CIFAR image
   → CLIP ViT-B/32 (frozen, ~87M)
   → CLS token [B, 1, 768]
   → MLP Connector (trained, ~1.2M)
   → [B, 1, 576] visual prefix
   → prepended to a chat-template prompt
   → SmolLM2-135M-Instruct (frozen, ~135M)
   → "a photo of a cat"
```

Only the **~1.2M-parameter MLP connector** is trained. The vision encoder and the LLM are completely frozen. Trains in minutes on a MacBook (MPS).

Companion project to the [inzva Applied AI Study Group 10 presentation](https://drive.google.com/file/d/1_19yDdZogyrrHMA_cb9ZiFtd8d_KmYsP/view?usp=sharing), Block 5.

<br>

## Architecture

| Component | Module | Frozen? | Params |
|---|---|---|---|
| Vision encoder | `openai/clip-vit-base-patch32` | ✅ | ~87M |
| MLP connector | `LayerNorm → Linear(768→576) → GELU → Linear(576→576)` | trained | ~1.2M |
| Language decoder | `HuggingFaceTB/SmolLM2-135M-Instruct` | ✅ | ~135M |

The connector follows LLaVA-1.5's `mlp2x_gelu` design with a prenorm. Both frozen models are loaded in **fp32** explicitly to avoid bf16 backprop overflow through the LLM (see [REPORT.md](REPORT.md) §3.3).

Training uses a chat-template format with **loss masked on the user prompt**:

```
<|im_start|>user
Describe this image briefly.<|im_end|>
<|im_start|>assistant
a photo of a cat<|im_end|>
```

Only the assistant's response contributes to cross-entropy loss.

<br>

## Quick Start

### Install

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync --all-extras
```

### Train

```bash
# Stage-1 connector training on CIFAR-10 (10 epochs, MPS, ~5 minutes on M-series)
uv run python src/train.py experiment=mini_vlm_cifar10

# Overrides
uv run python src/train.py experiment=mini_vlm_cifar10 trainer.max_epochs=20
uv run python src/train.py experiment=mini_vlm_cifar10 data.max_train_samples=null  # full 50k
uv run python src/train.py experiment=mini_vlm_cifar10 trainer=gpu                  # CUDA
```

Logs go to `logs/train/runs/<timestamp>/`. Checkpoints land in `logs/train/runs/<timestamp>/checkpoints/`.

### Inference

Open `notebooks/inference.ipynb`. It loads the latest connector checkpoint, embeds CIFAR-10 test images through CLIP → connector, prepends the visual prefix to the chat template, and greedy-generates captions for a grid of samples.

For a side-by-side comparison with the production SmolVLM2 model trained end-to-end on web-scale data, see `notebooks/smolvlm2_inference.ipynb`.

<br>

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/inference.ipynb` | mini-VLM inference on CIFAR-10 (frozen-frozen + trained connector) |
| `notebooks/smolvlm2_inference.ipynb` | Same task with `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` for comparison |

The two notebooks share the same architectural shape (vision encoder → projector → LM). The quality gap is the lesson: what end-to-end training on massive multimodal data buys you over a frozen-frozen connector trained on 5k images.

<br>

## Project Structure

```
mini-vlm/
├── configs/                          <- Hydra configs
│   ├── data/cifar10.yaml
│   ├── model/mini_vlm.yaml
│   ├── experiment/mini_vlm_cifar10.yaml
│   └── trainer/mps.yaml              <- Apple Silicon (default for this project)
│
├── src/
│   ├── data/
│   │   ├── cifar10_datamodule.py
│   │   └── components/cifar10_dataset.py
│   ├── models/
│   │   ├── mini_vlm_module.py        <- LightningModule (forward, loss, generate)
│   │   └── components/
│   │       ├── vision_encoder.py     <- frozen CLIP, CLS or patch tokens
│   │       ├── mlp_connector.py      <- trained, LLaVA-1.5 mlp2x_gelu + prenorm
│   │       └── language_decoder.py   <- frozen SmolLM2 with visual prefix prepending
│   ├── callbacks/
│   │   └── caption_sampler_callback.py  <- log generated captions during training
│   └── train.py
│
├── notebooks/                        <- inference demos
├── REPORT.md                         <- build log + every bug we hit, with root causes
└── pyproject.toml                    <- uv + ruff + pytest config
```

<br>

## What Could Bite You (and How We Solved It)

Four real problems we hit while building this. Every one of them generalises to "real" VLM training. Full root-cause analyses live in [REPORT.md](REPORT.md).

| Bug | Symptom | Fix |
|---|---|---|
| bf16 backprop through frozen LLM | NaN gradient at the connector after ~step 1 | Load the LM in **fp32** — frozen ≠ no backward pass |
| One NaN, all parameters dead | `val/loss: nan` forever after a single bad batch | `on_before_optimizer_step` zeros non-finite grads before AdamW corrupts its running averages |
| Captions are encyclopedia text | Generated *"autonomic (n) — the word autonomic..."* | Use **SmolLM2-135M-Instruct** + chat template (base LM treats visual prefix as document continuation) |
| Avg-pool ≠ CLS | LM ignores image | Use **CLS token** directly — CLIP's trained global summary, not an arbitrary average of patches |

These four lessons transfer directly to building any VLM connector, regardless of scale.

<br>

## Related

- [SmolVLM — Marafioti et al., 2025](https://arxiv.org/abs/2504.05299) — the production version of this idea, scaled up
- [Visual Instruction Tuning — Liu et al., 2023](https://arxiv.org/abs/2304.08485) — the original LLaVA paper
- [Improved Baselines with Visual Instruction Tuning — Liu et al., 2024](https://arxiv.org/abs/2310.03744) — LLaVA-1.5, source of the `mlp2x_gelu` connector design
- [SmolLM2 — Allal et al., 2025](https://arxiv.org/abs/2502.02737) — the language model we use as the frozen LM backbone

<br>

## Acknowledgements

Built on [`ai-lightning-hydra-template`](https://github.com/gsoykan/ai-lightning-hydra-template), itself adapted from [`ashleve/lightning-hydra-template`](https://github.com/ashleve/lightning-hydra-template).

<br>

## License

MIT — see [LICENSE](LICENSE).
