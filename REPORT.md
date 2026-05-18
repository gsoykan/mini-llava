# mini-VLM: Build Log & Problem-Solving Report

A stage-1 LLaVA-style Vision-Language Model trained on CIFAR-10 using
frozen CLIP + frozen LLM + a trainable MLP connector.

---

## 1. Goal

Build a minimal, pedagogically clear VLM from scratch:

```
CLIP ViT-B/32 (frozen) → MLP Connector (trained) → SmolLM2-135M-Instruct (frozen)
```

Only the MLP connector (~1.2 M parameters) is trained. It learns to project
CLIP's patch embeddings into the LLM's embedding space so that the frozen LLM
can autoregressively generate the correct class caption after seeing the image.

Task: CIFAR-10 images → "a photo of a {class}" captions via chat template.

---

## 2. Architecture

### Vision Encoder — `VisionEncoder`
- Model: `openai/clip-vit-base-patch32` (frozen, ~87 M params)
- Output: CLS token `[B, 1, 768]` — CLIP's trained global image summary
- Always runs in `float32` even if the rest of the pipeline is mixed precision;
  CIFAR-10 images are 32×32 upscaled to 224×224 which produces near-uniform
  patches that cause NaN in bf16 softmax

### MLP Connector — `MLPConnector`
- Architecture: `LayerNorm(768) → Linear(768→576) → GELU → Linear(576→576)`
  (matches LLaVA-1.5 mlp2x_gelu with prenorm)
- Only trained component; ~1.2 M parameters
- No spatial pooling needed when using the CLS token (`pool_size=None`)
- Output-layer initialized with `std=0.02` / zero bias to keep visual prefix
  near zero at step 0 and prevent early NaN loss

### Language Decoder — `LanguageDecoder`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (frozen, ~135 M params)
- Loaded in `float32` explicitly (`torch_dtype=torch.float32`); the HF config
  defaults to bfloat16 which produces NaN gradients on backprop
- `attn_implementation="eager"` avoids the MPS SDPA kernel bug
- `use_cache=False` ensures consistent behavior between train and eval modes

### Training format (chat template)
```
<|im_start|>user
Describe this image briefly.<|im_end|>
<|im_start|>assistant
a photo of a cat<|im_end|>
```
- User prompt + assistant header are masked from the loss (labels = -100)
- Only the assistant's response contributes to the cross-entropy loss

---

## 3. Problems Encountered & Solutions

### Problem 1: MPS `adaptive_avg_pool2d` crash
**Symptom:** Runtime error when using spatial pooling with `pool_size` that
didn't evenly divide the 7×7 grid.

**Root cause:** MPS requires `adaptive_avg_pool2d` input size to be divisible by
the output size. 7 ÷ 3 fails.

**Fix:** Replaced `F.adaptive_avg_pool2d` with `F.avg_pool2d(kernel_size=k, stride=k)`
where `k = H // pool_size`. Regular pool2d has no divisibility constraint.

---

### Problem 2: Training NaN loss (gradient explosion)
**Symptom:** `train/loss: nan` after a few steps when `pool_size=2` (4 tokens).

**Root cause:** Gradient flowing backward through 30 frozen transformer layers
amplified in bf16 → connector weights became NaN after the first optimizer step.
AdamW's running averages `m` and `v` then stayed NaN permanently.

**Fix 1:** `pool_size=3` (9 tokens) gave more stable gradient signal than 4 tokens.

**Fix 2:** `gradient_clip_val=1.0` in trainer (limits finite gradients, doesn't
help with NaN but catches magnitude spikes).

---

### Problem 3: Validation loss always NaN (train fine, val broken)

This was the longest-running bug. Full diagnosis trace:

**Symptom:** `val/loss: nan`, `test/loss: nan` every epoch despite training loss
being healthy (2.0–4.5).

**Attempted fixes that didn't work:**
- `num_sanity_val_steps: 0` — still NaN
- `use_cache=False` — still NaN
- `attn_implementation="eager"` — still NaN
- `inference_mode: False` in trainer — still NaN
- `torch.enable_grad()` wrapper in connector — still NaN

**Root cause (finally found via fine-grained debug prints):**

All connector parameters (LayerNorm, Linear ×2) were NaN by the time
validation started — meaning a NaN gradient had hit the connector during
training and AdamW had propagated it to all parameters.

The gradient was NaN because SmolLM2's HF config specifies
`torch_dtype: bfloat16`, so the model loaded in bf16 by default. The
backward pass through 30 attention layers in bf16 overflowed and sent NaN
gradients back to the connector. `gradient_clip_val=1.0` couldn't help:
`clip_grad_norm_` of a NaN tensor returns NaN coefficients and corrupts all
gradients.

**Fixes:**

1. **Load LM in float32**: `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)` — eliminates the bf16 overflow in the backward pass.

2. **NaN-gradient safeguard** in `on_before_optimizer_step`: detects NaN/Inf
   gradients and zeros them before AdamW sees them, so a single bad batch
   can't permanently corrupt all parameters.

---

### Problem 4: Captions were pure LLM priors, not vision-conditioned
**Symptom:** Generated captions like *"autonomic (n) — The word autonomic is
derived from the Greek…"* and *"The following is a list of the most common
terms…"* — classic SmolLM2 pretraining distribution (Wikipedia / dictionary).

**Root cause 1 — Wrong visual token:** We used `avg_pool2d` of 49 patches into
a single token. CLIP's patch tokens are *not* trained to be meaningfully
averageable — the average is a weak global signal. CLIP has a dedicated
global summary token: the CLS token.

**Fix:** `use_cls=True` in `VisionEncoder` — return `last_hidden_state[:, 0:1, :]`
instead of the averaged patches. CLS is CLIP's trained global representation,
perfect for a classification-style task.

**Root cause 2 — Base LM ignores soft prompts:** `SmolLM2-135M` (base) has no
concept of instruction following. Feeding `[visual_token, BOS]` and hoping it
produces a caption is entirely out-of-distribution — the base LM just continues
with whatever fits its pretraining prior.

**Fix:** Switch to `SmolLM2-135M-Instruct` and format all inputs with the
model's chat template. During training the model sees the full conversation
formatted as:

```
<|im_start|>user
Describe this image briefly.<|im_end|>
<|im_start|>assistant
a photo of a cat<|im_end|>
```

At generation time the same user prompt + assistant header is prepended after
the visual prefix, so the LM is in-distribution and generates the caption
rather than encyclopedia text.

**Root cause 3 — Learning rate too high:** `lr=2e-4` caused loss oscillation and
unstable training with limited data.

**Fix:** `lr=2e-5` (a conservative rate given the small connector, simple task,
and 5k training samples).

---

## 4. Final Configuration

| Component | Value |
|---|---|
| Vision encoder | CLIP ViT-B/32, frozen, fp32 |
| Visual tokens | 1 (CLS token) |
| Connector | LayerNorm → Linear(768→576) → GELU → Linear(576→576) |
| Language model | SmolLM2-135M-Instruct, frozen, fp32 |
| Text format | Chat template, loss only on assistant response |
| Optimizer | AdamW, lr=2e-5, weight_decay=0 |
| Scheduler | CosineAnnealingLR, T_max=epochs, eta_min=1e-5 |
| Precision | 32-true (full fp32) |
| Gradient clip | 1.0 (trainer) + NaN-grad safeguard (on_before_optimizer_step) |
| Accelerator | MPS (Apple Silicon), inference_mode=False |
| Training data | CIFAR-10, 5000 train samples |
| Max sequence length | 48 tokens (chat template + caption) |

---

## 5. Why It Works

The connector learns a mapping `f: R^768 → R^576` such that CLIP's CLS
embedding, when passed through `f`, lands in a region of the LLM's embedding
space that the Instruct model associates with the correct class word.

More concretely:

1. **CLIP CLS** encodes the image's global semantics — across 10 CIFAR classes
   these 768-dimensional embeddings are well-separated (CLIP was trained on
   classification-like contrastive objectives).

2. **The connector** linearly projects and normalizes those embeddings into the
   LLM's 576-dimensional space. The prenorm `LayerNorm(768)` aligns
   distributions before the cross-modal projection; the small output-layer
   initialization keeps the visual prefix near the LLM's zero-embedding at
   step 0, avoiding early divergence.

3. **The Instruct LM** sees `[visual_embed, <user prompt>, <assistant header>]`
   and has learned during SFT to follow the user's request using whatever
   prefix context is available. As the connector trains, the visual embed
   increasingly encodes "this is a cat" and the LM generates "a photo of a cat".

4. **Stability**: fp32 throughout (no bf16 overflow), NaN-grad safeguard (no
   parameter corruption from bad batches), conservative lr (smooth convergence).

---

## 6. Lessons Learned

- **Base LMs vs Instruct LMs**: A base completion model treats `[visual_token, BOS]`
  as the start of any document. An instruction-tuned model treats it as a prompt
  to respond to. For VLMs the visual prefix *is* the prompt — use Instruct.

- **CLS vs averaged patches**: Avg-pooling patch tokens is not the same as CLS.
  CLIP's CLS is specifically trained to summarize the whole image. For single-token
  global tasks (classification), CLS is the right choice.

- **bf16 backprop through frozen layers**: Even with `requires_grad=False`, frozen
  layers in bf16 run the backward pass in bf16. That's 30 attention layers of
  potential overflow. Always load frozen LMs in fp32 when stability matters more
  than memory.

- **AdamW and NaN gradients**: One NaN gradient permanently corrupts all parameters
  (via running averages `m` and `v`). `gradient_clip_val` cannot fix this — it
  must be detected and zeroed before the optimizer step.

- **MPS quirks**: Apple's MPS backend has several kernel divergences from CUDA:
  `adaptive_avg_pool2d` (divisibility constraint), SDPA (NaN in no_grad mode),
  and inference_mode (different kernel selection). On MPS: use `avg_pool2d`,
  `attn_implementation="eager"`, and `inference_mode=False`.
