from pathlib import Path

import torch
from lightning import Callback


class CaptionSamplerCallback(Callback):
    """Generates sample captions for a fixed set of val images at each epoch end.

    Saves a PNG grid (image + generated caption + reference caption) and appends
    a text log — both in the Lightning log directory. Watching the PNG evolve
    across epochs is the main qualitative signal for how well the connector trains.

    Args:
        n_samples:      Number of val images to caption each epoch.
        max_new_tokens: Token budget for each generated caption.
        output_subdir:  Sub-directory inside the Lightning log dir.
    """

    def __init__(
        self,
        n_samples: int = 4,
        max_new_tokens: int = 64,
        output_subdir: str = "caption_samples",
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.output_subdir = output_subdir

        self._probe_pixel_values: torch.Tensor | None = None
        self._probe_images: list | None = None  # original PIL images for display
        self._probe_references: list[list[str]] | None = None  # [n_samples, 5]
        self._tokenizer = None
        self._prompt_ids: torch.Tensor | None = None  # text prefix passed to generate()
        self._out_dir: Path | None = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit":
            return
        log_dir = Path(trainer.log_dir or trainer.default_root_dir)
        self._out_dir = log_dir / self.output_subdir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._collect_probe(trainer)

    def _collect_probe(self, trainer) -> None:
        dm = trainer.datamodule
        val_ds = dm.data_val
        self._tokenizer = val_ds.tokenizer

        # Build the same prompt prefix the dataset trained on (user prompt +
        # assistant header). The LM will see this as in-distribution context.
        user_prompt = getattr(val_ds, "user_prompt", None)
        if user_prompt is not None:
            messages = [{"role": "user", "content": user_prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )  # rendered template string
            ids = self._tokenizer.encode(text, add_special_tokens=False)  # list[int]
            self._prompt_ids = torch.tensor(ids, dtype=torch.long)  # [P]

        image_key = val_ds.image_key  # "image" (Flickr8k) or "img" (CIFAR-10)
        stride = val_ds.items_per_image  # 5 (Flickr8k) or 1 (CIFAR-10)
        pixel_list, image_list, ref_list = [], [], []
        for i in range(min(self.n_samples, len(val_ds.hf_dataset))):
            row = val_ds.hf_dataset[i]
            sample = val_ds[i * stride]  # first flat entry for this image
            pixel_list.append(sample["pixel_values"])
            image_list.append(row[image_key].convert("RGB"))
            ref_list.append(val_ds.get_reference(row))

        self._probe_pixel_values = torch.stack(pixel_list)  # [n_samples, 3, 224, 224]
        self._probe_images = image_list
        self._probe_references = ref_list

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        device = pl_module.device
        pixel_values = self._probe_pixel_values.to(device)

        generated_ids = pl_module.generate_caption(
            pixel_values,
            prompt_ids=self._prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        captions = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        self._save_plot(captions, epoch)
        self._append_text_log(captions, epoch)

    def _save_plot(self, captions: list[str], epoch: int) -> None:
        import matplotlib.pyplot as plt

        n = len(captions)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Caption samples — epoch {epoch}", fontsize=11)

        for ax, img, caption, refs in zip(
            axes, self._probe_images, captions, self._probe_references, strict=False
        ):
            ax.imshow(img)
            ax.axis("off")
            # wrap long captions for readability
            gen_text = _wrap(caption.strip(), 40)
            ref_text = _wrap(refs[0], 40)
            ax.set_title(
                f"Generated:\n{gen_text}\n\nReference:\n{ref_text}",
                fontsize=7,
                loc="left",
                pad=6,
            )

        plt.tight_layout()
        fig.savefig(self._out_dir / f"epoch_{epoch:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _append_text_log(self, captions: list[str], epoch: int) -> None:
        lines = [f"=== Epoch {epoch} ==="]
        for i, (caption, refs) in enumerate(zip(captions, self._probe_references, strict=False)):
            lines.append(f"[Image {i + 1}]")
            lines.append(f"  Generated : {caption.strip()}")
            lines.append(f"  Reference : {refs[0]}")
            lines.append("")
        output = "\n".join(lines)
        print(f"\n{output}")
        with open(self._out_dir / "captions.txt", "a") as f:
            f.write(output + "\n")


def _wrap(text: str, width: int) -> str:
    """Naive word-wrap to keep titles readable."""
    words, lines, line = text.split(), [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = []
        line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)
