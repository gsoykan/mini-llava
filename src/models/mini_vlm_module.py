from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.language_decoder import LanguageDecoder
from src.models.components.mlp_connector import MLPConnector
from src.models.components.vision_encoder import VisionEncoder


class MiniVLMModule(LightningModule):
    """`LightningModule` for Stage-1 mini-VLM (LLaVA-style) connector pretraining.

    Only the MLP connector is trained. The vision encoder (CLIP ViT-B/32) and
    the language decoder (SmolLM2-135M) are fully frozen.

    The connector learns to project CLIP patch tokens into the LLM's embedding
    space so that a frozen LLM can autoregressively generate the image caption.

    Parameter count breakdown:
        Vision encoder   — ~87 M  (frozen)
        MLP connector    —  ~2.8 M  (trained)
        Language decoder — ~135 M  (frozen)
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        connector: MLPConnector,
        language_decoder: LanguageDecoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        compile: bool = False,  # noqa: A002
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["vision_encoder", "connector", "language_decoder"],
        )

        self.vision_encoder = vision_encoder
        self.connector = connector
        self.language_decoder = language_decoder

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_tokens = self.vision_encoder(pixel_values)  # [B, 49, 768]  no grad
        visual_embeds = self.connector(patch_tokens)  # [B, N_vis, 576]  grad flows here
        loss, logits = self.language_decoder(visual_embeds, input_ids, attention_mask, labels)
        return loss, logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _ = self(**batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ppl", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # A single NaN/Inf gradient corrupts every connector param via AdamW
        # (m and v become NaN forever). Detect and zero them out so the step
        # is effectively skipped instead of nuking the run.
        bad = False
        for p in self.connector.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad = True
                break
        if bad:
            self.log("train/nan_grad", 1.0, on_step=True, prog_bar=False)
            for p in self.connector.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, _ = self(**batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, _ = self(**batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,  # [B, 3, 224, 224]
        prompt_ids: torch.Tensor | None = None,  # [P] or [B, P] — text prefix tokens
        max_new_tokens: int = 64,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate caption token IDs for a batch of images.

        The LM sees:  [visual_embeds] + [prompt_embeds] → generates the rest.

        If prompt_ids is None we fall back to a single BOS token (suitable for
        base LMs). For instruction-tuned LMs, pass the tokenized user prompt
        + assistant header (i.e. the same prefix the dataset masks out of
        loss), so the LM sees the in-distribution chat format.

        Returns raw token IDs [B, max_new_tokens]; caller decodes them.
        """
        was_training = self.training
        self.eval()
        try:
            lm = self.language_decoder.model
            device = self.device

            patch_tokens = self.vision_encoder(pixel_values)
            visual_embeds = self.connector(patch_tokens)
            visual_embeds = visual_embeds.to(lm.dtype)

            B, N_vis, _ = visual_embeds.shape

            if prompt_ids is None:
                prompt_ids = torch.full(
                    (B, 1),
                    lm.config.bos_token_id,
                    dtype=torch.long,
                    device=device,
                )
            else:
                prompt_ids = prompt_ids.to(device)
                if prompt_ids.dim() == 1:
                    prompt_ids = prompt_ids.unsqueeze(0).expand(B, -1)

            prompt_embeds = lm.get_input_embeddings()(prompt_ids)  # [B, P, H]
            inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
            attention_mask = torch.ones(
                B,
                N_vis + prompt_ids.size(1),
                dtype=torch.long,
                device=device,
            )

            generated = lm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=lm.config.eos_token_id,
                **generate_kwargs,
            )
            return generated  # [B, max_new_tokens]
        finally:
            self.train(was_training)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.connector = torch.compile(self.connector)

    def configure_optimizers(self) -> dict[str, Any]:
        # Only the connector parameters are passed — encoder and decoder are frozen
        optimizer = self.hparams.optimizer(params=self.connector.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
