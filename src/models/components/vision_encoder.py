import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class VisionEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 vision encoder.

    Two output modes:
        use_cls=False (LLaVA default): returns all 49 patch tokens, CLS dropped.
            The MLP connector maps each spatial token independently. CLS is
            excluded so the connector must work with spatial features.
        use_cls=True: returns only the CLS token [B, 1, H]. Useful for
            single-token settings (e.g. classification-style tasks) where CLIP's
            already-trained global summary is a stronger signal than averaging
            untrained-for-pooling patches.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        use_cls: bool = False,
    ) -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.use_cls = use_cls
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_patches(self) -> int:
        cfg = self.model.config
        return (cfg.image_size // cfg.patch_size) ** 2

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 224, 224] preprocessed by CLIPImageProcessor
        Returns:
            patch_tokens: [B, num_patches, hidden_size]  (CLS token excluded)
        """
        # Always run CLIP in float32 — low-res images (e.g. CIFAR-10 upscaled)
        # produce near-uniform patches that cause NaN in bf16 attention softmax.
        outputs = self.model(pixel_values=pixel_values.float())
        if self.use_cls:
            return outputs.last_hidden_state[:, 0:1, :]  # [B, 1, H] — CLS only
        # CLIP ViT-B/32: 224/32 = 7×7 = 49 patch tokens + 1 CLS → drop CLS
        return outputs.last_hidden_state[:, 1:, :]
