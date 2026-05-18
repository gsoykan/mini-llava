import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class LanguageDecoder(nn.Module):
    """Frozen SmolLM2-135M causal language model.

    Accepts a concatenated sequence of [visual_tokens | text_tokens] as input
    embeddings. The causal LM loss is computed only on the text positions —
    the visual prefix is masked with -100 in the labels so it never contributes
    to the loss, but it does attend to via self-attention and thus conditions
    the text generation.

    No @torch.no_grad() here: gradients must flow backward through this module
    and through visual_embeds so the connector can be updated. The decoder
    parameters themselves are frozen (requires_grad=False) and won't be updated.
    """

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M") -> None:
        super().__init__()
        # torch_dtype=float32: the HF config has torch_dtype=bfloat16, so without
        # this override the LM loads in bf16. bf16 backprop through 30 attention
        # layers overflows and produces NaN gradients on the connector, which
        # AdamW then propagates to all connector params on the next step.
        # attn_implementation="eager" avoids the MPS SDPA kernel (known buggy).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        visual_embeds: torch.Tensor,  # [B, N_vis, H]
        input_ids: torch.Tensor,  # [B, T]   — tokenised caption
        attention_mask: torch.Tensor,  # [B, T]   — 1 real, 0 pad
        labels: torch.Tensor,  # [B, T]   — same as input_ids but -100 for padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N_vis, _ = visual_embeds.shape

        # Cast connector output to the frozen LLM's dtype (e.g. bfloat16 for SmolLM2)
        visual_embeds = visual_embeds.to(self.model.dtype)

        # Embed text tokens and prepend visual tokens
        text_embeds = self.model.get_input_embeddings()(input_ids)  # [B, T, H]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # [B, N_vis+T, H]

        # Extend attention mask: visual prefix is never masked
        vis_mask = torch.ones(B, N_vis, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)  # [B, N_vis+T]

        # Extend labels: visual prefix positions never contribute to the loss
        vis_labels = torch.full((B, N_vis), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([vis_labels, labels], dim=1)  # [B, N_vis+T]

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            use_cache=False,
        )
        return out.loss, out.logits
