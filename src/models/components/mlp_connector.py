import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPConnector(nn.Module):
    """Two-layer MLP that projects vision patch tokens into the LLM's embedding space.

    This is the ONLY component trained in Stage 1 (~1.2 M parameters).
    Architecture matches LLaVA-1.5 mlp2x_gelu: Linear(vision→llm) → GELU → Linear(llm→llm).

    Optional spatial pooling (pool_size) reduces the token count before the MLP.
    E.g. pool_size=3 collapses CLIP ViT-B/32's 49 (7×7) patch tokens down to 9
    (3×3), which makes the connector easier to train with limited data.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 576,
        pool_size: int | None = None,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        # LLaVA-1.5 mlp2x_gelu + prenorm for cross-modal distribution alignment
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        # Small output-layer init keeps visual prefix near zero at the start of
        # training, preventing extreme inputs to the frozen LLM that cause NaN loss.
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, vision_dim]  — patch tokens from the vision encoder
                                     N is assumed to be a perfect square (e.g. 49 = 7×7)
        Returns:
            [B, N', llm_dim]       — N' = pool_size² if pool_size is set, else N
        """
        if self.pool_size is not None:
            B, N, C = x.shape
            H = W = math.isqrt(N)
            # Use avg_pool2d (not adaptive) — MPS requires input divisible by output
            # for adaptive pooling; regular pool2d works for any size.
            k = H // self.pool_size
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = F.avg_pool2d(x, kernel_size=k, stride=k)
            x = x.flatten(2).permute(0, 2, 1)
        return self.net(x)
