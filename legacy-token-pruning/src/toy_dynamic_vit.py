import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class ToyDynamicViTConfig:
    image_size: int = 32
    patch_size: int = 4
    num_classes: int = 10
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    prune_layers: Tuple[int, ...] = ()
    keep_ratios: Tuple[float, ...] = ()
    scorer_hidden_dim: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToyDynamicViTConfig":
        payload = dict(payload)
        payload["prune_layers"] = tuple(payload.get("prune_layers", ()))
        payload["keep_ratios"] = tuple(payload.get("keep_ratios", ()))
        return cls(**payload)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenPruner(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.scorer(patch_tokens).squeeze(-1)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        keep_ratio: float,
        hard_prune: bool,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_tokens, _ = patch_tokens.shape
        scores = self.score_tokens(patch_tokens)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(scores)
        else:
            candidate_mask = candidate_mask.to(dtype=scores.dtype)

        available_tokens = int(candidate_mask[0].sum().item())
        keep_tokens = max(1, int(math.ceil(available_tokens * keep_ratio)))

        masked_scores = scores.masked_fill(candidate_mask <= 0.0, float("-inf"))
        keep_indices = torch.topk(masked_scores, k=keep_tokens, dim=1).indices

        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, keep_indices, 1.0)
        hard_mask = hard_mask * candidate_mask

        if hard_prune:
            kept = torch.gather(
                patch_tokens,
                dim=1,
                index=keep_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1)),
            )
            return {
                "tokens": kept,
                "scores": scores,
                "keep_indices": keep_indices,
                "mask": hard_mask,
            }

        soft_mask = torch.sigmoid(scores) * candidate_mask
        mask_ste = hard_mask - soft_mask.detach() + soft_mask
        weighted = patch_tokens * mask_ste.unsqueeze(-1)
        return {
            "tokens": weighted,
            "scores": scores,
            "keep_indices": keep_indices,
            "mask": mask_ste,
            "hard_mask": hard_mask,
        }


class ToyDynamicViT(nn.Module):
    """Minimal class-project ViT with optional DynamicViT-style token pruning.

    This is intentionally small and isolated from the robotics code.
    It is a course-safe approximation of the token-pruning idea, not a full
    reproduction of the original large-scale paper setup.
    """

    def __init__(self, config: ToyDynamicViTConfig):
        super().__init__()
        if len(config.prune_layers) != len(config.keep_ratios):
            raise ValueError("prune_layers and keep_ratios must have the same length")
        if any(layer < 0 or layer >= config.depth for layer in config.prune_layers):
            raise ValueError("prune layer index out of range")
        if any(ratio <= 0.0 or ratio > 1.0 for ratio in config.keep_ratios):
            raise ValueError("keep ratios must be in (0, 1]")

        self.config = config
        self.patch_embed = PatchEmbed(config.image_size, config.patch_size, config.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        self.prune_layers = tuple(config.prune_layers)
        self.keep_ratios = tuple(config.keep_ratios)
        self.prune_map = {layer: idx for idx, layer in enumerate(self.prune_layers)}
        self.pruners = nn.ModuleList(
            [TokenPruner(config.embed_dim, config.scorer_hidden_dim) for _ in self.prune_layers]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_keep_ratios(self, keep_ratios: Sequence[float]) -> None:
        keep_ratios = tuple(float(ratio) for ratio in keep_ratios)
        if len(keep_ratios) != len(self.prune_layers):
            raise ValueError("keep_ratios must match the configured number of prune stages")
        if any(ratio <= 0.0 or ratio > 1.0 for ratio in keep_ratios):
            raise ValueError("keep ratios must be in (0, 1]")
        self.keep_ratios = keep_ratios

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
        prune_for_inference: bool = True,
    ):
        batch_size = x.size(0)
        patch_tokens = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.pos_drop(tokens)

        original_patch_ids = (
            torch.arange(self.num_patches, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        active_patch_ids = original_patch_ids
        active_hard_mask = torch.ones(
            batch_size,
            self.num_patches,
            device=x.device,
            dtype=patch_tokens.dtype,
        )
        key_padding_mask = None

        pruning_info: List[Dict[str, Any]] = []
        hard_prune = prune_for_inference and not self.training

        for layer_idx, block in enumerate(self.blocks):
            tokens = block(tokens, key_padding_mask=key_padding_mask)
            if key_padding_mask is not None and not hard_prune:
                tokens = torch.cat(
                    [tokens[:, :1], tokens[:, 1:] * active_hard_mask.unsqueeze(-1)],
                    dim=1,
                )

            if layer_idx not in self.prune_map:
                continue

            pruner_idx = self.prune_map[layer_idx]
            cls_tok = tokens[:, :1]
            patch_tok = tokens[:, 1:]

            pruned = self.pruners[pruner_idx](
                patch_tokens=patch_tok,
                keep_ratio=self.keep_ratios[pruner_idx],
                hard_prune=hard_prune,
                candidate_mask=None if hard_prune else active_hard_mask,
            )

            if hard_prune:
                selected_patch_ids = torch.gather(active_patch_ids, 1, pruned["keep_indices"])
                active_patch_ids = selected_patch_ids
                key_padding_mask = None
            else:
                active_hard_mask = active_hard_mask * pruned["hard_mask"]
                selected_patch_ids = torch.gather(original_patch_ids, 1, pruned["keep_indices"])
                key_padding_mask = torch.cat(
                    [
                        torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool),
                        active_hard_mask <= 0.0,
                    ],
                    dim=1,
                )

            tokens = torch.cat([cls_tok, pruned["tokens"]], dim=1)
            pruning_info.append(
                {
                    "layer": layer_idx,
                    "keep_ratio": self.keep_ratios[pruner_idx],
                    "kept_token_count": int(pruned["keep_indices"].size(1)),
                    "selected_patch_indices": selected_patch_ids.detach().cpu(),
                    "scores": pruned["scores"].detach().cpu(),
                }
            )

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])

        if not return_info:
            return logits

        if hard_prune:
            final_patch_ids = active_patch_ids.detach().cpu()
        else:
            final_patch_ids = torch.stack(
                [torch.nonzero(mask > 0.0, as_tuple=False).squeeze(-1) for mask in active_hard_mask],
                dim=0,
            ).detach().cpu()
        return {
            "logits": logits,
            "pruning": pruning_info,
            "final_patch_indices": final_patch_ids,
            "num_patches": self.num_patches,
            "grid_size": self.grid_size,
            "hard_pruned": hard_prune,
        }


def build_model(
    model_type: str,
    num_classes: int = 10,
    image_size: int = 32,
    patch_size: int = 4,
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    prune_layers: Sequence[int] = (1, 3),
    keep_ratios: Sequence[float] = (0.5, 0.5),
    scorer_hidden_dim: int = 128,
) -> ToyDynamicViT:
    if model_type not in {"dense", "pruned"}:
        raise ValueError("model_type must be 'dense' or 'pruned'")

    config = ToyDynamicViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        prune_layers=tuple(prune_layers) if model_type == "pruned" else (),
        keep_ratios=tuple(keep_ratios) if model_type == "pruned" else (),
        scorer_hidden_dim=scorer_hidden_dim,
    )
    return ToyDynamicViT(config)
