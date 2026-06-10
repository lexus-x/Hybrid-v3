import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AToMBlock(nn.Module):
    """
    Attention-Guided Token Absorption (AToM) Block.
    Performs standard self-attention, then uses the attention matrix to
    absorb the lowest-importance tokens into the tokens they attend to the most.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        keep_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        x: [B, N, D]
        returns:
            x: [B, N_new, D]
            info: Dict containing absorption details
        """
        B, N, D = x.shape
        attn_in = self.norm1(x)

        # 1. Forward self-attention, extracting weights
        attn_out, attn_weights = self.attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=True,
            average_attn_weights=True, # [B, N, N]
        )
        x = x + attn_out

        info = None
        if keep_ratio < 1.0 and N > 2:
            # We don't absorb the CLS token (index 0)
            # attn_weights: [B, Target(Query), Source(Key)]
            # Importance of a token is how much others attend to it
            # Sum over all queries (dim 1) to get importance of keys (dim 2)
            importance = attn_weights[:, :, 1:].sum(dim=1) # [B, N-1]

            num_keep = max(1, int(math.ceil((N - 1) * keep_ratio)))
            num_absorb = (N - 1) - num_keep

            if num_absorb > 0:
                # 2. Identify low-importance tokens
                sorted_idx = torch.argsort(importance, dim=1) # Ascending order
                absorb_idx = sorted_idx[:, :num_absorb] + 1 # +1 to offset CLS token [B, A]
                keep_idx = sorted_idx[:, num_absorb:] + 1 # [B, K]

                # 3. For each token to absorb, find which kept token it attends to the most
                # We only look at attention from absorb_idx (queries) to keep_idx (keys)
                
                # Gather rows for absorbed tokens [B, A, N]
                absorb_queries_attn = torch.gather(
                    attn_weights, 
                    1, 
                    absorb_idx.unsqueeze(-1).expand(-1, -1, N)
                )
                
                # Extract only the columns corresponding to keep_idx [B, A, K]
                absorb_to_keep_attn = torch.gather(
                    absorb_queries_attn,
                    2,
                    keep_idx.unsqueeze(1).expand(-1, num_absorb, -1)
                )
                
                # Find the argmax in the K dimension
                best_keep_target_local_idx = torch.argmax(absorb_to_keep_attn, dim=2) # [B, A]
                
                # Map back to global indices in the original sequence
                best_keep_target_global_idx = torch.gather(keep_idx, 1, best_keep_target_local_idx) # [B, A]
                
                # 4. Perform the absorption
                cls_token = x[:, 0:1, :]
                patch_tokens = x[:, 1:, :] # [B, N-1, D]
                
                # We want to route absorb_idx into best_keep_target_global_idx
                # and just keep keep_idx.
                # A clean way: 
                # Create a new tensor of size [B, K, D]
                # Initialize with the kept tokens
                kept_tokens = torch.gather(
                    x, 
                    1, 
                    keep_idx.unsqueeze(-1).expand(-1, -1, D)
                )
                
                # Gather the tokens to be absorbed
                tokens_to_absorb = torch.gather(
                    x,
                    1,
                    absorb_idx.unsqueeze(-1).expand(-1, -1, D)
                )
                
                # Scatter add the absorbed tokens into the kept tokens
                kept_tokens.scatter_add_(
                    1,
                    best_keep_target_local_idx.unsqueeze(-1).expand(-1, -1, D),
                    tokens_to_absorb
                )
                
                # We should theoretically track mass and average them, but scatter_add_ serves 
                # as a simple linear combination which works well for un-normalized features.
                
                x = torch.cat([cls_token, kept_tokens], dim=1)
                
                info = {
                    "keep_idx": keep_idx.detach().cpu(),
                    "absorb_idx": absorb_idx.detach().cpu(),
                    "target_idx": best_keep_target_global_idx.detach().cpu(),
                }

        # 5. Standard MLP
        x = x + self.mlp(self.norm2(x))
        return x, info


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AToMViT(nn.Module):
    """
    Vision Transformer equipped with Attention-Guided Token Absorption (AToM).
    Designed natively for 32x32 CIFAR-10 images to prove true efficiency.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4, # 4x4 patch -> 8x8 grid -> 64 patches (perfect for CIFAR-10)
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        prune_layers: Sequence[int] = (),
        keep_ratios: Sequence[float] = (),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [AToMBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.prune_layers = tuple(prune_layers)
        self.keep_ratios = tuple(keep_ratios)
        self.prune_map = {layer: ratio for layer, ratio in zip(self.prune_layers, self.keep_ratios)}

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def set_keep_ratios(self, keep_ratios: Sequence[float]) -> None:
        self.keep_ratios = tuple(float(r) for r in keep_ratios)
        self.prune_map = {layer: ratio for layer, ratio in zip(self.prune_layers, self.keep_ratios)}

    def forward(self, x: torch.Tensor, return_info: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        absorption_info = []

        for i, block in enumerate(self.blocks):
            keep_ratio = self.prune_map.get(i, 1.0)
            x, info = block(x, keep_ratio=keep_ratio)
            if info is not None:
                info["layer"] = i
                absorption_info.append(info)

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)

        if return_info:
            return {"logits": logits, "pruning": absorption_info}
        return logits


def build_atom_model(model_type: str = "dense", prune_layers=(1, 3, 5), keep_ratios=(0.75, 0.5, 0.25)):
    if model_type == "dense":
        return AToMViT(prune_layers=(), keep_ratios=())
    else:
        return AToMViT(prune_layers=prune_layers, keep_ratios=keep_ratios)
