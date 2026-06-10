import math
import types
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def patched_attn_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn += attn_mask
    attn = attn.softmax(dim=-1)
    
    # Save the averaged attention weights for AToM!
    # shape [B, num_heads, N, N] -> [B, N, N]
    self.saved_attn_weights = attn.mean(dim=1)
    
    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class AsymmetricTokenPruning(nn.Module):
    """
    Asymmetric Token Pruning (ATP) for Q1 Novelty.
    Slices Q to Top-K tokens, keeps K, V at full N tokens to natively read the entire background without routing overhead.
    """
    def __init__(self, embed_dim: int, num_heads: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Lightweight projection to perform the asymmetric read
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor, # [B, N, D]
        attn_weights: torch.Tensor, # [B, N+1, N+1] where 0 is CLS
        keep_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        
        B, N, D = patch_tokens.shape
        
        # Ignore CLS token (index 0) to compute importance
        patch_attn = attn_weights[:, 1:, 1:] # [B, N, N]
        importance = patch_attn.sum(dim=1) # [B, N]
        
        num_keep = max(1, int(math.ceil(N * keep_ratio)))
        
        if num_keep >= N:
            return {
                "tokens": patch_tokens,
                "keep_indices": torch.arange(N, device=patch_tokens.device).unsqueeze(0).expand(B, -1)
            }
            
        # Top-K active tokens
        keep_idx = torch.topk(importance, k=num_keep, dim=1).indices # [B, K]
        
        # Q is only Top-K
        q_tokens = torch.gather(patch_tokens, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D)) # [B, K, D]
        
        # Asymmetric Attention (Q is K tokens, K & V are N tokens)
        q = self.q(q_tokens).reshape(B, num_keep, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, K, d]
        
        kv = self.kv(patch_tokens).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # [B, H, N, d]
        
        # Native Attention (K x N) -> massive FLOP savings over NxN, zero routing overhead!
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, H, K, N]
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, num_keep, D) # [B, K, D]
        x = self.proj(x)
        
        # Residual combination to preserve foreground features while absorbing background
        out_tokens = q_tokens + x
        
        return {
            "tokens": out_tokens,
            "keep_indices": keep_idx,
        }


class SOTAHybridViT(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_large_patch16_384",
        num_classes: int = 10,
        pretrained: bool = True,
        prune_layers: Sequence[int] = (),
        keep_ratios: Sequence[float] = (),
        prune_mode: str = "lite",
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.embed_dim = self.backbone.embed_dim
        
        self.prune_layers = tuple(prune_layers)
        self.keep_ratios = tuple(keep_ratios)
        self.prune_map = {layer: idx for idx, layer in enumerate(self.prune_layers)}
        
        self.pruners_raw = nn.ModuleList(
            [AsymmetricTokenPruning(embed_dim=self.embed_dim, num_heads=self.backbone.blocks[0].attn.num_heads) for _ in self.prune_layers]
        )
        self.pruners = nn.ModuleList(
            [torch.compile(p) for p in self.pruners_raw]
        )
        
        # Monkey patch the attention modules at the prune layers to disable flash attention 
        # and save the attention matrix
        for i, block in enumerate(self.backbone.blocks):
            if i in self.prune_map:
                block.attn.fused_attn = False
                block.attn.forward = types.MethodType(patched_attn_forward, block.attn)
        
    def set_keep_ratios(self, keep_ratios: Sequence[float]) -> None:
        self.keep_ratios = tuple(float(r) for r in keep_ratios)

    def forward_features(self, x, return_info=False):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        
        cls_token = x[:, :1]
        patch_tokens = x[:, 1:]
        
        num_patches = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_patches))
        
        original_patch_ids = (
            torch.arange(num_patches, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )
        active_patch_ids = original_patch_ids
        
        pruning_info = []
        
        for i, block in enumerate(self.backbone.blocks):
            x = torch.cat([cls_token, patch_tokens], dim=1)
            x = block(x)
            cls_token = x[:, :1]
            patch_tokens = x[:, 1:]
            
            if i in self.prune_map:
                pruner_idx = self.prune_map[i]
                
                # Fetch the saved attention matrix from the monkey-patched block
                saved_attn_weights = block.attn.saved_attn_weights
                
                # Free memory early
                block.attn.saved_attn_weights = None
                
                if torch.jit.is_tracing():
                    pruner = self.pruners_raw[pruner_idx]
                else:
                    pruner = self.pruners[pruner_idx]
                    
                pruned = pruner(
                    patch_tokens=patch_tokens,
                    attn_weights=saved_attn_weights,
                    keep_ratio=self.keep_ratios[pruner_idx],
                )
                
                patch_tokens = pruned["tokens"]
                selected_patch_ids = torch.gather(active_patch_ids, 1, pruned["keep_indices"])
                active_patch_ids = selected_patch_ids

                pruning_info.append(
                    {
                        "layer": i,
                        "keep_ratio": self.keep_ratios[pruner_idx],
                        "kept_token_count": int(pruned["keep_indices"].size(1)),
                        "selected_patch_indices": selected_patch_ids.detach().cpu(),
                    }
                )

        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.backbone.norm(x)
        
        if return_info:
            return x, pruning_info, active_patch_ids, num_patches, grid_size
        return x

    def forward(self, x, return_info=False, prune_for_inference=False):
        # We ignore prune_for_inference here since AToM does not use STE
        # and natively absorbs tokens in both training and inference
        if return_info:
            x, pruning_info, final_patch_ids, num_patches, grid_size = self.forward_features(x, return_info=True)
            x = self.backbone.forward_head(x)
            return {
                "logits": x,
                "pruning": pruning_info,
                "final_patch_indices": final_patch_ids.detach().cpu(),
                "num_patches": num_patches,
                "grid_size": grid_size,
            }
        else:
            x = self.forward_features(x, return_info=False)
            x = self.backbone.forward_head(x)
            return x

def build_sota_model(model_type="dense", prune_layers=(3, 6, 9), keep_ratios=(0.75, 0.5, 0.25), prune_mode="lite", model_name="vit_small_patch16_224"):
    if model_type == "dense":
        return SOTAHybridViT(model_name=model_name, pretrained=True, prune_layers=(), keep_ratios=(), num_classes=10, prune_mode=prune_mode)
    else:
        return SOTAHybridViT(model_name=model_name, pretrained=True, prune_layers=prune_layers, keep_ratios=keep_ratios, num_classes=10, prune_mode=prune_mode)
