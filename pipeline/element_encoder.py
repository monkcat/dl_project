"""Element Token Encoder — SigLIPv2 + LoRA + Graph Position Embedding.

Produces multi-token element representations T(e) ∈ R^(K_e × D_proj) for late
interaction retrieval. GPE is injected token-broadcast with a β-gated scalar.

Forward flow per element:
    text       → SigLIPv2 text_model  (with LoRA)   → (K_text, D_hidden)
    image      → SigLIPv2 vision_model (with LoRA)  → (K_patches, D_hidden)
                     ↓
                + β · gpe(v).unsqueeze(0)       # token-broadcast, β = sigmoid(raw_β)
                     ↓
                LayerNorm
                     ↓
                Linear(D_hidden → D_proj)
                     ↓
                F.normalize(dim=-1)
                     ↓
                (K, D_proj) L2-normalized

Use:
    encoder = ElementTokenEncoder()
    out = encoder.forward_text(input_ids, attention_mask, gpe_features=...)
        → (B, K, D_proj), (B, K) mask
    out = encoder.forward_vision(pixel_values, gpe_features=...)
        → (B, K_patches, D_proj), None  (no padding for fixed patches)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .graph_pe import GraphPositionEmbedding

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class ElementTokenEncoder(nn.Module):
    """SigLIPv2 + LoRA + GPE token-level encoder for retrieval.

    Outputs (B, K, D_proj) L2-normalized multi-token representations matching
    ColBERT-style late interaction expectations.
    """

    def __init__(
        self,
        hf_id: str = "google/siglip2-base-patch16-224",
        proj_dim: int = 128,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        beta_init_raw: float = -3.0,
        max_text_len: int = 64,
        device: str = "cuda",
        gpe_active_facets: tuple[str, ...] = ("type", "role", "depth", "pos"),
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        print(f"[ElementTokenEncoder] loading {hf_id}")
        self.processor = AutoProcessor.from_pretrained(hf_id)
        base = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32)
        d_hidden = base.config.text_config.hidden_size

        # Apply LoRA to attention projections of both text and vision towers
        if use_lora and PEFT_AVAILABLE:
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                bias="none",
            )
            base = get_peft_model(base, lora_cfg)
            print(f"[ElementTokenEncoder] LoRA r={lora_rank} α={lora_alpha} applied")
            base.print_trainable_parameters()
        elif use_lora and not PEFT_AVAILABLE:
            print("[ElementTokenEncoder] WARNING: peft unavailable, training base in full")

        self.base = base
        self.d_hidden = d_hidden
        self.proj_dim = proj_dim
        self.max_text_len = max_text_len

        # GPE module (operates on D_hidden, before projection)
        self.gpe = GraphPositionEmbedding(d_model=d_hidden, active_facets=gpe_active_facets)
        # β gate for GPE × token combination (sigmoid-bounded, init small)
        self.raw_beta = nn.Parameter(torch.tensor(beta_init_raw))

        # LayerNorm + projection (D_hidden → D_proj). Separate text/vision heads
        # so each modality can specialize during retrieval-FT (cheap; 128k params each).
        self.ln_text = nn.LayerNorm(d_hidden)
        self.ln_vision = nn.LayerNorm(d_hidden)
        self.proj_text = nn.Linear(d_hidden, proj_dim)
        self.proj_vision = nn.Linear(d_hidden, proj_dim)
        nn.init.xavier_uniform_(self.proj_text.weight)
        nn.init.zeros_(self.proj_text.bias)
        nn.init.xavier_uniform_(self.proj_vision.weight)
        nn.init.zeros_(self.proj_vision.bias)

        self.target_device = device
        self.to(device)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)

    # ───────────────────────────────────────────────────────────────────────
    # Tokenization / preprocessing helpers
    # ───────────────────────────────────────────────────────────────────────

    def tokenize_text(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Run the processor's text branch. Returns dict suitable for forward_text.

        SigLIPv2 processor returns only input_ids; we derive attention_mask from
        pad_token_id (= 0 for SigLIPv2 sentencepiece tokenizer).
        """
        out = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = out["input_ids"]
        if "attention_mask" in out:
            attention_mask = out["attention_mask"]
        else:
            pad_id = self.processor.tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def preprocess_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Run the processor's image branch. Returns pixel_values tensor."""
        imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        out = self.processor(images=imgs, return_tensors="pt")
        return out["pixel_values"]

    # ───────────────────────────────────────────────────────────────────────
    # Core forward passes
    # ───────────────────────────────────────────────────────────────────────

    def _maybe_unwrap(self):
        """Get the inner SigLIPv2 model from a possibly PEFT-wrapped base."""
        m = self.base
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        return m

    def _add_gpe(
        self,
        tokens: torch.Tensor,                          # (B, K, D_hidden)
        gpe_features: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Add β · gpe(v) to all tokens (broadcast over K). Returns same shape."""
        if gpe_features is None:
            return tokens
        gpe_vec = self.gpe(
            gpe_features["type_ids"].to(tokens.device),
            gpe_features["role_ids"].to(tokens.device),
            gpe_features["depth"].to(tokens.device),
            gpe_features["intra_pos"].to(tokens.device),
        )  # (B, D_hidden)
        return tokens + self.beta * gpe_vec.unsqueeze(1)

    def forward_text(
        self,
        input_ids: torch.LongTensor,                   # (B, T)
        attention_mask: torch.LongTensor,              # (B, T)
        gpe_features: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts to (B, T, D_proj) tokens + (B, T) padding mask.

        Returns:
            tokens: (B, T, D_proj), L2-normalized
            mask:   (B, T) long, 1=valid, 0=pad
        """
        m = self._maybe_unwrap()
        out = m.text_model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                       # (B, T, D_hidden)
        h = self._add_gpe(h, gpe_features)
        h = self.ln_text(h)
        z = self.proj_text(h)                           # (B, T, D_proj)
        z = F.normalize(z, dim=-1)
        return z, attention_mask

    def forward_vision(
        self,
        pixel_values: torch.Tensor,                    # (B, C, H, W)
        gpe_features: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images to (B, K_patches, D_proj) tokens + ones mask.

        Returns:
            tokens: (B, K, D_proj), L2-normalized
            mask:   (B, K) long, all ones (no padding for fixed patches)
        """
        m = self._maybe_unwrap()
        out = m.vision_model(pixel_values=pixel_values)
        h = out.last_hidden_state                       # (B, K_patches, D_hidden)
        h = self._add_gpe(h, gpe_features)
        h = self.ln_vision(h)
        z = self.proj_vision(h)                         # (B, K, D_proj)
        z = F.normalize(z, dim=-1)
        mask = torch.ones(z.shape[:2], dtype=torch.long, device=z.device)
        return z, mask

    # ───────────────────────────────────────────────────────────────────────
    # Convenience wrappers (handle batching + GPE)
    # ───────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        gpe_features_list: list[dict] | None = None,
        batch_size: int = 64,
    ) -> list[torch.Tensor]:
        """Inference helper. Returns list of (T_valid, D_proj) tensors (CPU).

        gpe_features_list[i] = dict for element i, or None to skip GPE for that element.
        """
        results: list[torch.Tensor] = []
        self.eval()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tok = self.tokenize_text(batch_texts)
            input_ids = tok["input_ids"].to(self.target_device)
            attn_mask = tok["attention_mask"].to(self.target_device)

            gpe_batch = None
            if gpe_features_list is not None:
                # Stack per-element features into batch tensors
                batch_feats = gpe_features_list[i:i + batch_size]
                gpe_batch = self._stack_gpe_features(batch_feats)

            tokens, mask = self.forward_text(input_ids, attn_mask, gpe_batch)
            tokens = tokens.cpu()
            mask = mask.cpu()
            for j in range(tokens.shape[0]):
                k_valid = int(mask[j].sum().item())
                results.append(tokens[j, :k_valid].clone())  # (K_valid, D_proj)
        return results

    @torch.no_grad()
    def encode_images(
        self,
        images: list[Image.Image],
        gpe_features_list: list[dict] | None = None,
        batch_size: int = 16,
    ) -> list[torch.Tensor]:
        """Inference helper. Returns list of (K_patches, D_proj) tensors (CPU)."""
        results: list[torch.Tensor] = []
        self.eval()
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            px = self.preprocess_images(batch_imgs).to(self.target_device)

            gpe_batch = None
            if gpe_features_list is not None:
                batch_feats = gpe_features_list[i:i + batch_size]
                gpe_batch = self._stack_gpe_features(batch_feats)

            tokens, _ = self.forward_vision(px, gpe_batch)
            for j in range(tokens.shape[0]):
                results.append(tokens[j].cpu().clone())
        return results

    def _stack_gpe_features(self, batch_feats: list[dict | None]) -> dict[str, torch.Tensor] | None:
        """Stack per-element feature dicts into batched tensors. None → zero GPE."""
        if all(f is None for f in batch_feats):
            return None
        # Default for None entries: type_id=0, role_id=other (last), depth=1, pos=0
        from .graph_pe import TYPE_TO_ID, ROLE_TO_ID
        default = {
            "type_id": TYPE_TO_ID["text"],
            "role_id": ROLE_TO_ID["other"],
            "depth": 1.0,
            "intra_pos": 0.0,
        }
        type_ids = torch.tensor(
            [f.get("type_id", default["type_id"]) if f else default["type_id"] for f in batch_feats],
            dtype=torch.long,
        )
        role_ids = torch.tensor(
            [f.get("role_id", default["role_id"]) if f else default["role_id"] for f in batch_feats],
            dtype=torch.long,
        )
        depth = torch.tensor(
            [f.get("depth", default["depth"]) if f else default["depth"] for f in batch_feats],
            dtype=torch.float32,
        )
        intra_pos = torch.tensor(
            [f.get("intra_pos", default["intra_pos"]) if f else default["intra_pos"] for f in batch_feats],
            dtype=torch.float32,
        )
        return {
            "type_ids": type_ids,
            "role_ids": role_ids,
            "depth": depth,
            "intra_pos": intra_pos,
        }


# ────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    enc = ElementTokenEncoder(device=device, lora_rank=8)

    # Text smoke test
    texts = [
        "Figure 1 illustrates the proposed architecture for retrieval.",
        "Table 3 reports accuracy across baselines.",
    ]
    gpe_feats = [
        {"type_id": 3, "role_id": 4, "depth": 1.0, "intra_pos": 0.5},  # caption / method
        {"type_id": 3, "role_id": 5, "depth": 1.0, "intra_pos": 0.7},  # caption / result
    ]

    print("\n=== text forward (with GPE) ===")
    tok = enc.tokenize_text(texts)
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)
    gpe_batch = enc._stack_gpe_features(gpe_feats)
    z, mask = enc.forward_text(input_ids, attn_mask, gpe_batch)
    print(f"  text z shape: {z.shape}, mask shape: {mask.shape}")
    print(f"  z norm per token (first 5): {z[0, :5].norm(dim=-1).tolist()}")
    print(f"  β (gate value): {enc.beta.item():.4f}")

    print("\n=== text forward (no GPE, for L_cons drop pass) ===")
    z_nopgpe, _ = enc.forward_text(input_ids, attn_mask, None)
    print(f"  z norm: {z_nopgpe[0, :5].norm(dim=-1).tolist()}")
    delta = (z - z_nopgpe).norm() / z.norm()
    print(f"  relative norm diff (with - without GPE): {delta.item():.4f}")

    # Image smoke test
    print("\n=== vision forward ===")
    try:
        # Find a real figure
        img_root = Path(__file__).resolve().parent.parent / "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px"
        sample_pid = next(p for p in img_root.iterdir() if p.is_dir())
        sample_img = next(sample_pid.glob("*.png"))
        img = Image.open(sample_img).convert("RGB")
        px = enc.preprocess_images([img]).to(device)
        zv, mv = enc.forward_vision(px, None)
        print(f"  vision z shape: {zv.shape}, mask shape: {mv.shape}")
        print(f"  z norm per patch (first 5): {zv[0, :5].norm(dim=-1).tolist()}")
    except Exception as e:
        print(f"  vision test skipped: {e}")

    # Parameter count
    print("\n=== parameter summary ===")
    n_total = sum(p.numel() for p in enc.parameters())
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"  total params:     {n_total:>12,d}")
    print(f"  trainable params: {n_train:>12,d}  ({n_train/n_total*100:.2f}%)")

    # Gradient sanity
    print("\n=== gradient sanity ===")
    z, mask = enc.forward_text(input_ids, attn_mask, gpe_batch)
    loss = z.sum()  # trivial loss
    loss.backward()
    grad_norms = {
        "gpe.raw_alpha_type": enc.gpe.raw_alpha_type.grad.norm().item() if enc.gpe.raw_alpha_type.grad is not None else None,
        "raw_beta": enc.raw_beta.grad.norm().item() if enc.raw_beta.grad is not None else None,
        "proj_text.weight": enc.proj_text.weight.grad.norm().item() if enc.proj_text.weight.grad is not None else None,
    }
    for k, v in grad_norms.items():
        print(f"  {k}: {v}")
