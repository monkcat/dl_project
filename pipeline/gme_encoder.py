"""GME-Qwen2-VL adapter for the eval_full encoder interface.

GME-Qwen2-VL-2B is a decoder MLLM that emits ONE pooled, L2-normalized
embedding per text or image (via the model's `get_text_embeddings` /
`get_image_embeddings`, defined in the repo's custom `modeling_gme_qwen2vl.py`,
loaded with `trust_remote_code=True`). It is NOT a SigLIP/CLIP-style dual tower,
so `pipeline.element_encoder.ElementTokenEncoder` (which assumes
`config.text_config` / multi-token late interaction) cannot wrap it.

This adapter exposes exactly the methods `pipeline.eval_full` calls on an
encoder, returning each element/query as a single (K=1) normalized token. With
K=1 normalized vectors, the late-interaction MaxSim used by eval_full reduces to
plain cosine similarity — i.e. standard GME zero-shot retrieval, matching the
GME baseline in `eval/analysis/motivation_modality_gap.py`.

The model's custom code pins `transformers<4.52`, so the `gme` row must run
under the dedicated `.venv_gme` interpreter (see scripts/run_full_suite.sh).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GMEEncoder:
    """Drop-in encoder for eval_full backed by GME-Qwen2-VL embeddings.

    Implements: tokenize_text, forward_text, preprocess_images, forward_vision,
    _stack_gpe_features, proj_dim, eval(), to(), load_state_dict().
    GPE is ignored (gme is a zero-shot reference; no graph PE).
    """

    def __init__(
        self,
        hf_id: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        text_batch: int = 8,
        image_batch: int = 4,
        **_ignored,
    ):
        from transformers import AutoModel

        self.device = device
        self.text_batch = text_batch
        self.image_batch = image_batch
        dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
        print(f"[GMEEncoder] loading {hf_id} (dtype={dtype}, trust_remote_code)")
        self.model = (
            AutoModel.from_pretrained(hf_id, trust_remote_code=True, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        # GME hidden size (1536). eval_full uses proj_dim only for the empty-doc
        # zero fallback; query and element embeddings share this dim.
        self.proj_dim = int(getattr(self.model.config, "hidden_size", 1536))
        self._text_stash: list[str] | None = None
        self._img_stash: list | None = None

    # ── lifecycle no-ops (gme is frozen, zero-shot) ────────────────────────
    def eval(self):
        self.model.eval()
        return self

    def to(self, *_a, **_k):
        return self

    def load_state_dict(self, *_a, **_k):
        # gme has no trained checkpoint; eval_full never passes --ckpt for it,
        # but keep this a no-op so the interface is total.
        return None

    def _stack_gpe_features(self, _feats):
        return None  # GME ignores graph PE

    # ── text path ──────────────────────────────────────────────────────────
    def tokenize_text(self, texts):
        # eval_full does tok["input_ids"].to(device); stash the real strings and
        # return placeholder tensors that satisfy that call. tokenize_text is
        # always immediately followed by forward_text on the same batch.
        self._text_stash = list(texts)
        n = len(self._text_stash)
        return {
            "input_ids": torch.zeros(n, 1, dtype=torch.long),
            "attention_mask": torch.ones(n, 1, dtype=torch.long),
        }

    @torch.no_grad()
    def forward_text(self, input_ids, attention_mask, gpe_batch=None):
        texts = self._text_stash or []
        embs = []
        for i in range(0, len(texts), self.text_batch):
            chunk = texts[i : i + self.text_batch]
            e = self.model.get_text_embeddings(texts=chunk, batch_size=len(chunk))
            embs.append(F.normalize(e.float(), dim=-1))
        z = torch.cat(embs, dim=0) if embs else torch.zeros(0, self.proj_dim)
        z = z.to(self.device).unsqueeze(1)  # [B, 1, D]
        m = torch.ones(z.shape[0], 1, dtype=torch.long, device=z.device)
        return z, m

    # ── vision path ──────────────────────────────────────────────────────────
    def preprocess_images(self, images):
        # Stash PIL images; forward_vision consumes them. Return a placeholder
        # tensor so eval_full's `.to(device)` call works.
        self._img_stash = [
            (img.convert("RGB") if (img is not None and img.mode != "RGB") else img)
            for img in images
        ]
        return torch.zeros(len(self._img_stash), 1)

    @torch.no_grad()
    def forward_vision(self, pixel_values, gpe_batch=None):
        imgs = self._img_stash or []
        embs = []
        for i in range(0, len(imgs), self.image_batch):
            chunk = imgs[i : i + self.image_batch]
            e = self.model.get_image_embeddings(images=chunk, batch_size=len(chunk))
            embs.append(F.normalize(e.float(), dim=-1))
        z = torch.cat(embs, dim=0) if embs else torch.zeros(0, self.proj_dim)
        z = z.to(self.device).unsqueeze(1)  # [B, 1, D]
        m = torch.ones(z.shape[0], 1, dtype=torch.long, device=z.device)
        return z, m
