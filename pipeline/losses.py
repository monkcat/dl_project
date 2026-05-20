"""Three losses for graph-induced relevance learning.

L_GRCL: Graph-Relevance Contrastive Loss (main, listwise CE with normalized weights)
L_cov:  Coverage Boundary Loss (set boundary penalty; enforces evidence in top-K)
L_cons: PE-Dropout Consistency Loss (asymmetric KL distillation: GPE-on → GPE-off)

Total: L = L_GRCL + λ_cov · L_cov + λ_cons · L_cons

Signatures use 2-D tensors (B=batch, N=candidate pool size) but degrade
gracefully when given 1-D inputs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def grcl_loss(
    scores: torch.Tensor,        # (N,) or (B, N) — model scores s(q, e)
    relevances: torch.Tensor,    # (N,) or (B, N) — graph relevance g(q, e), in [0, ∞)
    tau: float = 0.07,
) -> torch.Tensor:
    """Graph-Relevance Contrastive Loss.

    Weighted multi-positive InfoNCE:
        w̃(q,e) = max(0, g(q,e)) / Σ_x max(0, g(q,x))
        L_GRCL = -Σ_e w̃(q,e) · log P_model(e | q)

    where P_model(e | q) = softmax(s(q,·) / τ).

    Args:
        scores:     model retrieval scores.
        relevances: graph relevance (0 for negatives).
        tau:        retrieval temperature (default 0.07, SigLIP-style).

    Returns:
        scalar loss (mean over batch). When an anchor has no positive relevance
        (Σ w == 0), it contributes 0 to the mean.
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevances = relevances.unsqueeze(0)
    if scores.shape != relevances.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs relevances {relevances.shape}")

    log_p = F.log_softmax(scores / tau, dim=-1)            # (B, N)
    w = relevances.clamp(min=0.0)
    w_sum = w.sum(dim=-1, keepdim=True)                    # (B, 1)
    has_positive = (w_sum > 1e-9).squeeze(-1)              # (B,)
    w_safe = w / w_sum.clamp(min=1e-9)
    per_anchor = -(w_safe * log_p).sum(dim=-1)             # (B,)
    if has_positive.any():
        return per_anchor[has_positive].mean()
    return scores.new_zeros(())


def coverage_loss(
    scores: torch.Tensor,        # (N,) or (B, N)
    relevances: torch.Tensor,    # (N,) or (B, N)
    K: int = 10,
    eta: float = 5.0,
    strong_threshold: float = 0.5,
    neg_threshold: float = 0.1,
) -> torch.Tensor:
    """Coverage Boundary Loss.

    For strong positives  S+ = {e : g(q,e) >= strong_threshold}  and
        θ_K = K-th highest score over negatives (g < neg_threshold):

        L_cov = (1/|S+|) Σ_p softplus(η · (θ_K - s(q,p)))

    Zero when all strong positives outscore the K-th negative. Zero also when
    there are fewer than K negatives or zero strong positives.
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevances = relevances.unsqueeze(0)

    B, _ = scores.shape
    losses: list[torch.Tensor] = []
    for b in range(B):
        s_b = scores[b]
        r_b = relevances[b]
        strong_mask = r_b >= strong_threshold
        neg_mask = r_b < neg_threshold
        if int(strong_mask.sum()) == 0 or int(neg_mask.sum()) < K:
            continue
        neg_scores = s_b[neg_mask]
        # K-th highest negative score (= K-th order statistic from top)
        topk_vals = neg_scores.topk(K, sorted=False).values
        theta_k = topk_vals.min()
        pos_scores = s_b[strong_mask]
        margin = eta * (theta_k - pos_scores)
        losses.append(F.softplus(margin).mean())

    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def consistency_loss(
    scores_full: torch.Tensor,   # (N,) or (B, N) — anchor encoded with GPE
    scores_drop: torch.Tensor,   # (N,) or (B, N) — anchor encoded without GPE
    tau: float = 0.07,
) -> torch.Tensor:
    """PE-Dropout Consistency Loss (asymmetric distillation).

    Teacher  p_full(e|q) = softmax(s(z_q^full, z_e) / τ)  (stopgrad)
    Student  p_drop(e|q) = softmax(s(z_q^drop, z_e) / τ)
    Loss     KL(p_full || p_drop)

    Applied only on batches where the anchor has GPE attributes (element
    anchor). On NL-query batches, callers should skip this loss.
    """
    if scores_full.dim() == 1:
        scores_full = scores_full.unsqueeze(0)
        scores_drop = scores_drop.unsqueeze(0)
    if scores_full.shape != scores_drop.shape:
        raise ValueError(
            f"shape mismatch: scores_full {scores_full.shape} vs scores_drop {scores_drop.shape}"
        )

    p_full = F.softmax(scores_full / tau, dim=-1).detach()
    log_p_drop = F.log_softmax(scores_drop / tau, dim=-1)
    # F.kl_div(log_input, target) computes  Σ target * (log target - log_input)
    # = KL(target || input) = KL(p_full || p_drop).  Reduction 'batchmean' divides by B.
    return F.kl_div(log_p_drop, p_full, reduction="batchmean")


def total_loss(
    grcl_value: torch.Tensor,
    cov_value: torch.Tensor,
    cons_value: torch.Tensor,
    lambda_cov: float = 0.3,
    lambda_cons: float = 0.5,
) -> torch.Tensor:
    """Combine the three loss terms with default weights."""
    return grcl_value + lambda_cov * cov_value + lambda_cons * cons_value


# ────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 4, 16

    # Build a synthetic batch:
    #   - Each anchor has 1-2 strong positives (g=1.0), a few weak (g=0.3),
    #     and many zeros.
    relevances = torch.zeros(B, N)
    for b in range(B):
        # 1 strong positive
        relevances[b, b] = 1.0
        # weak positives at fixed offsets
        relevances[b, (b + 1) % N] = 0.6
        relevances[b, (b + 2) % N] = 0.3

    print("=== relevance matrix ===")
    print(relevances)

    # Scenario 1: scores ALIGNED with relevance — losses should be small.
    scores_aligned = relevances * 8.0 + torch.randn(B, N) * 0.1
    L1 = grcl_loss(scores_aligned, relevances)
    L2 = coverage_loss(scores_aligned, relevances, K=4)
    print(f"\nAligned scores: GRCL = {L1.item():.4f}  Coverage = {L2.item():.4f}  (both small)")

    # Scenario 2: scores RANDOM — GRCL should be larger.
    scores_random = torch.randn(B, N)
    L1r = grcl_loss(scores_random, relevances)
    L2r = coverage_loss(scores_random, relevances, K=4)
    print(f"Random scores:  GRCL = {L1r.item():.4f}  Coverage = {L2r.item():.4f}  (both larger)")

    # Scenario 3: scores ANTI-ALIGNED — GRCL maximum.
    scores_anti = -relevances * 8.0 + torch.randn(B, N) * 0.1
    L1a = grcl_loss(scores_anti, relevances)
    L2a = coverage_loss(scores_anti, relevances, K=4)
    print(f"Anti-aligned:   GRCL = {L1a.item():.4f}  Coverage = {L2a.item():.4f}  (highest)")

    # Consistency loss
    Lc = consistency_loss(scores_aligned, scores_aligned)
    print(f"\nConsistency (same dist):  {Lc.item():.4f}  (~0)")
    Lc2 = consistency_loss(scores_aligned, scores_random)
    print(f"Consistency (different):  {Lc2.item():.4f}  (positive)")

    # Total loss
    L_tot = total_loss(L1, L2, Lc, lambda_cov=0.3, lambda_cons=0.5)
    print(f"\nTotal loss (aligned scenario): {L_tot.item():.4f}")

    # Sanity: gradient flows
    scores_aligned.requires_grad_(True)
    L = total_loss(
        grcl_loss(scores_aligned, relevances),
        coverage_loss(scores_aligned, relevances, K=4),
        consistency_loss(scores_aligned, scores_aligned.detach()),
    )
    L.backward()
    print(f"gradient norm on scores: {scores_aligned.grad.norm():.4f}")
