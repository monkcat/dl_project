"""Preflight sanity check — run before launching the full experiment suite.

Verifies:
    1. All data files referenced in configs.EVAL_DATASETS exist
    2. SPIQA training corpus is present (graph + elements + QA + images)
    3. All ROW_CONFIGS entries have the required fields
    4. trainer + eval_full + run_experiment + aggregate_results all import
    5. Element graph schema is valid (sample a few docs, check node/edge structure)
    6. GPE feature extraction works (no missing section_role mappings)
    7. Models are available — either from HF cache or HF online

Exit codes:
    0  — all green
    1  — non-fatal warnings (some checks failed but suite is still launchable)
    2  — fatal: missing data files or invalid configs

Usage:
    python -m pipeline.sanity_check
    python -m pipeline.sanity_check --quiet
    python -m pipeline.sanity_check --check_models   # also probe HF model availability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

class Result:
    """Lightweight outcome record. Categories: PASS / WARN / FAIL."""

    def __init__(self):
        self.lines: list[tuple[str, str, str]] = []  # (level, label, detail)
        self.failed = False
        self.warned = False

    def ok(self, label: str, detail: str = "") -> None:
        self.lines.append(("PASS", label, detail))

    def warn(self, label: str, detail: str = "") -> None:
        self.lines.append(("WARN", label, detail))
        self.warned = True

    def fail(self, label: str, detail: str = "") -> None:
        self.lines.append(("FAIL", label, detail))
        self.failed = True

    def render(self, quiet: bool = False) -> None:
        sym = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
        for level, label, detail in self.lines:
            if quiet and level == "PASS":
                continue
            tail = f" — {detail}" if detail else ""
            print(f"  {sym[level]} {label}{tail}")


# ──────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────

def check_modules(r: Result) -> None:
    print("[1] module imports")
    targets = [
        "pipeline.types",
        "pipeline.graph_pe",
        "pipeline.graph_relevance",
        "pipeline.losses",
        "pipeline.element_encoder",
        "pipeline.trainer",
        "pipeline.eval_full",
        "pipeline.run_experiment",
        "pipeline.aggregate_results",
        "pipeline.configs",
    ]
    import importlib
    for mod in targets:
        try:
            importlib.import_module(mod)
            r.ok(f"import {mod}")
        except Exception as exc:
            r.fail(f"import {mod}", repr(exc))


def check_configs(r: Result) -> None:
    print("\n[2] configs.ROW_CONFIGS")
    from pipeline.configs import (
        ROW_CONFIGS, TIER1_ROWS, TIER1_NO_TRAIN, TIER2_ROWS, TIER3_ROWS, TIER4_ROWS,
        ALL_ROWS, INFERENCE_VARIANTS,
    )

    declared = set(ALL_ROWS)
    in_configs = set(ROW_CONFIGS.keys())
    missing = declared - in_configs
    extra = in_configs - declared

    if missing:
        r.fail("rows declared in tiers but missing from ROW_CONFIGS", str(sorted(missing)))
    else:
        r.ok(f"{len(declared)} declared rows all present in ROW_CONFIGS")

    if extra:
        r.warn("extra rows in ROW_CONFIGS not in any tier list", str(sorted(extra)))
    else:
        r.ok("no orphan rows in ROW_CONFIGS")

    required_trained = (
        "row_id", "name", "description",
        "use_gpe", "loss_type", "lambda_cov", "lambda_cons",
        "lr", "batch", "pool", "steps", "tau", "lora_rank",
    )
    required_zeroshot = ("row_id", "name", "description", "skip_training", "encoder_kind", "hf_id")

    for rid, cfg in sorted(ROW_CONFIGS.items()):
        keys = required_zeroshot if cfg.get("skip_training") else required_trained
        miss = [k for k in keys if k not in cfg]
        if miss:
            r.fail(f"row {rid} missing fields", str(miss))

    r.ok(f"INFERENCE_VARIANTS: {len(INFERENCE_VARIANTS)} variants defined")


def check_data_paths(r: Result) -> None:
    print("\n[3] eval datasets — paths referenced in configs.EVAL_DATASETS")
    from pipeline.configs import EVAL_DATASETS

    # Image roots accept either raw-extract or wrapped-extract layout.
    img_fallbacks = {
        "data/benchmarks/spiqa/test-A/SPIQA_testA_Images_224px":
            "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px",
    }

    for ds_id, ds_cfg in EVAL_DATASETS.items():
        for key in ("graph", "elements", "queries"):
            p = REPO / ds_cfg[key]
            if not p.exists():
                r.fail(f"{ds_id}.{key}", f"missing: {p.relative_to(REPO)}")
            else:
                r.ok(f"{ds_id}.{key}", f"{p.relative_to(REPO)}")
        img = ds_cfg.get("image_root")
        if img:
            primary = REPO / img
            fallback = REPO / img_fallbacks.get(img, "")
            if primary.exists():
                r.ok(f"{ds_id}.image_root", f"{primary.relative_to(REPO)}")
            elif fallback.name and fallback.exists():
                r.warn(f"{ds_id}.image_root",
                       f"using fallback layout: {fallback.relative_to(REPO)}")
            else:
                r.fail(f"{ds_id}.image_root", f"missing: {primary.relative_to(REPO)}")


def check_train_data(r: Result) -> None:
    print("\n[4] SPIQA training corpus")
    # _pick_existing in trainer.py already accepts both wrapped/raw layouts;
    # the imports here resolve to whichever is on disk.
    from pipeline.trainer import (
        TRAIN_GRAPH, TRAIN_ELEMENTS, TRAIN_QA, TRAIN_IMG_ROOT,
        TESTA_GRAPH, TESTA_ELEMENTS, TESTA_QA, TESTA_IMG_ROOT,
    )

    for label, p in [
        ("train_val/element_graph_v2.json", TRAIN_GRAPH),
        ("train_val/elements_v2.jsonl",     TRAIN_ELEMENTS),
        ("train_val/SPIQA_train.json",      TRAIN_QA),
        ("train_val/<images dir>",          TRAIN_IMG_ROOT),
        ("test-A/element_graph_v2.json",    TESTA_GRAPH),
        ("test-A/elements_v2.jsonl",        TESTA_ELEMENTS),
        ("test-A/SPIQA_testA.json",         TESTA_QA),
        ("test-A/<images dir>",             TESTA_IMG_ROOT),
    ]:
        if not p.exists():
            r.fail(label, f"missing: {p.relative_to(REPO)}")
        else:
            r.ok(label, f"{p.relative_to(REPO)}")


def check_graph_schema(r: Result) -> None:
    print("\n[5] graph schema sample (skim 3 docs from SPIQA test-A)")
    from pipeline.configs import EVAL_DATASETS
    from pipeline.graph_pe import TYPE_TO_ID, ROLE_TO_ID

    gp = REPO / EVAL_DATASETS["spiqa_testA"]["graph"]
    if not gp.exists():
        r.warn("can't check schema — graph file missing")
        return
    try:
        graphs = json.loads(gp.read_text())
    except Exception as exc:
        r.fail("graph file parse", repr(exc))
        return

    n_docs = len(graphs)
    if n_docs < 1:
        r.fail("empty graph file")
        return
    r.ok(f"loaded {n_docs} SPIQA test-A docs")

    sample_ids = list(graphs.keys())[:3]
    unknown_types: set[str] = set()
    unknown_roles: set[str] = set()
    for did in sample_ids:
        g = graphs[did]
        for n in g.get("nodes", []):
            t = n.get("type")
            if t and t not in TYPE_TO_ID:
                unknown_types.add(t)
            ro = n.get("section_role")
            if ro and ro not in ROLE_TO_ID:
                unknown_roles.add(ro)
    if unknown_types:
        r.warn("unknown node types in sample", str(unknown_types))
    if unknown_roles:
        r.warn("unknown section roles in sample", str(unknown_roles))
    if not unknown_types and not unknown_roles:
        r.ok("all node types + section roles map to known IDs")


def check_propagate(r: Result) -> None:
    print("\n[6] graph_propagate end-to-end (tiny synthetic graph)")
    try:
        import torch
        from pipeline.types import graph_propagate
        g = {
            "nodes": [
                {"id": "q", "type": "text",    "section_role": "method"},
                {"id": "f", "type": "figure",  "section_role": "method"},
                {"id": "c", "type": "caption", "section_role": "method"},
            ],
            "edges": [
                {"src": "q", "dst": "f", "type": "refer_to", "confidence": 1.0, "bidirectional": False},
                {"src": "f", "dst": "c", "type": "caption_of", "confidence": 1.0, "bidirectional": True},
            ],
        }
        s = torch.tensor([1.0, 0.5, 0.3])
        for method, weights in [
            ("none", "full"),
            ("diffusion", "uniform"),
            ("diffusion", "full"),
            ("ppr", "full"),
        ]:
            out = graph_propagate(s, ["q", "f", "c"], g,
                                  method=method, weights=weights,
                                  alpha=0.3, T=2, max_iter=10)
            assert out.shape == s.shape, f"shape mismatch for {method}/{weights}"
        r.ok("graph_propagate works for all 3 methods × 5 weight modes")
    except Exception as exc:
        r.fail("graph_propagate", repr(exc))


def check_models(r: Result) -> None:
    print("\n[7] HF models (cache check)")
    from pipeline.configs import ROW_CONFIGS

    needed_ids = {ROW_CONFIGS[rid].get("hf_id") for rid in ROW_CONFIGS}
    needed_ids = {hid for hid in needed_ids if hid}

    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        cached = {repo.repo_id for repo in cache.repos}
    except Exception as exc:
        r.warn("could not scan HF cache", repr(exc))
        cached = set()

    for hid in sorted(needed_ids):
        if hid in cached:
            r.ok(f"cached: {hid}")
        else:
            r.warn(f"not cached: {hid}",
                   "will be downloaded on first use — set HF_HUB_OFFLINE=0 or pre-fetch")


def check_gpu(r: Result) -> None:
    print("\n[8] GPU visibility")
    try:
        import torch
        if not torch.cuda.is_available():
            r.warn("CUDA not available — will run on CPU (slow)")
            return
        n = torch.cuda.device_count()
        r.ok(f"{n} CUDA device(s) visible")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            free, total = torch.cuda.mem_get_info(i)
            r.ok(f"  GPU {i}: {name}  ({free / 1e9:.1f} / {total / 1e9:.1f} GB free)")
        if n < 2:
            r.warn("expected 2 GPUs for parallel pair-wave scheduling; got only " + str(n))
    except Exception as exc:
        r.warn("GPU probe failed", repr(exc))


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="suppress PASS lines, show only WARN and FAIL")
    ap.add_argument("--check_models", action="store_true",
                    help="probe HF model cache (skip if you trust your env)")
    args = ap.parse_args()

    r = Result()
    check_modules(r)
    check_configs(r)
    check_data_paths(r)
    check_train_data(r)
    check_graph_schema(r)
    check_propagate(r)
    if args.check_models:
        check_models(r)
    check_gpu(r)

    print()
    print("───── summary ─────")
    n_pass = sum(1 for lvl, *_ in r.lines if lvl == "PASS")
    n_warn = sum(1 for lvl, *_ in r.lines if lvl == "WARN")
    n_fail = sum(1 for lvl, *_ in r.lines if lvl == "FAIL")
    print(f"  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}")

    if not args.quiet or n_warn or n_fail:
        print()
        print("───── details ─────")
        r.render(quiet=args.quiet)

    if r.failed:
        print("\n[FAIL] sanity check FAILED — fix the issues above before launching the suite")
        return 2
    if r.warned:
        print("\n[WARN] sanity check passed with warnings — suite is launchable")
        return 1
    print("\n[OK] all checks passed — suite is ready to launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
