"""Sample SPIQA train paper IDs and download the corresponding arXiv PDFs.

Output: data/benchmarks/spiqa/train_subset/pdfs/<paper_id>.pdf

Strategy:
  - Sample N paper IDs from SPIQA_train.json deterministically (seed=0)
  - For each, try arxiv direct URL: https://arxiv.org/pdf/<id>.pdf
  - Fallback: try without version suffix (1611.04684v1 -> 1611.04684)
  - Rate-limit: 1.5s between requests (polite to arxiv)
  - Resume capability: skip if PDF already exists
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
import urllib.error
from pathlib import Path


ROOT = Path("data/benchmarks/spiqa")
TRAIN_JSON = ROOT / "train_val" / "SPIQA_train.json"
OUT_DIR = ROOT / "train_subset" / "pdfs"

ARXIV_BASE = "https://arxiv.org/pdf/"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project)"}


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def download_pdf(arxiv_id: str, out_path: Path, timeout: int = 30) -> bool:
    """Download arXiv PDF. Returns True on success."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True  # already downloaded
    urls = [
        f"{ARXIV_BASE}{arxiv_id}.pdf",                # 1611.04684v1.pdf
        f"{ARXIV_BASE}{strip_version(arxiv_id)}.pdf", # 1611.04684.pdf
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    continue
                data = resp.read()
                if len(data) < 1000:
                    continue
                out_path.write_bytes(data)
                return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        except Exception as e:
            print(f"  unexpected error for {arxiv_id}: {e}")
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_papers", type=int, default=500,
                    help="Sample N paper IDs from SPIQA_train.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.5, help="Seconds between requests")
    ap.add_argument("--max_failures", type=int, default=50,
                    help="Stop if too many consecutive download failures")
    args = ap.parse_args()

    print("Loading SPIQA_train.json...")
    train = json.load(open(TRAIN_JSON))
    all_ids = list(train.keys())
    print(f"  total papers in train: {len(all_ids)}")

    rng = random.Random(args.seed)
    sample_ids = rng.sample(all_ids, args.n_papers)
    print(f"  sampling {len(sample_ids)} paper IDs (seed={args.seed})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    consecutive_fail = 0
    t0 = time.time()
    for i, arxiv_id in enumerate(sample_ids):
        out_path = OUT_DIR / f"{arxiv_id}.pdf"
        if out_path.exists() and out_path.stat().st_size > 1000:
            n_ok += 1
            consecutive_fail = 0
            continue

        ok = download_pdf(arxiv_id, out_path)
        if ok:
            n_ok += 1
            consecutive_fail = 0
        else:
            n_fail += 1
            consecutive_fail += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(sample_ids) - i - 1) / rate
            print(f"  [{i+1}/{len(sample_ids)}] ok={n_ok} fail={n_fail} "
                  f"({rate:.1f}/s, ETA {eta/60:.1f}m)")

        if consecutive_fail >= args.max_failures:
            print(f"  too many consecutive failures ({consecutive_fail}), aborting")
            break
        time.sleep(args.sleep)

    print(f"\nDone. Downloaded {n_ok}/{len(sample_ids)} PDFs to {OUT_DIR}")
    print(f"  failures: {n_fail}")
    sizes = [p.stat().st_size for p in OUT_DIR.glob("*.pdf")]
    if sizes:
        print(f"  avg PDF size: {sum(sizes)/len(sizes)/1024:.0f} KB, "
              f"total: {sum(sizes)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
