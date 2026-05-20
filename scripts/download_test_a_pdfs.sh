#!/usr/bin/env bash
# Download all SPIQA test-A paper PDFs from arXiv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDF_DIR="$ROOT/data/benchmarks/spiqa/test-A/pdfs"
mkdir -p "$PDF_DIR"

python3 -c "
import json, os, sys, time, urllib.request
data = json.load(open('$ROOT/data/benchmarks/spiqa/test-A/SPIQA_testA.json'))
ids = list(data.keys())
print(f'{len(ids)} papers to download')
ok, skip, fail = 0, 0, 0
for i, pid in enumerate(ids, 1):
    out = '$PDF_DIR/' + pid + '.pdf'
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        skip += 1
        continue
    url = f'https://arxiv.org/pdf/{pid}.pdf'
    try:
        urllib.request.urlretrieve(url, out)
        ok += 1
        if ok % 10 == 0:
            print(f'  [{i}/{len(ids)}] downloaded {ok}, skipped {skip}, failed {fail}')
        time.sleep(0.5)  # courteous
    except Exception as e:
        fail += 1
        print(f'  FAILED {pid}: {type(e).__name__}: {e}')
print(f'DONE. ok={ok}, skip={skip}, fail={fail}')
"
