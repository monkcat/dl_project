# Experiment Results — element-graph v2.1 ablation

Loaded 44 completed runs.

## §7.1 Main ablation (Tier 1, rows a-h + k=gme)

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (a) baseline InfoNCE | enc only |  59.5 |  84.4 |  32.8 |  84.4 |  84.4 |   —   | 666 |
| (a) baseline InfoNCE | full α=0.3 T=2 |  46.1 |  69.8 |  29.3 |  69.8 |  69.8 |   —   | 666 |
| (b) +GPE type only, InfoNCE | enc only |  58.0 |  82.1 |  31.7 |  82.1 |  82.1 |   —   | 666 |
| (b) +GPE type only, InfoNCE | full α=0.3 T=2 |  45.5 |  69.7 |  28.6 |  69.7 |  69.7 |   —   | 666 |
| (c) +GPE type+role, InfoNCE | enc only |  58.7 |  83.8 |  32.3 |  83.8 |  83.8 |   —   | 666 |
| (c) +GPE type+role, InfoNCE | full α=0.3 T=2 |  46.1 |  69.2 |  29.1 |  69.2 |  69.2 |   —   | 666 |
| (d) +GPE full, InfoNCE | enc only |  54.7 |  79.9 |  29.2 |  79.9 |  79.9 |   —   | 666 |
| (d) +GPE full, InfoNCE | full α=0.3 T=2 |  42.8 |  66.2 |  27.4 |  66.2 |  66.2 |   —   | 666 |
| (e) GRCL, no GPE | enc only |  65.2 |  87.1 |  38.2 |  87.1 |  87.1 |   —   | 666 |
| (e) GRCL, no GPE | full α=0.3 T=2 |  45.6 |  70.3 |  30.4 |  70.3 |  70.3 |   —   | 666 |
| (f) GRCL + GPE | enc only |  62.2 |  85.1 |  35.2 |  85.1 |  85.1 |   —   | 666 |
| (f) GRCL + GPE | full α=0.3 T=2 |  47.4 |  70.9 |  29.8 |  70.9 |  70.9 |   —   | 666 |
| (g) GRCL + GPE + L_cov | enc only |  64.7 |  87.1 |  39.0 |  87.1 |  87.1 |   —   | 666 |
| (g) GRCL + GPE + L_cov | full α=0.3 T=2 |  47.3 |  70.0 |  30.6 |  70.0 |  70.0 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (k) GME zero-shot | enc only |  47.1 |  58.3 |  33.1 |  58.3 |  58.3 |   —   | 666 |
| (k) GME zero-shot | full α=0.3 T=2 |  78.2 |  84.4 |  63.3 |  84.4 |  84.4 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (a) baseline InfoNCE | enc only |   0.5 |   0.9 |   2.9 |   0.9 |   0.9 |  24.1 | 1623 |
| (a) baseline InfoNCE | full α=0.3 T=2 |   0.9 |   1.3 |   3.2 |   1.3 |   1.2 |  16.1 | 1623 |
| (b) +GPE type only, InfoNCE | enc only |   0.5 |   0.9 |   2.9 |   0.9 |   0.9 |  22.8 | 1623 |
| (b) +GPE type only, InfoNCE | full α=0.3 T=2 |   0.8 |   1.3 |   2.9 |   1.3 |   1.4 |  16.8 | 1623 |
| (c) +GPE type+role, InfoNCE | enc only |   0.5 |   1.1 |   2.8 |   1.1 |   1.0 |  22.5 | 1623 |
| (c) +GPE type+role, InfoNCE | full α=0.3 T=2 |   0.7 |   1.3 |   3.1 |   1.3 |   1.2 |  15.2 | 1623 |
| (d) +GPE full, InfoNCE | enc only |   0.5 |   1.1 |   2.9 |   1.1 |   1.0 |  20.3 | 1623 |
| (d) +GPE full, InfoNCE | full α=0.3 T=2 |   0.8 |   1.4 |   3.3 |   1.4 |   1.3 |  13.6 | 1623 |
| (e) GRCL, no GPE | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.6 | 1623 |
| (e) GRCL, no GPE | full α=0.3 T=2 |   1.0 |   1.5 |   3.6 |   1.5 |   1.0 |  17.4 | 1623 |
| (f) GRCL + GPE | enc only |   0.4 |   1.1 |   3.3 |   1.1 |   1.0 |  23.7 | 1623 |
| (f) GRCL + GPE | full α=0.3 T=2 |   0.9 |   1.4 |   3.8 |   1.4 |   1.1 |  15.5 | 1623 |
| (g) GRCL + GPE + L_cov | enc only |   0.5 |   1.0 |   3.4 |   1.0 |   0.9 |  27.8 | 1623 |
| (g) GRCL + GPE + L_cov | full α=0.3 T=2 |   0.9 |   1.4 |   3.7 |   1.4 |   1.1 |  17.1 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (k) GME zero-shot | enc only |   1.2 |   2.3 |   6.0 |   2.3 |   1.2 |  51.3 | 1623 |
| (k) GME zero-shot | full α=0.3 T=2 |   1.2 |   2.0 |   5.4 |   2.0 |   1.0 |  37.3 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (a) baseline InfoNCE | enc only |   1.1 |   2.0 |   2.7 |   2.0 |   2.6 |  18.6 | 389 |
| (a) baseline InfoNCE | full α=0.3 T=2 |   1.5 |   3.7 |   2.8 |   3.7 |   3.9 |   6.8 | 389 |
| (b) +GPE type only, InfoNCE | enc only |   1.0 |   2.5 |   2.7 |   2.5 |   3.1 |  16.9 | 389 |
| (b) +GPE type only, InfoNCE | full α=0.3 T=2 |   1.1 |   3.9 |   2.9 |   3.9 |   4.1 |   6.8 | 389 |
| (c) +GPE type+role, InfoNCE | enc only |   1.1 |   2.3 |   2.8 |   2.3 |   2.8 |  16.9 | 389 |
| (c) +GPE type+role, InfoNCE | full α=0.3 T=2 |   1.1 |   3.7 |   2.7 |   3.7 |   3.9 |   6.8 | 389 |
| (d) +GPE full, InfoNCE | enc only |   0.9 |   2.4 |   2.9 |   2.4 |   2.8 |  13.6 | 389 |
| (d) +GPE full, InfoNCE | full α=0.3 T=2 |   2.1 |   3.9 |   2.8 |   3.9 |   4.1 |   6.8 | 389 |
| (e) GRCL, no GPE | enc only |   1.7 |   2.5 |   2.9 |   2.5 |   2.8 |  15.3 | 389 |
| (e) GRCL, no GPE | full α=0.3 T=2 |   2.3 |   3.8 |   2.9 |   3.8 |   3.9 |   5.1 | 389 |
| (f) GRCL + GPE | enc only |   1.6 |   2.3 |   2.6 |   2.3 |   2.6 |  13.6 | 389 |
| (f) GRCL + GPE | full α=0.3 T=2 |   2.1 |   3.5 |   2.8 |   3.5 |   3.6 |   3.4 | 389 |
| (g) GRCL + GPE + L_cov | enc only |   1.9 |   2.6 |   3.1 |   2.6 |   2.8 |  18.6 | 389 |
| (g) GRCL + GPE + L_cov | full α=0.3 T=2 |   1.9 |   3.9 |   2.8 |   3.9 |   3.9 |   5.1 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (k) GME zero-shot | enc only |  15.7 |  18.9 |  13.9 |  18.9 |  17.7 |  25.4 | 389 |
| (k) GME zero-shot | full α=0.3 T=2 |  12.3 |  15.7 |  10.1 |  15.7 |  14.4 |  20.3 | 389 |

## §7.2 Negative controls (Tier 2, rows m-p)

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (m) shuffled section_role | enc only |  65.3 |  86.3 |  38.7 |  86.3 |  86.3 |   —   | 666 |
| (m) shuffled section_role | full α=0.3 T=2 |  45.9 |  70.7 |  30.3 |  70.7 |  70.7 |   —   | 666 |
| (n) random section_role | enc only |  63.5 |  85.1 |  37.6 |  85.1 |  85.1 |   —   | 666 |
| (n) random section_role | full α=0.3 T=2 |  47.0 |  70.6 |  29.9 |  70.6 |  70.6 |   —   | 666 |
| (o) no query PE dropout | enc only |  66.2 |  87.1 |  38.0 |  87.1 |  87.1 |   —   | 666 |
| (o) no query PE dropout | full α=0.3 T=2 |  46.7 |  70.4 |  30.3 |  70.4 |  70.4 |   —   | 666 |
| (p) encoder swap CLIP-L/14 | enc only |  68.0 |  88.1 |  36.2 |  88.1 |  88.1 |   —   | 666 |
| (p) encoder swap CLIP-L/14 | full α=0.3 T=2 |  51.1 |  73.9 |  31.9 |  73.9 |  73.9 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (m) shuffled section_role | enc only |   0.5 |   0.9 |   3.6 |   0.9 |   0.9 |  26.9 | 1623 |
| (m) shuffled section_role | full α=0.3 T=2 |   0.8 |   1.3 |   3.6 |   1.3 |   1.0 |  14.9 | 1623 |
| (n) random section_role | enc only |   0.4 |   1.2 |   3.6 |   1.2 |   1.0 |  26.9 | 1623 |
| (n) random section_role | full α=0.3 T=2 |   0.7 |   1.2 |   3.2 |   1.2 |   1.0 |  16.5 | 1623 |
| (o) no query PE dropout | enc only |   0.5 |   1.0 |   3.4 |   1.0 |   0.9 |  26.6 | 1623 |
| (o) no query PE dropout | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.1 |  18.4 | 1623 |
| (p) encoder swap CLIP-L/14 | enc only |   0.8 |   1.5 |   4.6 |   1.5 |   1.0 |  47.5 | 1623 |
| (p) encoder swap CLIP-L/14 | full α=0.3 T=2 |   0.8 |   1.5 |   4.2 |   1.5 |   1.0 |  25.3 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (m) shuffled section_role | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (m) shuffled section_role | full α=0.3 T=2 |   2.3 |   3.9 |   2.9 |   3.9 |   3.9 |   5.1 | 389 |
| (n) random section_role | enc only |   2.0 |   3.2 |   3.3 |   3.2 |   3.3 |  16.9 | 389 |
| (n) random section_role | full α=0.3 T=2 |   2.0 |   3.6 |   2.9 |   3.6 |   3.6 |   5.1 | 389 |
| (o) no query PE dropout | enc only |   1.7 |   2.5 |   2.9 |   2.5 |   2.6 |  20.3 | 389 |
| (o) no query PE dropout | full α=0.3 T=2 |   2.5 |   3.7 |   2.9 |   3.7 |   3.6 |   5.1 | 389 |
| (p) encoder swap CLIP-L/14 | enc only |   2.7 |   4.2 |   3.6 |   4.2 |   4.1 |  16.9 | 389 |
| (p) encoder swap CLIP-L/14 | full α=0.3 T=2 |   1.9 |   3.9 |   3.0 |   3.9 |   4.1 |   8.5 | 389 |

## §7.3 Sub-ablation sweeps (Tier 3)

### γ sweep (GRCL 2-hop decay)

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) gamma_03 | enc only |  59.9 |  83.9 |  34.4 |  83.9 |  83.9 |   —   | 666 |
| (sub) gamma_03 | full α=0.3 T=2 |  47.7 |  70.3 |  29.8 |  70.3 |  70.3 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (sub) gamma_07 | enc only |  61.1 |  85.7 |  35.3 |  85.7 |  85.7 |   —   | 666 |
| (sub) gamma_07 | full α=0.3 T=2 |  46.7 |  69.8 |  29.7 |  69.8 |  69.8 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) gamma_03 | enc only |   0.4 |   0.9 |   3.4 |   0.9 |   1.0 |  27.5 | 1623 |
| (sub) gamma_03 | full α=0.3 T=2 |   0.7 |   1.1 |   3.2 |   1.1 |   1.0 |  15.2 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (sub) gamma_07 | enc only |   0.5 |   1.2 |   3.3 |   1.2 |   1.2 |  25.0 | 1623 |
| (sub) gamma_07 | full α=0.3 T=2 |   0.9 |   1.4 |   3.6 |   1.4 |   1.1 |  14.2 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) gamma_03 | enc only |   1.8 |   2.5 |   3.4 |   2.5 |   2.8 |  16.9 | 389 |
| (sub) gamma_03 | full α=0.3 T=2 |   2.3 |   3.6 |   2.8 |   3.6 |   3.9 |  10.2 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (sub) gamma_07 | enc only |   1.4 |   2.4 |   2.6 |   2.4 |   2.6 |  13.6 | 389 |
| (sub) gamma_07 | full α=0.3 T=2 |   1.9 |   3.1 |   2.7 |   3.1 |   3.1 |   3.4 | 389 |

### λ_cov sweep

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) cov_00 | enc only |  66.2 |  85.4 |  38.0 |  85.4 |  85.4 |   —   | 666 |
| (sub) cov_00 | full α=0.3 T=2 |  46.7 |  70.9 |  30.5 |  70.9 |  70.9 |   —   | 666 |
| (sub) cov_01 | enc only |  66.4 |  87.1 |  37.7 |  87.1 |  87.1 |   —   | 666 |
| (sub) cov_01 | full α=0.3 T=2 |  45.9 |  71.0 |  30.1 |  71.0 |  71.0 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (sub) cov_05 | enc only |  65.3 |  87.1 |  37.3 |  87.1 |  87.1 |   —   | 666 |
| (sub) cov_05 | full α=0.3 T=2 |  45.2 |  69.8 |  29.9 |  69.8 |  69.8 |   —   | 666 |
| (sub) cov_10 | enc only |  63.1 |  85.7 |  38.1 |  85.7 |  85.7 |   —   | 666 |
| (sub) cov_10 | full α=0.3 T=2 |  46.1 |  71.0 |  29.1 |  71.0 |  71.0 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) cov_00 | enc only |   0.5 |   1.0 |   3.7 |   1.0 |   0.9 |  26.9 | 1623 |
| (sub) cov_00 | full α=0.3 T=2 |   0.9 |   1.5 |   3.7 |   1.5 |   1.0 |  17.7 | 1623 |
| (sub) cov_01 | enc only |   0.4 |   1.0 |   3.5 |   1.0 |   0.9 |  26.6 | 1623 |
| (sub) cov_01 | full α=0.3 T=2 |   0.8 |   1.5 |   3.6 |   1.5 |   1.1 |  15.8 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (sub) cov_05 | enc only |   0.4 |   0.9 |   3.4 |   0.9 |   0.9 |  27.5 | 1623 |
| (sub) cov_05 | full α=0.3 T=2 |   0.9 |   1.4 |   3.6 |   1.4 |   1.1 |  17.4 | 1623 |
| (sub) cov_10 | enc only |   0.4 |   1.0 |   3.5 |   1.0 |   0.9 |  27.8 | 1623 |
| (sub) cov_10 | full α=0.3 T=2 |   1.0 |   1.5 |   3.6 |   1.5 |   1.2 |  15.2 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) cov_00 | enc only |   1.5 |   2.5 |   2.9 |   2.5 |   2.6 |  15.3 | 389 |
| (sub) cov_00 | full α=0.3 T=2 |   2.5 |   4.2 |   2.9 |   4.2 |   4.1 |   6.8 | 389 |
| (sub) cov_01 | enc only |   1.6 |   2.4 |   3.0 |   2.4 |   2.6 |  22.0 | 389 |
| (sub) cov_01 | full α=0.3 T=2 |   2.1 |   3.8 |   3.0 |   3.8 |   3.6 |   5.1 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (sub) cov_05 | enc only |   1.7 |   2.4 |   2.8 |   2.4 |   2.6 |  18.6 | 389 |
| (sub) cov_05 | full α=0.3 T=2 |   2.1 |   3.4 |   2.8 |   3.4 |   3.3 |   5.1 | 389 |
| (sub) cov_10 | enc only |   1.6 |   2.5 |   2.8 |   2.5 |   2.6 |  20.3 | 389 |
| (sub) cov_10 | full α=0.3 T=2 |   1.9 |   3.4 |   2.7 |   3.4 |   3.3 |   5.1 | 389 |

### λ_cons sweep

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) cons_00 | enc only |  64.7 |  86.2 |  37.8 |  86.2 |  86.2 |   —   | 666 |
| (sub) cons_00 | full α=0.3 T=2 |  46.5 |  70.9 |  30.4 |  70.9 |  70.9 |   —   | 666 |
| (sub) cons_01 | enc only |  64.4 |  85.7 |  35.1 |  85.7 |  85.7 |   —   | 666 |
| (sub) cons_01 | full α=0.3 T=2 |  45.3 |  70.0 |  29.4 |  70.0 |  70.0 |   —   | 666 |
| (sub) cons_03 | enc only |  63.5 |  86.9 |  34.9 |  86.9 |  86.9 |   —   | 666 |
| (sub) cons_03 | full α=0.3 T=2 |  45.8 |  70.1 |  29.5 |  70.1 |  70.1 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (sub) cons_10 | enc only |  62.2 |  86.0 |  35.3 |  86.0 |  86.0 |   —   | 666 |
| (sub) cons_10 | full α=0.3 T=2 |  45.9 |  70.3 |  29.7 |  70.3 |  70.3 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) cons_00 | enc only |   0.4 |   0.9 |   3.5 |   0.9 |   0.9 |  28.5 | 1623 |
| (sub) cons_00 | full α=0.3 T=2 |   0.8 |   1.5 |   3.4 |   1.5 |   1.1 |  15.5 | 1623 |
| (sub) cons_01 | enc only |   0.5 |   1.1 |   3.4 |   1.1 |   1.0 |  22.5 | 1623 |
| (sub) cons_01 | full α=0.3 T=2 |   0.9 |   1.3 |   3.7 |   1.3 |   1.1 |  14.9 | 1623 |
| (sub) cons_03 | enc only |   0.5 |   1.0 |   3.3 |   1.0 |   1.0 |  23.4 | 1623 |
| (sub) cons_03 | full α=0.3 T=2 |   0.9 |   1.4 |   3.4 |   1.4 |   1.1 |  15.5 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (sub) cons_10 | enc only |   0.4 |   1.1 |   3.4 |   1.1 |   1.0 |  23.4 | 1623 |
| (sub) cons_10 | full α=0.3 T=2 |   0.9 |   1.3 |   3.5 |   1.3 |   1.1 |  15.5 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) cons_00 | enc only |   1.6 |   2.5 |   2.8 |   2.5 |   2.6 |  20.3 | 389 |
| (sub) cons_00 | full α=0.3 T=2 |   1.8 |   3.6 |   2.9 |   3.6 |   3.3 |   3.4 | 389 |
| (sub) cons_01 | enc only |   1.7 |   2.4 |   2.6 |   2.4 |   2.6 |  18.6 | 389 |
| (sub) cons_01 | full α=0.3 T=2 |   1.9 |   3.1 |   2.7 |   3.1 |   3.3 |   3.4 | 389 |
| (sub) cons_03 | enc only |   1.6 |   2.2 |   2.8 |   2.2 |   2.6 |  16.9 | 389 |
| (sub) cons_03 | full α=0.3 T=2 |   1.7 |   3.2 |   2.7 |   3.2 |   3.3 |   3.4 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (sub) cons_10 | enc only |   1.6 |   2.4 |   2.9 |   2.4 |   2.6 |  16.9 | 389 |
| (sub) cons_10 | full α=0.3 T=2 |   1.9 |   3.2 |   2.7 |   3.2 |   3.3 |   3.4 | 389 |

### LoRA rank sweep

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) lora_r4 | enc only |  60.4 |  83.0 |  34.7 |  83.0 |  83.0 |   —   | 666 |
| (sub) lora_r4 | full α=0.3 T=2 |  43.1 |  67.3 |  27.3 |  67.3 |  67.3 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (sub) lora_r16 | enc only |  62.8 |  86.2 |  36.2 |  86.2 |  86.2 |   —   | 666 |
| (sub) lora_r16 | full α=0.3 T=2 |  48.5 |  71.6 |  30.9 |  71.6 |  71.6 |   —   | 666 |
| (sub) lora_r32 | enc only |  67.0 |  84.7 |  38.8 |  84.7 |  84.7 |   —   | 666 |
| (sub) lora_r32 | full α=0.3 T=2 |  48.8 |  70.4 |  30.1 |  70.4 |  70.4 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) lora_r4 | enc only |   0.6 |   1.2 |   3.6 |   1.2 |   1.0 |  27.5 | 1623 |
| (sub) lora_r4 | full α=0.3 T=2 |   0.8 |   1.4 |   3.9 |   1.4 |   1.1 |  19.3 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (sub) lora_r16 | enc only |   0.9 |   1.4 |   3.4 |   1.4 |   1.2 |  22.5 | 1623 |
| (sub) lora_r16 | full α=0.3 T=2 |   0.9 |   1.5 |   3.8 |   1.5 |   1.1 |  14.9 | 1623 |
| (sub) lora_r32 | enc only |   0.7 |   1.3 |   4.0 |   1.3 |   0.9 |  29.7 | 1623 |
| (sub) lora_r32 | full α=0.3 T=2 |   0.7 |   1.5 |   3.9 |   1.5 |   1.1 |  16.1 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) lora_r4 | enc only |   1.3 |   2.1 |   2.6 |   2.1 |   2.3 |  20.3 | 389 |
| (sub) lora_r4 | full α=0.3 T=2 |   2.3 |   3.8 |   3.0 |   3.8 |   3.6 |   8.5 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (sub) lora_r16 | enc only |   2.2 |   3.1 |   3.0 |   3.1 |   3.6 |  10.2 | 389 |
| (sub) lora_r16 | full α=0.3 T=2 |   2.3 |   4.3 |   3.6 |   4.3 |   4.1 |   3.4 | 389 |
| (sub) lora_r32 | enc only |   1.5 |   2.2 |   2.9 |   2.2 |   2.3 |  15.3 | 389 |
| (sub) lora_r32 | full α=0.3 T=2 |   2.4 |   3.7 |   2.9 |   3.7 |   3.9 |   6.8 | 389 |

### Learning rate sweep

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) lr_1e5 | enc only |  55.4 |  79.9 |  32.8 |  79.9 |  79.9 |   —   | 666 |
| (sub) lr_1e5 | full α=0.3 T=2 |  41.6 |  63.4 |  26.4 |  63.4 |  63.4 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (sub) lr_1e4 | enc only |  70.1 |  88.6 |  42.6 |  88.6 |  88.6 |   —   | 666 |
| (sub) lr_1e4 | full α=0.3 T=2 |  52.3 |  73.7 |  30.9 |  73.7 |  73.7 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) lr_1e5 | enc only |   0.6 |   1.3 |   2.9 |   1.3 |   1.1 |  20.9 | 1623 |
| (sub) lr_1e5 | full α=0.3 T=2 |   1.0 |   1.4 |   3.3 |   1.4 |   1.2 |  13.9 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (sub) lr_1e4 | enc only |   0.8 |   1.6 |   4.4 |   1.6 |   1.1 |  33.9 | 1623 |
| (sub) lr_1e4 | full α=0.3 T=2 |   0.8 |   1.4 |   4.4 |   1.4 |   1.0 |  20.6 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) lr_1e5 | enc only |   0.8 |   1.4 |   2.3 |   1.4 |   1.8 |  11.9 | 389 |
| (sub) lr_1e5 | full α=0.3 T=2 |   1.5 |   2.2 |   2.4 |   2.2 |   2.3 |   5.1 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (sub) lr_1e4 | enc only |   2.9 |   4.3 |   3.5 |   4.3 |   4.4 |  20.3 | 389 |
| (sub) lr_1e4 | full α=0.3 T=2 |   3.5 |   6.2 |   3.8 |   6.2 |   6.2 |  13.6 | 389 |

### Temperature τ sweep

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) tau_005 | enc only |  65.0 |  85.6 |  37.8 |  85.6 |  85.6 |   —   | 666 |
| (sub) tau_005 | full α=0.3 T=2 |  45.5 |  70.4 |  29.7 |  70.4 |  70.4 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| (sub) tau_010 | enc only |  64.9 |  86.2 |  37.2 |  86.2 |  86.2 |   —   | 666 |
| (sub) tau_010 | full α=0.3 T=2 |  45.3 |  70.4 |  28.9 |  70.4 |  70.4 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) tau_005 | enc only |   0.6 |   1.0 |   3.3 |   1.0 |   1.0 |  25.9 | 1623 |
| (sub) tau_005 | full α=0.3 T=2 |   1.0 |   1.4 |   3.5 |   1.4 |   1.1 |  16.5 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| (sub) tau_010 | enc only |   0.6 |   1.0 |   3.5 |   1.0 |   0.9 |  23.7 | 1623 |
| (sub) tau_010 | full α=0.3 T=2 |   0.9 |   1.3 |   3.2 |   1.3 |   1.1 |  15.8 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) tau_005 | enc only |   1.0 |   2.6 |   2.8 |   2.6 |   2.8 |  13.6 | 389 |
| (sub) tau_005 | full α=0.3 T=2 |   1.8 |   3.4 |   2.8 |   3.4 |   3.3 |   5.1 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| (sub) tau_010 | enc only |   1.4 |   2.2 |   2.8 |   2.2 |   2.3 |  18.6 | 389 |
| (sub) tau_010 | full α=0.3 T=2 |   1.3 |   3.2 |   2.5 |   3.2 |   3.1 |   5.1 | 389 |

### Anchor kind sweep

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) anchor_caption | enc only |  64.1 |  86.0 |  39.6 |  86.0 |  86.0 |   —   | 666 |
| (sub) anchor_caption | full α=0.3 T=2 |  44.6 |  63.5 |  27.4 |  63.5 |  63.5 |   —   | 666 |
| (sub) anchor_refer | enc only |  49.4 |  74.9 |  29.1 |  74.9 |  74.9 |   —   | 666 |
| (sub) anchor_refer | full α=0.3 T=2 |  43.8 |  64.0 |  27.2 |  64.0 |  64.0 |   —   | 666 |
| (sub) anchor_nlqa | enc only |  64.4 |  84.8 |  37.7 |  84.8 |  84.8 |   —   | 666 |
| (sub) anchor_nlqa | full α=0.3 T=2 |  46.2 |  69.8 |  29.4 |  69.8 |  69.8 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) anchor_caption | enc only |   0.7 |   1.3 |   3.3 |   1.3 |   1.1 |  19.9 | 1623 |
| (sub) anchor_caption | full α=0.3 T=2 |   0.8 |   1.2 |   2.8 |   1.2 |   1.1 |  13.3 | 1623 |
| (sub) anchor_refer | enc only |   0.5 |   1.4 |   3.6 |   1.4 |   1.2 |  27.8 | 1623 |
| (sub) anchor_refer | full α=0.3 T=2 |   1.1 |   1.8 |   3.7 |   1.8 |   1.4 |  19.0 | 1623 |
| (sub) anchor_nlqa | enc only |   0.7 |   1.4 |   3.6 |   1.4 |   1.0 |  26.9 | 1623 |
| (sub) anchor_nlqa | full α=0.3 T=2 |   0.8 |   1.3 |   3.8 |   1.3 |   1.1 |  16.1 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) anchor_caption | enc only |   0.6 |   1.5 |   1.8 |   1.5 |   2.1 |  13.6 | 389 |
| (sub) anchor_caption | full α=0.3 T=2 |   1.3 |   3.0 |   2.9 |   3.0 |   3.1 |   5.1 | 389 |
| (sub) anchor_refer | enc only |   1.0 |   1.8 |   2.5 |   1.8 |   2.1 |   8.5 | 389 |
| (sub) anchor_refer | full α=0.3 T=2 |   1.3 |   2.6 |   2.5 |   2.6 |   3.1 |   6.8 | 389 |
| (sub) anchor_nlqa | enc only |   1.4 |   2.2 |   3.1 |   2.2 |   2.6 |  15.3 | 389 |
| (sub) anchor_nlqa | full α=0.3 T=2 |   2.5 |   3.5 |   2.7 |   3.5 |   3.6 |   3.4 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |

### Edge type isolation in GRCL

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) edge_caption_of | enc only |  60.7 |  83.5 |  32.7 |  83.5 |  83.5 |   —   | 666 |
| (sub) edge_caption_of | full α=0.3 T=2 |  43.2 |  67.4 |  28.2 |  67.4 |  67.4 |   —   | 666 |
| (sub) edge_refer_to | enc only |  70.7 |  90.4 |  41.9 |  90.4 |  90.4 |   —   | 666 |
| (sub) edge_refer_to | full α=0.3 T=2 |  38.1 |  50.2 |  25.4 |  50.2 |  50.2 |   —   | 666 |
| (sub) edge_contains | enc only |  70.7 |  88.4 |  44.1 |  88.4 |  88.4 |   —   | 666 |
| (sub) edge_contains | full α=0.3 T=2 |  47.1 |  68.8 |  28.5 |  68.8 |  68.8 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) edge_caption_of | enc only |   0.6 |   1.3 |   3.1 |   1.3 |   1.1 |  29.4 | 1623 |
| (sub) edge_caption_of | full α=0.3 T=2 |   0.9 |   1.5 |   3.1 |   1.5 |   1.4 |  15.5 | 1623 |
| (sub) edge_refer_to | enc only |   0.9 |   1.3 |   2.9 |   1.3 |   0.9 |  24.7 | 1623 |
| (sub) edge_refer_to | full α=0.3 T=2 |   1.0 |   1.3 |   2.7 |   1.3 |   0.9 |  15.8 | 1623 |
| (sub) edge_contains | enc only |   0.8 |   1.4 |   4.0 |   1.4 |   1.0 |  31.3 | 1623 |
| (sub) edge_contains | full α=0.3 T=2 |   0.6 |   1.1 |   3.5 |   1.1 |   0.9 |  19.6 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) edge_caption_of | enc only |   1.8 |   3.1 |   3.2 |   3.1 |   3.6 |  13.6 | 389 |
| (sub) edge_caption_of | full α=0.3 T=2 |   1.5 |   3.7 |   2.8 |   3.7 |   3.9 |   1.7 | 389 |
| (sub) edge_refer_to | enc only |   3.0 |   5.7 |   3.6 |   5.7 |   6.2 |  22.0 | 389 |
| (sub) edge_refer_to | full α=0.3 T=2 |   1.5 |   3.6 |   3.2 |   3.6 |   3.9 |  16.9 | 389 |
| (sub) edge_contains | enc only |   0.8 |   2.1 |   2.1 |   2.1 |   2.3 |  15.3 | 389 |
| (sub) edge_contains | full α=0.3 T=2 |   2.1 |   3.8 |   3.2 |   3.8 |   3.9 |   6.8 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |

### Visual tokens per element

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) tokens_016 | enc only |  62.0 |  83.0 |  33.8 |  83.0 |  83.0 |   —   | 666 |
| (sub) tokens_016 | full α=0.3 T=2 |  46.7 |  70.3 |  29.7 |  70.3 |  70.3 |   —   | 666 |
| (sub) tokens_064 | enc only |  62.9 |  84.7 |  37.9 |  84.7 |  84.7 |   —   | 666 |
| (sub) tokens_064 | full α=0.3 T=2 |  45.8 |  70.6 |  30.0 |  70.6 |  70.6 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) tokens_016 | enc only |   0.5 |   1.1 |   3.2 |   1.1 |   1.0 |  24.4 | 1623 |
| (sub) tokens_016 | full α=0.3 T=2 |   1.0 |   1.4 |   3.6 |   1.4 |   1.2 |  14.6 | 1623 |
| (sub) tokens_064 | enc only |   0.4 |   1.0 |   3.2 |   1.0 |   0.9 |  25.9 | 1623 |
| (sub) tokens_064 | full α=0.3 T=2 |   0.8 |   1.4 |   3.4 |   1.4 |   1.2 |  14.2 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| (sub) tokens_016 | enc only |   1.8 |   2.4 |   3.2 |   2.4 |   2.6 |  16.9 | 389 |
| (sub) tokens_016 | full α=0.3 T=2 |   1.4 |   2.9 |   2.5 |   2.9 |   3.1 |   3.4 | 389 |
| (sub) tokens_064 | enc only |   1.5 |   2.6 |   3.0 |   2.6 |   2.8 |  16.9 | 389 |
| (sub) tokens_064 | full α=0.3 T=2 |   2.0 |   3.3 |   2.7 |   3.3 |   3.3 |   5.1 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |

## §7.4 Follow-up extensions (Tier 4)

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h + best-HP combo | enc only |  66.5 |  88.0 |  42.2 |  88.0 |  88.0 |   —   | 666 |
| h + best-HP combo | full α=0.3 T=2 |  52.0 |  74.9 |  30.3 |  74.9 |  74.9 |   —   | 666 |
| h + best-HP combo, 16k steps | enc only |  66.4 |  85.9 |  39.7 |  85.9 |  85.9 |   —   | 666 |
| h + best-HP combo, 16k steps | full α=0.3 T=2 |  52.0 |  74.0 |  31.5 |  74.0 |  74.0 |   —   | 666 |
| h + stronger GPE init | enc only |  61.4 |  85.1 |  34.9 |  85.1 |  85.1 |   —   | 666 |
| h + stronger GPE init | full α=0.3 T=2 |  45.5 |  70.6 |  29.4 |  70.6 |  70.6 |   —   | 666 |
| h (seed=43) | enc only |  59.0 |  85.6 |  35.9 |  85.6 |  85.6 |   —   | 666 |
| h (seed=43) | full α=0.3 T=2 |  45.8 |  70.1 |  28.9 |  70.1 |  70.1 |   —   | 666 |
| h (seed=44) | enc only |  61.3 |  83.0 |  35.0 |  83.0 |  83.0 |   —   | 666 |
| h (seed=44) | full α=0.3 T=2 |  44.4 |  68.9 |  29.5 |  68.9 |  68.9 |   —   | 666 |
| (h) Full method | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| (h) Full method | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h + best-HP combo | enc only |   0.8 |   1.5 |   4.4 |   1.5 |   1.0 |  32.3 | 1623 |
| h + best-HP combo | full α=0.3 T=2 |   0.6 |   1.2 |   4.1 |   1.2 |   0.9 |  17.4 | 1623 |
| h + best-HP combo, 16k steps | enc only |   0.9 |   1.7 |   4.3 |   1.7 |   1.2 |  31.6 | 1623 |
| h + best-HP combo, 16k steps | full α=0.3 T=2 |   0.8 |   1.4 |   4.4 |   1.4 |   1.0 |  16.1 | 1623 |
| h + stronger GPE init | enc only |   0.5 |   1.0 |   3.4 |   1.0 |   1.0 |  24.4 | 1623 |
| h + stronger GPE init | full α=0.3 T=2 |   0.9 |   1.3 |   3.9 |   1.3 |   1.1 |  14.9 | 1623 |
| h (seed=43) | enc only |   0.7 |   1.2 |   3.2 |   1.2 |   1.0 |  21.5 | 1623 |
| h (seed=43) | full α=0.3 T=2 |   0.6 |   1.1 |   2.9 |   1.1 |   1.0 |  18.0 | 1623 |
| h (seed=44) | enc only |   0.5 |   1.0 |   3.2 |   1.0 |   0.9 |  28.5 | 1623 |
| h (seed=44) | full α=0.3 T=2 |   0.9 |   1.4 |   3.9 |   1.4 |   1.1 |  15.8 | 1623 |
| (h) Full method | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| (h) Full method | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h + best-HP combo | enc only |   3.2 |   4.2 |   4.0 |   4.2 |   4.1 |  18.6 | 389 |
| h + best-HP combo | full α=0.3 T=2 |   3.0 |   5.8 |   3.7 |   5.8 |   5.7 |   8.5 | 389 |
| h + best-HP combo, 16k steps | enc only |   3.3 |   4.4 |   4.1 |   4.4 |   4.6 |  20.3 | 389 |
| h + best-HP combo, 16k steps | full α=0.3 T=2 |   2.7 |   5.3 |   3.4 |   5.3 |   5.4 |  10.2 | 389 |
| h + stronger GPE init | enc only |   1.9 |   2.4 |   2.9 |   2.4 |   2.6 |  16.9 | 389 |
| h + stronger GPE init | full α=0.3 T=2 |   2.0 |   3.1 |   2.7 |   3.1 |   3.1 |   5.1 | 389 |
| h (seed=43) | enc only |   1.4 |   2.2 |   2.5 |   2.2 |   2.6 |  13.6 | 389 |
| h (seed=43) | full α=0.3 T=2 |   1.5 |   2.7 |   2.6 |   2.7 |   2.8 |   6.8 | 389 |
| h (seed=44) | enc only |   1.6 |   2.5 |   3.0 |   2.5 |   2.6 |  11.9 | 389 |
| h (seed=44) | full α=0.3 T=2 |   2.4 |   3.5 |   2.8 |   3.5 |   3.3 |   3.4 | 389 |
| (h) Full method | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| (h) Full method | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |

## §7.5 Propagation sub-ablation on (h) — 3 regimes × hyperparam sweep

All variants applied to the (h) checkpoint (eval-only).

### Group A — Uniform weights (structure only, modifier-design ablation)

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| h_prop_sweep | uniform α=0.1 T=2 |  41.3 |  71.5 |  28.0 |  71.5 |  71.5 |   —   | 666 |
| h_prop_sweep | uniform α=0.3 T=1 |  43.1 |  74.5 |  28.2 |  74.5 |  74.5 |   —   | 666 |
| h_prop_sweep | uniform α=0.3 T=2 |  40.1 |  67.0 |  27.4 |  67.0 |  67.0 |   —   | 666 |
| h_prop_sweep | uniform α=0.3 T=3 |  42.6 |  66.8 |  28.3 |  66.8 |  66.8 |   —   | 666 |
| h_prop_sweep | uniform α=0.5 T=2 |  41.6 |  65.3 |  28.1 |  65.3 |  65.3 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| h_prop_sweep | uniform α=0.1 T=2 |   0.7 |   1.3 |   3.5 |   1.3 |   1.0 |  17.4 | 1623 |
| h_prop_sweep | uniform α=0.3 T=1 |   0.7 |   1.3 |   3.4 |   1.3 |   1.0 |  18.0 | 1623 |
| h_prop_sweep | uniform α=0.3 T=2 |   0.7 |   1.3 |   3.5 |   1.3 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | uniform α=0.3 T=3 |   0.8 |   1.3 |   3.5 |   1.3 |   1.0 |  16.1 | 1623 |
| h_prop_sweep | uniform α=0.5 T=2 |   0.8 |   1.3 |   3.5 |   1.3 |   1.0 |  15.5 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| h_prop_sweep | uniform α=0.1 T=2 |   2.5 |   4.1 |   2.8 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | uniform α=0.3 T=1 |   2.4 |   4.1 |   2.8 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | uniform α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | uniform α=0.3 T=3 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | uniform α=0.5 T=2 |   2.3 |   4.1 |   3.0 |   4.1 |   3.9 |   5.1 | 389 |

### Group B — Weighted diffusion (modifier ablation + α/T sweep)

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| h_prop_sweep | base α=0.3 T=2 |  46.2 |  70.7 |  29.9 |  70.7 |  70.7 |   —   | 666 |
| h_prop_sweep | base+role α=0.3 T=2 |  46.2 |  70.7 |  29.9 |  70.7 |  70.7 |   —   | 666 |
| h_prop_sweep | base+visual α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| h_prop_sweep | full α=0.3 T=2 |  46.5 |  70.7 |  30.0 |  70.7 |  70.7 |   —   | 666 |
| h_prop_sweep | full α=0.1 T=2 |  48.8 |  74.3 |  30.6 |  74.3 |  74.3 |   —   | 666 |
| h_prop_sweep | full α=0.5 T=2 |  46.2 |  67.6 |  30.7 |  67.6 |  67.6 |   —   | 666 |
| h_prop_sweep | full α=0.7 T=2 |  42.3 |  65.5 |  28.5 |  65.5 |  65.5 |   —   | 666 |
| h_prop_sweep | full α=0.3 T=1 |  50.6 |  78.7 |  31.4 |  78.7 |  78.7 |   —   | 666 |
| h_prop_sweep | full α=0.3 T=3 |  46.2 |  68.9 |  30.5 |  68.9 |  68.9 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| h_prop_sweep | base α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | base+role α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | base+visual α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | full α=0.3 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | full α=0.1 T=2 |   0.8 |   1.3 |   3.6 |   1.3 |   1.0 |  17.4 | 1623 |
| h_prop_sweep | full α=0.5 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  15.5 | 1623 |
| h_prop_sweep | full α=0.7 T=2 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | full α=0.3 T=1 |   0.8 |   1.3 |   3.5 |   1.3 |   1.0 |  18.0 | 1623 |
| h_prop_sweep | full α=0.3 T=3 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.1 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| h_prop_sweep | base α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | base+role α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | base+visual α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | full α=0.3 T=2 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | full α=0.1 T=2 |   2.5 |   4.1 |   2.8 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | full α=0.5 T=2 |   2.3 |   4.1 |   3.0 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | full α=0.7 T=2 |   2.2 |   4.2 |   2.9 |   4.2 |   4.1 |   5.1 | 389 |
| h_prop_sweep | full α=0.3 T=1 |   2.4 |   4.1 |   2.8 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | full α=0.3 T=3 |   2.3 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |

### Group C — Personalized PageRank

### SPIQA test-A

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |  64.9 |  86.9 |  37.4 |  86.9 |  86.9 |   —   | 666 |
| h_prop_sweep | PPR α=0.15 |  41.0 |  63.5 |  28.2 |  63.5 |  63.5 |   —   | 666 |
| h_prop_sweep | PPR α=0.30 |  44.0 |  65.9 |  29.6 |  65.9 |  65.9 |   —   | 666 |
| h_prop_sweep | PPR α=0.50 |  47.3 |  69.1 |  31.4 |  69.1 |  69.1 |   —   | 666 |
| h_prop_sweep | PPR α=0.70 |  48.0 |  72.2 |  31.0 |  72.2 |  72.2 |   —   | 666 |
| h_prop_sweep | PPR α=0.85 |  48.5 |  73.4 |  30.9 |  73.4 |  73.4 |   —   | 666 |
| h_prop_sweep | PPR α=0.30 (uniform) |  42.6 |  64.4 |  29.3 |  64.4 |  64.4 |   —   | 666 |
| h_prop_sweep | PPR α=0.50 (uniform) |  43.8 |  68.3 |  29.6 |  68.3 |  68.3 |   —   | 666 |

### SciEGQA

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |   0.5 |   1.0 |   3.5 |   1.0 |   0.9 |  26.9 | 1623 |
| h_prop_sweep | PPR α=0.15 |   0.9 |   1.5 |   3.6 |   1.5 |   1.0 |  15.8 | 1623 |
| h_prop_sweep | PPR α=0.30 |   0.8 |   1.4 |   3.6 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | PPR α=0.50 |   0.8 |   1.4 |   3.5 |   1.4 |   1.0 |  17.1 | 1623 |
| h_prop_sweep | PPR α=0.70 |   0.8 |   1.3 |   3.5 |   1.3 |   1.0 |  18.0 | 1623 |
| h_prop_sweep | PPR α=0.85 |   0.7 |   1.3 |   3.5 |   1.3 |   1.0 |  17.4 | 1623 |
| h_prop_sweep | PPR α=0.30 (uniform) |   0.8 |   1.4 |   3.5 |   1.4 |   1.0 |  16.5 | 1623 |
| h_prop_sweep | PPR α=0.50 (uniform) |   0.8 |   1.3 |   3.4 |   1.3 |   1.0 |  17.1 | 1623 |

### MMDocIR

| Row | Variant | R@5 | R@10 | MRR | Cov@10 | Pf@10 | Xpg | n |
|---|---|---|---|---|---|---|---|---|
| h_prop_sweep | enc only |   1.6 |   2.4 |   2.8 |   2.4 |   2.6 |  20.3 | 389 |
| h_prop_sweep | PPR α=0.15 |   2.1 |   3.9 |   2.9 |   3.9 |   3.9 |   5.1 | 389 |
| h_prop_sweep | PPR α=0.30 |   2.1 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | PPR α=0.50 |   2.3 |   3.9 |   2.8 |   3.9 |   3.9 |   5.1 | 389 |
| h_prop_sweep | PPR α=0.70 |   2.5 |   3.9 |   2.8 |   3.9 |   3.9 |   5.1 | 389 |
| h_prop_sweep | PPR α=0.85 |   2.3 |   4.1 |   2.8 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | PPR α=0.30 (uniform) |   2.1 |   4.1 |   2.9 |   4.1 |   3.9 |   5.1 | 389 |
| h_prop_sweep | PPR α=0.50 (uniform) |   2.3 |   3.9 |   2.8 |   3.9 |   3.9 |   5.1 | 389 |

## §7.6 Quick comparison — R@10 (no propagation)

| Row | SPIQA test-A | SciEGQA | MMDocIR |
|---|---|---|---|
| (a) baseline InfoNCE |  84.4 |   0.9 |   2.0 |
| (b) +GPE type only, InfoNCE |  82.1 |   0.9 |   2.5 |
| (c) +GPE type+role, InfoNCE |  83.8 |   1.1 |   2.3 |
| (d) +GPE full, InfoNCE |  79.9 |   1.1 |   2.4 |
| (e) GRCL, no GPE |  87.1 |   1.0 |   2.5 |
| (f) GRCL + GPE |  85.1 |   1.1 |   2.3 |
| (g) GRCL + GPE + L_cov |  87.1 |   1.0 |   2.6 |
| (h) Full method |  86.9 |   1.0 |   2.4 |
| (k) GME zero-shot |  58.3 |   2.3 |  18.9 |
| (m) shuffled section_role |  86.3 |   0.9 |   2.4 |
| (n) random section_role |  85.1 |   1.2 |   3.2 |
| (o) no query PE dropout |  87.1 |   1.0 |   2.5 |
| (p) encoder swap CLIP-L/14 |  88.1 |   1.5 |   4.2 |
| (sub) gamma_03 |  83.9 |   0.9 |   2.5 |
| (sub) gamma_07 |  85.7 |   1.2 |   2.4 |
| (sub) cov_00 |  85.4 |   1.0 |   2.5 |
| (sub) cov_01 |  87.1 |   1.0 |   2.4 |
| (sub) cov_05 |  87.1 |   0.9 |   2.4 |
| (sub) cov_10 |  85.7 |   1.0 |   2.5 |
| (sub) cons_00 |  86.2 |   0.9 |   2.5 |
| (sub) cons_01 |  85.7 |   1.1 |   2.4 |
| (sub) cons_03 |  86.9 |   1.0 |   2.2 |
| (sub) cons_10 |  86.0 |   1.1 |   2.4 |
| (sub) lora_r4 |  83.0 |   1.2 |   2.1 |
| (sub) lora_r16 |  86.2 |   1.4 |   3.1 |
| (sub) lora_r32 |  84.7 |   1.3 |   2.2 |
| (sub) lr_1e5 |  79.9 |   1.3 |   1.4 |
| (sub) lr_1e4 |  88.6 |   1.6 |   4.3 |
| (sub) tau_005 |  85.6 |   1.0 |   2.6 |
| (sub) tau_010 |  86.2 |   1.0 |   2.2 |
| (sub) anchor_caption |  86.0 |   1.3 |   1.5 |
| (sub) anchor_refer |  74.9 |   1.4 |   1.8 |
| (sub) anchor_nlqa |  84.8 |   1.4 |   2.2 |
| (sub) edge_caption_of |  83.5 |   1.3 |   3.1 |
| (sub) edge_refer_to |  90.4 |   1.3 |   5.7 |
| (sub) edge_contains |  88.4 |   1.4 |   2.1 |
| (sub) tokens_016 |  83.0 |   1.1 |   2.4 |
| (sub) tokens_064 |  84.7 |   1.0 |   2.6 |
| h + best-HP combo |  88.0 |   1.5 |   4.2 |
| h + best-HP combo, 16k steps |  85.9 |   1.7 |   4.4 |
| h + stronger GPE init |  85.1 |   1.0 |   2.4 |
| h (seed=43) |  85.6 |   1.2 |   2.2 |
| h (seed=44) |  83.0 |   1.0 |   2.5 |

