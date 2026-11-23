## QR4 - AUROC recovery (conditions where shift costs vanilla >= 0.01 AUROC; n=7 of 45)

| variant | k=5 | k=10 | k=25 | k=50 |
|---|---|---|---|---|
| b2a | 0.47 | 0.48 | 0.48 | 0.47 |
| b2b | 0.77 | 0.84 | 0.81 | 0.80 |
| oracle_a | 0.49 | 0.48 | 0.48 | 0.48 |

Cell = mean over (category, shift) of (AUROC_variant - AUROC_vanilla) / (AUROC_clean - AUROC_vanilla); 1.0 = fully recovered.

- b2a: gain(k=10)/gain(k=50) median 1.00 over 5 conditions with gain(50) > 0.005; share of conditions with ratio >= 0.5: 100%
- b2b: gain(k=10)/gain(k=50) median 1.00 over 6 conditions with gain(50) > 0.005; share of conditions with ratio >= 0.5: 100%

## Decision level at alpha = 0.1 (mean over shifted conditions, all categories)

| variant | k | AUROC | FPR@source-thr | FPR@target-conformal-thr | TPR@target-thr |
|---|---|---|---|---|---|
| none | 0 | 0.994 | 0.397 | - | - |
| none | 5 | 0.994 | 0.397 | 0.176 | 0.963 |
| none | 10 | 0.994 | 0.397 | 0.096 | 0.961 |
| none | 25 | 0.994 | 0.397 | 0.071 | 0.962 |
| none | 50 | 0.994 | 0.397 | 0.092 | 0.963 |
| b2a | 5 | 0.996 | 0.345 | 0.202 | 0.989 |
| b2a | 10 | 0.996 | 0.343 | 0.110 | 0.988 |
| b2a | 25 | 0.996 | 0.343 | 0.085 | 0.990 |
| b2a | 50 | 0.996 | 0.343 | 0.108 | 0.990 |
| b2b | 5 | 0.992 | 0.444 | 0.378 | 0.997 |
| b2b | 10 | 0.994 | 0.360 | 0.185 | 0.995 |
| b2b | 25 | 0.998 | 0.316 | 0.108 | 0.995 |
| b2b | 50 | 0.998 | 0.305 | 0.125 | 0.996 |
| oracle_a | 5 | 0.996 | 0.344 | 0.199 | 0.989 |
| oracle_a | 10 | 0.996 | 0.345 | 0.109 | 0.988 |
| oracle_a | 25 | 0.996 | 0.344 | 0.085 | 0.990 |
| oracle_a | 50 | 0.996 | 0.344 | 0.108 | 0.990 |

## QR3 - realised FPR of the conformal threshold calibrated on k target normals (alpha = 0.1)

| variant | k | mean FPR | p90 FPR | 99% band (theory) | cells over band | mean TPR |
|---|---|---|---|---|---|---|
| none | 5 | 0.176 | 0.478 | 0.60 | 4% | 0.963 |
| none | 10 | 0.096 | 0.300 | 0.37 | 7% | 0.961 |
| none | 25 | 0.071 | 0.250 | 0.24 | 11% | 0.962 |
| none | 50 | 0.092 | 0.250 | 0.21 | 14% | 0.963 |
| b2a | 5 | 0.202 | 0.500 | 0.60 | 4% | 0.989 |
| b2a | 10 | 0.110 | 0.344 | 0.37 | 9% | 0.988 |
| b2a | 25 | 0.085 | 0.250 | 0.24 | 10% | 0.990 |
| b2a | 50 | 0.109 | 0.253 | 0.21 | 17% | 0.990 |
| b2b | 5 | 0.378 | 0.801 | 0.60 | 25% | 0.997 |
| b2b | 10 | 0.185 | 0.450 | 0.37 | 18% | 0.995 |
| b2b | 25 | 0.108 | 0.300 | 0.24 | 16% | 0.995 |
| b2b | 50 | 0.125 | 0.300 | 0.21 | 24% | 0.996 |

(test-good sets are 20-40 images, so a single cell's FPR moves in steps of 2.5-5 points; read means, not cells.)
