# LightShift-PatchCore

PatchCore made robust to lighting shift, without retraining. Built on MVTec AD,
externally validated on MVTec AD 2 (can, vial, sheet_metal).

Project of the 14-week sujet "LightShift-PatchCore" (smartex-lab).
Local checkout lives on the machine SSD (`/mnt/data-ssd/lightshift-patchcore`);
the dataset lives at `/mnt/data-ssd/mvtec-ad-2`.

## Method

Given a vanilla PatchCore fitted on reference lighting:

- **B3 — conformal threshold.** Upload n normal images from the target lighting,
  take the n residuals, threshold = k-th smallest with k = ceil((n+1)(1-alpha)).
  Realized FPR small-sample behavior ~ Beta(n+1-k, k). No retraining.
- **B2 — output statistics alignment.** Align per-channel statistics of the
  patch embedding to the target-lighting distribution (AdaBN-style variants a/b,
  CORAL-style variant c), estimated from k target normals. Negative control:
  discoloured defects (per-instance normalization must not erase them).
- **B1 — memory-bank photometric augmentation** at equal bank-size budget (baseline).
- **B4 — other model** (EfficientAD / Reverse Distillation), comparator.
- **Reference implementation** from literature (SPARC / RDA) for comparison.

## Layout

```
docs/            hypotheses (QR1-QR6), plan, MVTec AD 2 checklist, sujet notes
configs/         per-category experiment configs (pinned)
src/lightshift/  package: data, shift generator, patchcore, b2, b3, metrics, api
scripts/         runnable entry points (fit, evaluate, calibrate, sweep)
receipts/        fingerprinted, resumable experiment receipts (JSON)
tests/           unit tests (equal-memory budgets, conformal coverage, metrics)
data/            symlinks only - no dataset files
```

## Data

MVTec AD 2, (c) MVTec Software GmbH, CC BY-NC-SA 4.0. Download via the request
form at mvtec.com. Private test rows are evaluated on the quota'd public server
at benchmark.mvtec.com; see docs/mvtec-ad2-checklist.md before any submission.
