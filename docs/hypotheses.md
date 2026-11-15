# Registered hypotheses and research questions

Registered before running any experiment. Each entry is dated and falsifiable.
Later results must reference these IDs; hedges and amendments get a new dated
section, never an in-place edit.

Registration: 2026-11-15 (week 0, before any LightShift experiment ran).
Prior in-house evidence (single category, clean + 2 shifts, raw-vs-clipped
scores) is listed at the bottom as context, not as confirmation.

## QR1 — Does per-channel feature-statistic drift predict accuracy drop?

Hypothesis: across lighting shifts (gamma, brightness, contrast, colour
temperature, directional shading; 3-5 severities) and layers (layer2, layer3),
the Cohen's-d-style shift of per-channel patch-embedding statistics
(||mu_shift - mu_ref||, mean over channels) correlates with AUROC drop with
Spearman rho >= 0.6 per category.

Falsified if median rho across categories/shift families < 0.35 or if any
modifier (B2 variant) breaks the rank ordering implied by measured drift.

## QR2 — AUROC vs F1 at fixed threshold disagree under shift

Hypothesis: at severity >= 2 shifts, measuring AUROC only overstates usable
quality: F1 (image level) at the source-lighting threshold drops by >= 15
percentage points more than AUROC, on at least one of the three categories.

Falsified if median (AUROC drop - F1@source-threshold drop) over
categories x shifts x severities < 5 points.

## QR3 — B3 conformal threshold retains coverage under shift

Hypothesis: with n >= 20 target normals and alpha = 0.1, the B3 threshold keeps
realized per-shift FPR <= alpha + one small-sample quadratic band implied by
Beta(n+1-k,k), without retraining, for every lighting shift family, in >= 2 of
3 categories. A naive source threshold does not satisfy this at severity >= 2.

Falsified if realized FPR exceeds alpha + band in >= 2 of 3 categories.

## QR4 — B2 alignment recovers accuracy with small n

Hypothesis: AdaBN-style alignment from k=10 target normals recovers >= 50% of
the AUROC drop measured between n=0 and n=50 (the "n-curve"), averaged over
categories and moderate-severity shifts; no variant erases discoloured defects
(negative control: defect visible at reference, colour-matched at target, must
stay detectable at >= 80% of reference AU-PRO).

## QR5 — Cost budget

Hypothesis: the full serving path (predict + capacitate-to-n<=50 calibration)
runs within 2x vanilla latency, p95, on the hardware used for the report, with
bank size held at the reproduction config.

## QR6 — External validity on MVTec AD 2

Hypothesis: gains transfer to the MVTec AD 2 can/vial/sheet_metal public test
splits; at least the sign of (proposed - vanilla) image-level SegF1 /
AUROC delta is preserved for moderate shifts, using calibration normals drawn
from validation split only.

## Prior in-house evidence (context only, 2026-11-12 measurements)

- Raw (unclamped) anomalies scores are required: clamped post-processed scores
  collapse the score distribution used by B2/B3. Seed the raw-score forward-hook
  pattern from `apps/vision-inspection/raw_score_sweep.py` of smartex-lab-p0.
- The sujet's B3 wording ("k-th largest") is backwards; correct reading is
  k-th smallest, else n=50, alpha=0.1 rejects ~90% of good parts. Fix
  documented, tests to be written here.
- Pilot drift-vs-AUROC table exists for one category and two shifts; did not
  yet implement the QR1 correlation proper (no severity ladder, no per-layer).
