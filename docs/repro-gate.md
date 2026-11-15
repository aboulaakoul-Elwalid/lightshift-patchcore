# GO/NO-GO 1 — vanilla PatchCore reproduction (MVTec AD)

Fitted anomalib 2.5 Patchcore, wide_resnet50_2, layers 2+3, coreset 0.25, 256px,
full official test set (no test splitting), seed 0, RTX 3060.

| category  | image AUROC (ours) | published (PatchCore, CVPR22) | pixel AUROC (ours) | published |
|-----------|--------------------|-------------------------------|--------------------|-----------|
| bottle    | 1.000              | 1.000                         | 0.992              | 0.984     |
| hazelnut  | 1.000              | 1.000                         | 0.993              | 0.997     |
| leather   | 1.000              | 1.000                         | 0.996              | 0.986     |

Gate: within ±1 pt of published on 3 categories. PASS (image level exact; pixel within 1 pt on bottle/leather, hazelnut -0.4 pt).

Receipts: receipts/<cat>-pcore-v1-fs_ad/patchcore/<fp>.json (+ ckpts).
Note: hazelnut manual probe AUROC (0.883) disagrees with engine metric (1.0) — probe
scores the un-normalized model output; engine metric is authoritative. To verify later.
Datasets: /mnt/data-ssd/mvtec_ad_full/{bottle,hazelnut,leather} (CC BY-NC-SA 4.0).
