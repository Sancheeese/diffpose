# SXH CoMIR Pose Comparison Design

## Objective

Evaluate the completed anti-shortcut CoMIR checkpoint in its training domain:
virtual CT-DRR and MRCP maximum-intensity projections rendered by the verified
SXH geometry. The evaluation must distinguish correctly paired same-pose
images from deliberately mismatched poses. It does not evaluate real ERCP
X-rays or clinical registration accuracy.

## Inputs

- Checkpoint: `xmr/case/sxh/runs/feature_common_comir_antishortcut/checkpoints/last.pt`.
- Reference pose A: the renderer's verified `xray031` trusted global pose.
- Perturbed pose B: one deterministic centred perturbation in the renderer's
  documented Euler-ZYX/local-translation convention. The full A and B values
  will be emitted to JSON.
- Rendering mode: CT DRR and MRCP max projection with the same SXH refined
  CT-grid MRCP assets used for training.

## Comparisons

The report contains three pairs:

1. A/A: CT-DRR(A) and MRCP(A), the geometric positive control.
2. B/B: CT-DRR(B) and MRCP(B), an unseen but geometrically correct pose.
3. A/B: CT-DRR(A) and MRCP(B), the deliberate cross-pose negative control.

All images use the existing per-image normalizer and canonical square crop;
no random crop or C4 augmentation is used during this deterministic test.

## Outputs

An independent evaluation script writes one output directory containing:

- `pose_feature_comparison.png`: rows for A/A, B/B, and A/B with CT-DRR,
  MRCP, and joint-PCA feature maps in a common feature basis.
- `similarity_maps.png`: per-pixel cosine-similarity maps plus distributions
  for corresponding and random-shift controls.
- `summary.json`: checkpoint path, device, full poses, rendering settings,
  and mean/median cosine values for each pair.

For A/A and B/B, cosine similarity is measured at identical detector pixels.
For A/B, it is intentionally measured at the same output pixels without a
geometric correspondence claim. Each row also includes a deterministic random
pixel permutation score as a lower-quality control.

## Success Criteria

- The script loads the final checkpoint and all three renders without changing
  training files or clinical input paths.
- A/A and B/B both have finite dense features and similarity statistics.
- A/B is clearly labelled as a cross-pose mismatch; it is not reported as a
  registration metric.
- The generated report preserves sufficient metadata to reproduce the exact
  poses and visualizations.

## Error Handling

The evaluation reuses the renderer's refined-input validation. It fails before
rendering for a missing checkpoint, non-CUDA device, non-finite features, or
missing refined SXH inputs. It never overwrites a checkpoint or training log.
