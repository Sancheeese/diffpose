# SXH Feature Pose Optimization Plan

## Goal

Use a frozen cross-modal CoMIR feature network to optimize the 2D/3D pose of the refined MRCP projection. This mirrors the centerline/guidewire optimization pipeline, but replaces the geometric centerline-to-guidewire loss with a dense feature-patch similarity loss.

## Two Registration Tests

1. Virtual CT-DRR to MRCP projection
   - Fixed image: CT DRR rendered at the trusted xray031 pose.
   - Moving image: MRCP projection rendered at the optimized pose.
   - Purpose: closed-loop geometry sanity check. This should work before using real X-ray.

2. Real X-ray to MRCP projection
   - Fixed image: real xray031 image from the SXH dataset.
   - Moving image: MRCP projection rendered at the optimized pose.
   - Purpose: final target setting with real image appearance, devices, noise, and domain gap.

## Initialization

Use the same refined SXH setup as the centerline/guidewire visualization and optimization:

- Build `SXHFeatureTrainingRenderer(projection_mode="sum", device="cuda:0")` for the first differentiable optimizer test.
- The trained feature network used max-projection MRCP images, so `sum` is first a gradient-path sanity check. If it works, save max-projection renderings later for visual comparison.
- Use `renderer.trusted_pose` as the default center pose.
- Optimize local 6-DoF offsets around `renderer.trusted_pose` with `compose_centered_perturbations`.
- The optimized pose is:

```text
pose = trusted_pose.compose(center_pose.inverse()).compose(offset).compose(center_pose)
```

This keeps the pose parameterization consistent with the synthetic feature training data.

## Frozen Feature Network

Load a CoMIR v2 checkpoint:

```text
runs/feature_common_comir_antishortcut/checkpoints/last.pt
```

The network is frozen:

- `CoMIRTwoBranchFeatureNetwork(feature_channels=32)`
- X-ray branch extracts fixed CT-DRR or real X-ray features.
- MRCP branch extracts moving MRCP projection features.
- Output shape: `[1, 32, 256, 256]`.

The MRCP-inverted experiment uses the same optimizer script with `--mrcp-invert` and an inverted-MRCP checkpoint.

## Image Pipeline

All images must follow the training-time feature pipeline:

1. Render or load image.
2. `PerImageLegacyTransform(size=256, radius=119)`.
3. `CanonicalSquareCrop(image_size=256, radius=119, output_size=256)`.
4. Optional MRCP intensity inversion for the inverted checkpoint.
5. Feed `[1, 1, 256, 256]` tensors into the frozen feature network.

The largest inscribed square crop removes the circular FOV boundary before feature extraction.

## Loss

First implementation uses a direct corresponding-patch feature loss:

```text
loss_patch = 1 - cosine(flatten(xray_feature_patch), flatten(mrcp_feature_patch))
loss = mean(loss_patch over sampled patch centers)
```

Default patch settings:

- patch size: 32
- patch centers per step: 32
- valid center margin: 16 pixels
- ROI mode: regular grid over the central image area

This optimization loss intentionally starts without negatives. Negatives are useful for feature training, but for pose optimization they can inject unstable gradients. If virtual CT-DRR registration works, a later version can add local heatmap or InfoNCE-style negative terms.

## Optimizer

Default:

- optimizer: Adam
- steps: 300
- learning rate: 0.03 for rotation offsets, 2.0 for translation offsets
- gradient clipping: 10.0
- AMP: disabled by default for the differentiable renderer path

Pose variables:

- `rotation_delta`: `[1, 3]`, Euler ZYX radians
- `translation_delta`: `[1, 3]`, millimeters

## Outputs

Each run writes to:

```text
runs/feature_pose_optimization/<mode>_xray031_<timestamp>/
```

Saved files:

- `config.json`
- `history.jsonl`
- `initial_pose.npy`
- `final_pose.npy`
- `pose_history.npz`
- `initial_fixed.png`
- `initial_mrcp.png`
- `final_mrcp.png`
- `initial_overlay.png`
- `final_overlay.png`
- `loss_curve.png`
- `feature_pca_initial.png`
- `feature_pca_final.png`
- `summary.json`

## Success Criteria

Virtual mode:

- Loss decreases clearly.
- Final MRCP projection visually aligns better with the trusted CT-DRR.
- Pose delta approaches zero if initialized by a synthetic perturbation.

Real mode:

- Loss decrease is useful but not sufficient.
- Visual overlay must be checked against the X-ray anatomy and guidewire/bile-duct region.
- If virtual mode succeeds but real mode fails, the bottleneck is likely real-image domain gap rather than the differentiable pose optimization chain.

## First Command

```bash
conda run --no-capture-output -n mr2ct \
  python diffpose/ours/xmr/case/sxh/optimize_feature_pose_sxh.py \
  --device cuda:0 \
  --mode virtual \
  --projection sum \
  --steps 300
```

Then run real X-ray:

```bash
conda run --no-capture-output -n mr2ct \
  python diffpose/ours/xmr/case/sxh/optimize_feature_pose_sxh.py \
  --device cuda:0 \
  --mode real \
  --projection sum \
  --steps 300
```
