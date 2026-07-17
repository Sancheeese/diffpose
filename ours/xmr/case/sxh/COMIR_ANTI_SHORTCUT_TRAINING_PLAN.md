# SXH CoMIR Anti-Shortcut Feature Training Plan

## 1. Goal

Build a new training version from scratch that keeps the CoMIR idea but removes
the same-coordinate shortcut as much as possible.

The current CoMIR patch version proves the code path works:

```text
xray_net: CT-DRR -> [B, 32, 256, 256]
mrcp_net: MRCP   -> [B, 32, 256, 256]
loss: symmetric patch InfoNCE
```

However, training positives are still same-coordinate patches. The feature PCA
looks like a coordinate texture, so the next version should make fixed image
coordinates unreliable.

This plan creates a new experiment. Do not modify the previous training files
or outputs.

## 2. New Files And Outputs

New shared modules:

```text
diffpose/ours/xmr/feature_network_comir_v2/
```

New SXH entry points:

```text
diffpose/ours/xmr/case/sxh/train_feature_network_sxh_comir_antishortcut.py
diffpose/ours/xmr/case/sxh/visualize_feature_checkpoint_sxh_comir_antishortcut.py
```

New output directory:

```text
diffpose/ours/xmr/case/sxh/runs/feature_common_comir_antishortcut/
```

Do not write to:

```text
diffpose/ours/xmr/case/sxh/runs/feature_common/
diffpose/ours/xmr/case/sxh/runs/feature_common_comir_patch/
```

## 3. High-Level Training Idea

Start from a matched synthetic pair:

```text
CT-DRR image I_x
MRCP projection I_m
```

They are rendered with the same 3D pose, so before augmentation a detector
point `p` corresponds to the same detector point `p`.

Then apply independent known 2D transforms:

```text
I_x' = A_x(I_x)
I_m' = A_m(I_m)
```

The positive correspondence is no longer same coordinate. It is:

```text
xray location: A_x(p)
mrcp location: A_m(p)
```

The network sees only the transformed images. The loss knows the transform
matrices and samples corresponding patch centers using this relation.

This is the most important change. It directly breaks:

```text
feature at (y, x) should match feature at (y, x)
```

and replaces it with:

```text
feature at A_x(y, x) should match feature at A_m(y, x)
```

Before these independent transforms, remove the circular detector edge. Crop
the largest center square fully inside the circular FOV, then resize it back
to 256 x 256. The network should never see the original circular boundary.
All later random crop/rotation happens inside this canonical square image.

## 4. Network

Keep the CoMIR-style two-branch network:

```text
xray_net: [B, 1, 256, 256] -> [B, 32, 256, 256]
mrcp_net: [B, 1, 256, 256] -> [B, 32, 256, 256]
```

Architecture:

```text
same U-Net architecture for both branches
independent weights
no MLP
final channel count = 32
final L2 normalization across channels
```

No DNS in this version. The aim is to test whether the CoMIR-style direct
representation can learn content-following dense features once coordinate
shortcuts are made unstable.

## 5. 2D Transform Design

Use transforms that avoid interpolation artifacts as much as possible in the
first version.

### 5.1 Canonical square crop: remove circular FOV edge

The renderer returns a normalized `256 x 256` image with a circular FOV of
radius 119 pixels. This circular edge is an easy shortcut, so remove it before
any network input.

Crop the largest center square fully inside the circle:

```text
side ~= floor(119 * sqrt(2)) = 168
y: 44 ... 211
x: 44 ... 211
```

Then resize this square back to:

```text
256 x 256
```

Apply this same canonical square crop to CT-DRR, MRCP, masks, and debug bile
projection. After this step, the working coordinate system is no longer the
original circular detector image. It is:

```text
canonical square coordinate: 256 x 256
```

### 5.2 Independent crop origins with a shared crop size

From the canonical square image, sample one shared crop size:

```text
crop size: randomly choose from {160, 176, 192, 208, 224}
```

Sample the X-ray and MRCP crop origins independently. Their overlap must be
large enough to contain the required number of complete corresponding feature
patches. The shared size keeps the scale of a matching patch identical, while
independent origins make its output coordinate differ between modalities.

Resize crop back to:

```text
256 x 256
```

Record the crop-to-output affine matrix for each modality.

### 5.3 Independent C4 rotations

For each modality independently choose:

```text
k_xray in {0, 1, 2, 3}
k_mrcp in {0, 1, 2, 3}
```

Apply:

```text
rot90(image, k)
```

Because the image is square, this creates no black triangular padding. This
directly follows CoMIR's C4 rotation strategy.

### 5.4 Optional integer translation inside valid crop

After C4 is working, add small independent integer translations:

```text
dx, dy in [-16, 16]
```

Do not pad with black and do not let shifted patches touch invalid regions.
Implement this as a larger crop followed by an inner valid sampling region, or
use crop-origin changes instead of explicit image padding.

For the first runnable version, canonical square crop + random crop + C4 is
enough. Add translation only after visualization confirms the pipeline is
correct.

## 6. Correspondence Tracking And Patch Sampling

Every sampled feature-patch center `p` is represented in canonical square coordinates:

```text
p = [y, x, 1]
```

For each image in the batch, maintain two affine maps:

```text
A_x: canonical square coordinate -> transformed xray image coordinate
A_m: canonical square coordinate -> transformed mrcp image coordinate
```

The center is only used to locate a patch. The training sample itself is a pair
of aligned feature patches:

```text
p_x = A_x(p)
p_m = A_m(p)

X_i = xray_feature[p_x, 32 x 32]
M_i = mrcp_feature[p_m, 32 x 32]
```

The canonical patch grid is mapped through both transforms and sampled with
bilinear `grid_sample`. This aligns the two output patches to the same
canonical orientation before comparison.

Only keep a patch center `p` if both output patches are valid:

```text
32 x 32 patch around p_x lies fully in xray valid mask
32 x 32 patch around p_m lies fully in mrcp valid mask
```

This validity check must happen after canonical square crop, random crop,
resize, and C4 rotation.

## 7. Patch Loss

Patch size:

```text
P = 32
```

Feature channels:

```text
C = 32
```

Patch descriptor:

```text
[C, P, P] -> [32768]
L2 normalize
```

For each sampled feature-patch pair `(X_i, M_i)`:

```text
query: X_i
positive: M_i
negatives: M_j for j != i
```

Use bidirectional cross-modal InfoNCE only. Same-modality patches are not
negative samples:

```text
L_xray_to_mrcp = CE((X @ M.T) / tau, diag_targets)
L_mrcp_to_xray = CE((M @ X.T) / tau, diag_targets)
L_total = 0.5 * (L_xray_to_mrcp + L_mrcp_to_xray)
```

Temperature:

```text
tau = 0.07
```

Sampling:

```text
render_batch_size = 8
feature_patch_pairs_per_rendered_image = 24
total positive patch pairs per step = 192

For every query patch, within its own rendered X-ray/MRCP pair and either direction:

```text
1 positive patch + 23 cross-modal negative patches
```
```

The 24 sampled patches from one rendered pair are spatially separated enough
not to overlap in canonical-square coordinates. Local-negative exclusion uses
canonical coordinates, not transformed coordinates:

```text
exclude same-image negatives whose canonical-square points are within 32 px Chebyshev distance
```

This avoids treating heavily overlapping anatomy as a negative even when the
two modalities have different C4 rotations.

## 8. Avoiding Border And Padding Shortcuts

This version should be strict:

```text
1. no black padding from rotation;
2. circular FOV boundary removed before network input;
3. no patch touching canonical-square edge or random-crop boundary;
4. no patch touching optional translation boundary;
5. no negative patch sampled from invalid or artificial border regions.
```

The valid mask must be transformed with the same canonical-square/crop/resize/C4
operations as the image. Patch centers are sampled only from the intersection of:

```text
xray transformed valid patch centers
mrcp transformed valid patch centers
known correspondence map validity
```

If not enough valid patch centers exist, resample the crop/rotation for that
batch item.

## 9. Training Configuration

Initial defaults:

```text
steps:                  100000
render_batch_size:      8
render_chunk_size:      2
feature_channels:       32
patch_size:             32
feature_patch_pairs_per_rendered_image: 24
patch_pairs_per_step:   192
negative_patches_per_query: 23
crop_sizes:             160,176,192,208,224
canonical_square_size:  168
c4_rotation:            enabled, independent per modality
translation:            disabled in first runnable version
optimizer:              AdamW
learning_rate:          1e-4
weight_decay:           1e-4
warmup_steps:           500
min_learning_rate:      1e-6
gradient_clip:          5.0
amp_dtype:              bf16
checkpoint_interval:    1000
log_interval:           10
```

Save every checkpoint interval:

```text
checkpoints/last.pt
checkpoints/step_XXXXXX.pt
```

Log the transform parameters for each batch summary:

```text
crop origin
crop size
canonical square origin and size
xray rotation k
mrcp rotation k
number of valid correspondence points
temperature
patch descriptor shape
```

## 10. Smoke Test

Smoke mode should run 2 to 3 optimizer steps and assert:

```text
transformed xray shape == [B, 1, 256, 256]
transformed mrcp shape == [B, 1, 256, 256]
xray feature shape == [B, 32, 256, 256]
mrcp feature shape == [B, 32, 256, 256]
xray patch descriptors == [B, 24, 32768]
mrcp patch descriptors == [B, 24, 32768]
loss is finite
gradients are finite
one optimizer update changes parameters
last.pt and step_000003.pt are written
```

Also save debug images:

```text
raw CT-DRR
raw MRCP
canonical-square CT-DRR
canonical-square MRCP
transformed CT-DRR
transformed MRCP
transformed correspondence overlay
```

The overlay should draw at least five correspondence pairs:

```text
xray transformed point p_x
mrcp transformed point p_m
```

## 11. Required Visualizations

After training or during checkpoints, visualize:

```text
1. raw same-pose CT/MRCP pair
2. canonical-square CT/MRCP pair without circular border
3. transformed xray and transformed mrcp
4. xray CoMIR PCA
5. mrcp CoMIR PCA
6. patch correspondence heatmaps using transformed true positions
```

Add a stress-test visualization:

```text
hold xray transform fixed
change mrcp C4/crop transform
check whether predicted patch follows A_m(p), not the original coordinate
```

This should replace the earlier padding-based shift test because C4/crop does
not introduce artificial black borders.

## 12. Success Criteria

This version is not judged only by low training loss.

A checkpoint is useful only if:

```text
1. transformed correspondence heatmaps peak near A_m(p);
2. predictions do not stay at the same output coordinate when A_m changes;
3. heatmaps work for bile/duct regions, bone/edge regions, and smooth regions;
4. PCA maps change with image content, not just fixed detector coordinates;
5. no peak is dominated by FOV or crop borders.
```

If loss goes near zero but PCA still looks like a fixed coordinate texture,
increase anti-shortcut pressure in this order:

```text
1. more crop-size variation;
2. more crops per rendered pose;
3. enable integer translation without padding;
4. add multi-case data when available;
5. consider AE pretraining like ContraReg.
```

## 13. Why This Should Help

The previous version allows:

```text
xray_net(y, x) ~= mrcp_net(y, x)
```

for all training positives.

The new version requires:

```text
xray_net(A_x(p)) ~= mrcp_net(A_m(p))
```

where `A_x` and `A_m` change independently. A fixed coordinate code is no
longer sufficient. To succeed, the networks must output features that follow
the transformed image patch.
