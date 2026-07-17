# SXH CoMIR-Style Patch Feature Training Plan

## 1. Goal

Train a new CT-DRR/MRCP common feature extractor inspired by CoMIR, without
modifying the existing `train_feature_network_sxh.py` experiment.

The new experiment changes the current design in three important ways:

```text
1. use one network per modality, not one shared network;
2. output the common representation directly, with no MLP head;
3. compute contrastive loss on 32 x 32 feature patches, not single points.
```

This stage still uses only synthetic paired data rendered from SXH CT/MRCP.
It does not train PoseNet, does not use real X-ray, and does not add DNS,
style augmentation, simulated bile contrast, or guidewire simulation.

## 2. Files

Keep the old shared-network code and outputs untouched.

New code should be added under:

```text
diffpose/ours/xmr/feature_network_comir/
diffpose/ours/xmr/case/sxh/train_feature_network_sxh_comir_patch.py
diffpose/ours/xmr/case/sxh/visualize_feature_checkpoint_sxh_comir_patch.py
```

New training outputs should be saved under:

```text
diffpose/ours/xmr/case/sxh/runs/feature_common_comir_patch/
```

Do not write into:

```text
diffpose/ours/xmr/case/sxh/runs/feature_common/
```

That directory belongs to the previous shared-network point-loss experiment.

## 3. Data Generation

Reuse the existing SXH online renderer:

```text
diffpose/ours/xmr/case/sxh/feature_training_renderer.py
```

The renderer already does the required case-specific work:

```text
center pose: automatic CT-Xray xray031 pose
pose perturbation: get_random_offset distribution
CT projection: CT-DRR
MRCP projection: max projection
normalization: resize 256, per-image min-max, inversion, FOV mask, legacy mean/std
debug overlay: bile projection only for QA images
```

The first CoMIR patch version should keep CT-DRR and MRCP rendered at the same
pose, matching the current renderer behavior. This keeps the first code change
focused on the network/loss design. A later version should add independent
known 2D transforms for the two modalities to remove the same-coordinate
shortcut.

Input tensors:

```text
ct_drr:          [B, 1, 256, 256]
mrcp_projection:[B, 1, 256, 256]
valid_mask:      [B, 1, 256, 256]
```

Default batch size remains:

```text
B = 8
```

## 4. Network

Use two independent full-resolution U-Net style networks:

```text
ct_drr          -> xray_net -> xray_comir: [B, 32, 256, 256]
mrcp_projection -> mrcp_net -> mrcp_comir: [B, 32, 256, 256]
```

The two networks should have the same architecture but independent weights.

```text
xray_net parameters != mrcp_net parameters
```

This follows the CoMIR idea of one neural network per modality. The networks
are allowed to learn modality-specific front-end processing, while the loss
forces their outputs to live in a comparable common representation space.

Recommended first architecture:

```text
input channels: 1
output channels: 32
spatial size: 256 x 256 in, 256 x 256 out
encoder widths: 32, 64, 128, 256
decoder widths: 128, 64, 32
normalization: GroupNorm
activation: LeakyReLU
final layer: 1 x 1 convolution to 32 channels
final normalization: L2 normalize across channels
```

No MLP projection head is used. The final `32 x 256 x 256` output is the
CoMIR-style representation.

## 5. Patch Definition

Use patches on the output feature maps, not raw image patches.

Patch size:

```text
P = 32
```

For a sampled patch center `p = (y, x)`, extract:

```text
xray_patch[p]: [32, 32, 32]
mrcp_patch[p]: [32, 32, 32]
```

Here the dimensions mean:

```text
[feature_channels, patch_height, patch_width]
```

Patch centers must be sampled only where the full patch lies inside the valid
circular FOV:

```text
margin = P // 2 = 16
```

For the first version, also require all pixels in the 32 x 32 patch to be
inside `valid_mask`. This avoids using padded/FOV-border content as a shortcut.

## 6. Patch Similarity

Do not add an MLP. Compare the two output patches directly.

Flatten each patch:

```text
[32, 32, 32] -> [32768]
```

Then compute cosine similarity:

```text
sim(a, b) = dot(normalize(a), normalize(b))
```

Equivalent implementation detail:

```text
1. extract patches with unfold;
2. flatten channel and spatial patch dimensions;
3. L2 normalize each flattened patch descriptor;
4. compute descriptor dot products.
```

This is the direct CoMIR-style comparison: the network output itself is the
representation, and the patch from that representation is the descriptor.

## 7. Contrastive Loss

Use symmetric InfoNCE.

For one sampled xray patch descriptor `X_i`, the positive key is the MRCP
patch descriptor from the same image and same patch center:

```text
positive: M_i
```

All other sampled MRCP patches in the batch are negatives:

```text
negatives: M_j, j != i
```

Directional loss:

```text
L_xray_to_mrcp =
  CE((X @ M.T) / tau, target=diagonal_index)
```

Reverse direction:

```text
L_mrcp_to_xray =
  CE((M @ X.T) / tau, target=diagonal_index)
```

Total:

```text
L_total = 0.5 * (L_xray_to_mrcp + L_mrcp_to_xray)
```

Initial temperature:

```text
tau = 0.07
```

Sample count:

```text
patches_per_image = 32
total patches per step = B * 32 = 256
similarity matrix per direction = [256, 256]
```

This keeps memory reasonable even though each patch descriptor has 32768
elements.

Local negative exclusion:

```text
exclude same-image negatives whose patch centers are within 32 pixels
Chebyshev distance of the query center
```

Reason: neighbouring 32 x 32 patches overlap heavily and may contain nearly
the same anatomy, so treating them as hard negatives is harmful.

## 8. Training Configuration

Start with the same practical training settings as the existing SXH feature
run:

```text
steps:                  100000
batch_size:             8
render_chunk_size:      2
patches_per_image:      32
patch_size:             32
feature_channels:       32
optimizer:              AdamW
learning_rate:          1e-4
weight_decay:           1e-4
warmup_steps:           500
min_learning_rate:      1e-6
gradient_clip:          5.0
amp_dtype:              bf16 by default
checkpoint_interval:    1000
log_interval:           10
```

Use CUDA in the real environment for training. The agent sandbox may not have
GPU access; GPU runs should be requested with escalation and kept inside
`/data/zsr/project`.

## 9. Checkpoints And Logs

Each checkpoint interval should save both:

```text
last.pt
step_XXXXXX.pt
```

Do not keep only the final or last checkpoint. The intermediate `step_*.pt`
files are needed for checking whether feature quality improves and then
degrades.

Each checkpoint should save:

```text
step
xray_net_state_dict
mrcp_net_state_dict
optimizer_state_dict
scheduler_state_dict
scaler_state_dict
training config
renderer metadata
```

Logs should be JSONL and include:

```text
step
loss
xray_to_mrcp
mrcp_to_xray
gradient_norm
gradient_finite
skipped_step
learning_rate
ct_render_seconds
mrcp_render_seconds
network_seconds
gpu_memory_mb
contrast_multiplier
patch_size
patches_per_image
feature_channels
temperature
local_negative_exclusion
amp_dtype
```

Save first-batch QA images:

```text
ct_drr.png
mrcp_max.png
ct_drr_bile_overlay.png
```

## 10. Smoke Test

Before a long run, implement a smoke mode that runs 2 to 3 steps and checks:

```text
ct_drr shape == [B, 1, 256, 256]
mrcp shape == [B, 1, 256, 256]
xray_comir shape == [B, 32, 256, 256]
mrcp_comir shape == [B, 32, 256, 256]
patch descriptor shape == [B * patches_per_image, 32768]
loss is finite
gradient norm is finite and positive
at least one parameter changes after optimizer.step()
checkpoint can be written
```

## 11. First Visualization

After the first checkpoint, visualize:

```text
1. CT-DRR image
2. MRCP max projection
3. xray feature PCA RGB
4. mrcp feature PCA RGB
5. several 32 x 32 patch anchors
6. xray-to-mrcp patch similarity heatmaps
7. mrcp-to-xray patch similarity heatmaps
```

The expected early diagnostic is not just low loss. A useful feature should
produce similarity peaks at corresponding anatomical structures. If heatmaps
always peak at identical coordinates even after future independent 2D
transforms, the network is still using a coordinate shortcut.

## 12. Later Required Fix

This plan intentionally keeps the first implementation close to the current
renderer. However, same-pose same-coordinate positives can still permit a
coordinate shortcut, even with two independent networks and patch loss.

The next version should add known independent 2D transforms:

```text
xray image  -> T_x
mrcp image  -> T_m
positive pair: xray patch center T_x(p), mrcp patch center T_m(p)
```

That change should be implemented only after the new CoMIR patch code runs
stably and produces checkpoints/visualizations.
