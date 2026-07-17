# SXH Common Feature Network: First Runnable Training Plan

## 1. Scope

This is the minimum first training stage. Train one shared network that maps a
virtual CT X-ray (CT-DRR) and an MRCP maximum-intensity projection rendered at
the same pose to directly comparable dense feature maps.

```text
CT volume   -- pose T --> CT-DRR (virtual X-ray) -- shared U-Net --> DNS --> D_xray
MRCP volume -- pose T --> MRCP max projection   -- shared U-Net --> DNS --> D_mrcp
```

The first runnable version includes only the two cross-modal contrastive loss
directions:

```text
CT-DRR feature -> MRCP feature
MRCP feature   -> CT-DRR feature
```

It does not include style augmentation, self-augmentation loss, real ERCP
X-rays, bile-contrast simulation, guidewire simulation, wrong-pose losses,
validation experiments, ablations, PoseNet, or registration optimization.

## 2. Fixed SXH Data Contract

Use only the refined SXH geometry already verified by the xray031 viewer:

- CT renderer: `IntubationDatasetMR` and `DRR`.
- MRCP renderer: `DRRMRCP` with
  `mrcp_501_registered_to_ct_refined.nii.gz`.
- MRCP projection mode: `max`.
- Bile-duct segmentation:
  `mr_bile_duct_registered_to_ct_refined.nii.gz`.
- Center pose: the automatic CT-Xray pose for xray031 used by the refined
  visualization. Never use a manual pose.

The refined MRCP volume is already mapped into the CT DRR coordinate system.
Do not apply the legacy ICP transform a second time.

For every pair, CT and MRCP must use the exact same `RigidTransform` object,
detector geometry, DRR spacing, crop, factors, axis convention,
`reverse_x_axis`, and output resolution.

## 3. Online Virtual Sample Generation

### 3.1 Centered random pose

Let `T_center` be the trusted global xray031 CT-Xray pose and let `C` be the
existing SXH `center_pose`. A sampled pose is:

```text
T = T_center @ inverse(C) @ Delta @ C
```

This preserves the old PoseNet convention of applying local perturbations
about the specimen center while guaranteeing that `Delta = identity` returns
exactly to `T_center`.

Sample `Delta` using the active distribution in
`ours.case.my_util2.get_random_offset`:

```text
Euler ZYX rotation ~ Normal(0, [pi/8, pi/10, pi/12])
translation mm    ~ Normal(0, [30, 50, 30])
```

The distribution is not clipped. Large tail samples are retained unless the
MRCP projection has no usable pixels in the circular field of view.

### 3.2 One batch

For one batch of `B=8` sampled poses:

1. Render CT-DRR with the sampled pose batch.
2. Render MRCP `max` projection with the same pose batch.
3. Render the refined bile-duct segmentation with the same pose batch.
4. Sample one CT bone attenuation multiplier for the full batch from
   `Uniform(0.5, 8.0)`, matching the earlier CT PoseNet data generation.
5. Keep the bile-duct projection only for saved geometry overlays and debug
   images. It is not a network input and is not used to choose positives.

The term `xray` below always means this virtual CT-DRR, not a real ERCP image.

### 3.3 Exact image normalization

Apply the normalization from `train_sxh_addspdeco.py` to CT-DRR and MRCP in
the same way:

```text
resize to 256 x 256
per-image min-max normalization
intensity inversion
circular field of view, radius 119 pixels
Normalize(mean=0.3080, std=0.1494)
```

The min/max operation is independent for every rendered image, even when the
renderer returns a batch. It is implemented by
`ours.xmr.feature_network.PerImageLegacyTransform`; all remaining operations
match the legacy SXH training transform. The output values after the final
normalization are allowed to be negative and must not be clipped.

No image augmentation is applied in this version.

## 4. Network

### 4.1 Full-resolution shared U-Net

The input and output spatial sizes are identical:

```text
input image:       B x 1  x 256 x 256
U-Net feature h:   B x 64 x 256 x 256
DNS feature:       B x 12 x 256 x 256
final feature D:   B x 32 x 256 x 256
```

Use one shared-weight 2D U-Net for CT-DRR and MRCP. Do not create modality
specific encoders or stems.

The U-Net has four encoder levels with channel widths 32, 64, 128, and 256.
Each level uses two 3x3 convolutions, GroupNorm with eight groups, and
LeakyReLU with slope 0.1. Downsampling uses BlurPool plus stride-2 convolution;
upsampling uses bilinear interpolation plus convolution. Skip connections
restore the feature map to 256 x 256 before the DNS block.

GroupNorm is used because the batch is small and CT-DRR/MRCP intensity
statistics are different.

### 4.2 Full-resolution DNS and final descriptor

DNS is computed from the full-resolution U-Net feature `h`, not from a
downsampled feature map:

1. use the direct four-neighbour layout (up, down, left, right);
2. use the same four-neighbour layout with dilation 3;
3. compute the six pairwise feature distances in each layout;
4. convert the distances into self-similarities;
5. concatenate the two layouts into 12 DNS channels.

Use reflection padding at the image boundary. A feature-squeezing head maps
DNS to the final descriptor:

```text
12 DNS channels -> 1x1 conv (32) -> LeakyReLU
                -> 3x3 conv (32) -> LeakyReLU
                -> 3x3 conv (32) -> L2 normalization across channels
```

`D_xray` and `D_mrcp` are the only tensors compared by the loss.

## 5. Symmetric Cross-Modal Loss

The feature maps remain 256 x 256. To keep the InfoNCE similarity matrix
bounded, randomly sample 256 valid pixels per image from the radius-119
circular field of view. Sampling is for the loss only; the network still
outputs full-resolution feature maps.

For a sampled pixel coordinate `p`, `D_xray[p]` and `D_mrcp[p]` are the
positive pair because they were rendered at the same pose and detector pixel.

Let all sampled MRCP descriptors in the batch be the candidate set for a CT
query, and vice versa. With temperature `tau = 0.07`:

```text
L_xray_to_mrcp = InfoNCE(query=D_xray, keys=D_mrcp)
L_mrcp_to_xray = InfoNCE(query=D_mrcp, keys=D_xray)
L_total = 0.5 * (L_xray_to_mrcp + L_mrcp_to_xray)
```

The two directions are both required. There is no original-to-augmented loss,
no `L_self`, and no wrong-pose loss in the first runnable version.

All non-corresponding sampled pixels from the batch are negatives. To avoid
forcing nearly identical local pixels apart, same-image pixels within a
Chebyshev distance of two from `p` are removed from that query's negative set.

The bile-duct mask is not used in this loss. With no simulated contrast in the
CT-DRR, forcing bile-duct pixels to dominate the positives would assume a CT
signal that may not be visible.

## 6. Training Configuration

Use the following initial configuration:

```text
resolution:                 256 x 256
render batch size:          8
descriptor samples/image:   256
optimizer:                  AdamW
learning rate:              1e-4
betas:                      (0.9, 0.999)
weight decay:               1e-4
mixed precision:            enabled
gradient clipping:          5.0
warmup:                     500 steps
schedule:                   cosine decay to 1e-6
target training length:     100,000 optimizer steps
```

Do not use gradient accumulation initially. It does not increase the number of
same-step contrastive negatives and makes the first training flow harder to
debug.

At every training step, log `L_total`, both directional losses, gradient norm,
GPU memory, CT render time, MRCP render time, and network time. Save the last
checkpoint every 1,000 steps. No validation run is required in this phase.

## 7. First Runnable Deliverable

The immediate objective is only an end-to-end training run. Implement in this
order:

1. a paired online renderer that returns normalized CT-DRR, normalized MRCP,
   bile-duct overlay for debugging, `T`, and `Delta`;
2. the full-resolution shared U-Net, DNS block, and squeezing head;
3. the two directional InfoNCE losses with full-resolution coordinate sampling;
4. a trainer that renders, normalizes, forwards both images, backpropagates,
   updates weights, logs losses, and saves checkpoints;
5. a small smoke run that confirms finite images, feature shapes
   `B x 32 x 256 x 256`, finite losses, nonzero gradients, and one successful
   optimizer update;
6. the requested 100,000-step training run after the smoke run succeeds.

The smoke run is an execution check, not a validation experiment. Do not add
feature retrieval, pose landscapes, real-X-ray evaluation, sum/max ablations,
or PoseNet until the base training flow is stable.

## 8. Files and Outputs

Keep all implementation and run outputs under `ours/xmr/case/sxh/`:

- paired render dataset and normalization helpers;
- full-resolution DNS U-Net;
- training entry point;
- `runs/feature_common/` checkpoints and JSON configuration;
- a small set of saved CT-DRR / MRCP / bile-overlay triptychs for debugging.

Every checkpoint must record the xray031 center pose, random seed, pose
distribution, CT attenuation multiplier, renderer configuration, and input
paths so that the same synthetic batch can be regenerated.
