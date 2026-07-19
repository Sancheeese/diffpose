# SXH CoMIR Pose Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible visual and numeric comparison of the completed SXH CoMIR checkpoint for same-pose and cross-pose virtual CT-DRR/MRCP pairs.

**Architecture:** Add a standalone case-level evaluator that reuses `SXHFeatureTrainingRenderer`, the checkpoint's v2 two-branch network, and the training normalizer/canonical crop. It renders trusted pose A and a seeded centred perturbation B, evaluates A/A, B/B, and A/B without random image augmentation, then writes PNG reports and a JSON summary.

**Tech Stack:** Python, PyTorch CUDA, DiffPose `RigidTransform`, Matplotlib, NumPy, pytest.

---

### Task 1: Define CPU-only metric and pose-pair helpers

**Files:**
- Create: `ours/xmr/case/sxh/pose_feature_comparison.py`
- Test: `ours/xmr/case/sxh/tests/test_pose_feature_comparison.py`

- [ ] **Step 1: Write the failing metric test**

```python
def test_similarity_summary_separates_matched_from_permuted_features():
    xray = torch.eye(4).reshape(1, 4, 2, 2)
    mrcp = xray.clone()
    result = summarize_similarity(xray, mrcp, seed=7)
    assert result["corresponding_mean"] == pytest.approx(1.0)
    assert result["corresponding_mean"] > result["permuted_mean"]
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run -n mr2ct pytest ours/xmr/case/sxh/tests/test_pose_feature_comparison.py -q`

Expected: FAIL because `pose_feature_comparison` and `summarize_similarity` do not exist.

- [ ] **Step 3: Implement the minimal pure helper**

```python
def summarize_similarity(xray_features, mrcp_features, *, seed):
    xray = F.normalize(xray_features.flatten(2), dim=1)
    mrcp = F.normalize(mrcp_features.flatten(2), dim=1)
    corresponding = (xray * mrcp).sum(dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(corresponding.shape[-1], generator=generator)
    permuted = (xray * mrcp[..., permutation]).sum(dim=1)
    return {"corresponding_mean": corresponding.mean().item(), "permuted_mean": permuted.mean().item()}
```

- [ ] **Step 4: Verify the test passes**

Run: `conda run -n mr2ct pytest ours/xmr/case/sxh/tests/test_pose_feature_comparison.py -q`

Expected: PASS.

### Task 2: Render and evaluate the three defined pose pairs

**Files:**
- Modify: `ours/xmr/case/sxh/pose_feature_comparison.py`
- Test: `ours/xmr/case/sxh/tests/test_pose_feature_comparison.py`

- [ ] **Step 1: Write the failing pair-order test**

```python
def test_pair_specifications_keep_the_cross_pose_pair_explicit():
    specs = pair_specs("A", "B")
    assert specs == [("A/A", "A", "A"), ("B/B", "B", "B"), ("A/B", "A", "B")]
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run -n mr2ct pytest ours/xmr/case/sxh/tests/test_pose_feature_comparison.py -q`

Expected: FAIL because `pair_specs` does not exist.

- [ ] **Step 3: Implement deterministic renderer evaluation**

```python
torch.manual_seed(args.seed)
renderer = SXHFeatureTrainingRenderer(projection_mode="max", render_chunk_size=1, device=args.device)
offset_b = get_random_offset(1, renderer.device)
pose_b = compose_centered_perturbations(renderer.trusted_pose, renderer.specimen_center_pose, offset_b)
render_a = renderer.render_poses(renderer.trusted_pose, contrast_multiplier=args.contrast_multiplier, include_bile_projection=False)
render_b = renderer.render_poses(pose_b, contrast_multiplier=args.contrast_multiplier, include_bile_projection=False)
```

Load `last.pt`, construct `CoMIRTwoBranchFeatureNetwork`, apply the existing canonical square crop, and forward CT/MRCP tensors in inference mode. Reject a missing checkpoint, non-finite features, or CPU renderer. Preserve the global `RigidTransform` values in the report.

- [ ] **Step 4: Verify CPU helper tests remain green**

Run: `conda run -n mr2ct pytest ours/xmr/case/sxh/tests/test_pose_feature_comparison.py -q`

Expected: PASS.

### Task 3: Produce reproducible reports

**Files:**
- Modify: `ours/xmr/case/sxh/pose_feature_comparison.py`

- [ ] **Step 1: Implement the report writers**

```python
def save_reports(output_dir, pair_results, summary):
    # Joint-PCA basis is fit across every pair's two feature maps.
    # Write pose_feature_comparison.png, similarity_maps.png, and summary.json.
```

The main report has A/A, B/B, and A/B rows with CT-DRR, MRCP, CT feature PCA, and MRCP feature PCA. The similarity report has corresponding and permuted similarity maps plus a compact distribution panel. `summary.json` must label A/B as a deliberately mismatched comparison, not a registration score.

- [ ] **Step 2: Run the evaluator on CUDA 1**

Run: `python -m xmr.case.sxh.pose_feature_comparison --device cuda:1 --checkpoint xmr/case/sxh/runs/feature_common_comir_antishortcut/checkpoints/last.pt --output-dir xmr/case/sxh/runs/feature_common_comir_antishortcut/pose_comparison`

Expected: `pose_feature_comparison.png`, `similarity_maps.png`, and `summary.json` exist, with finite metrics for all three pairs.

- [ ] **Step 3: Inspect artifacts and run focused regression tests**

Run: `conda run -n mr2ct pytest ours/xmr/feature_network_comir_v2/test_augment_and_loss.py ours/xmr/case/sxh/tests/test_pose_feature_comparison.py -q`

Expected: PASS.

- [ ] **Step 4: Commit source, test, and plan**

```bash
git add docs/superpowers/plans/2026-07-18-sxh-comir-pose-comparison.md \
  ours/xmr/case/sxh/pose_feature_comparison.py \
  ours/xmr/case/sxh/tests/test_pose_feature_comparison.py
git commit -m "feat: compare SXH CoMIR features across poses"
```
