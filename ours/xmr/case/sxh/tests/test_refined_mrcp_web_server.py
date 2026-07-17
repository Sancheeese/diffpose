from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "diffpose"))

from ours.xmr.case.sxh.web_drr_server_nii_sxh_refined_mrcp import (  # noqa: E402
    DEFAULT_CENTERLINE_MASK,
    DEFAULT_GUIDEWIRE_OVERLAY,
    DEFAULT_REFINED_BILE_DUCT,
    DEFAULT_REFINED_CT_TO_MR,
    DEFAULT_START_INDEX,
    load_guidewire_chain_from_overlay,
    parse_args,
    validate_refined_inputs,
)


def test_refined_server_defaults_to_xray031_and_refined_registration_outputs():
    args = parse_args([])

    assert args.start_index == DEFAULT_START_INDEX == 31
    assert args.init_pose == "registered"
    assert args.refined_bile_duct == DEFAULT_REFINED_BILE_DUCT
    assert args.refined_ct_to_mr == DEFAULT_REFINED_CT_TO_MR
    assert args.centerline_mask == DEFAULT_CENTERLINE_MASK
    assert args.guidewire_overlay == DEFAULT_GUIDEWIRE_OVERLAY
    assert args.projection == "max"
    assert args.host == "0.0.0.0"


def test_refined_server_requires_all_refined_inputs(tmp_path):
    args = parse_args([])
    args.refined_bile_duct = tmp_path / "missing.nii.gz"

    with pytest.raises(FileNotFoundError, match="refined bile duct"):
        validate_refined_inputs(args)


def test_trusted_xray031_overlay_contains_an_ordered_guidewire_chain():
    chain = load_guidewire_chain_from_overlay(DEFAULT_GUIDEWIRE_OVERLAY)

    assert len(chain) > 10
    assert chain.shape[1] == 2


def test_refined_server_accepts_sum_mrcp_projection_mode():
    assert parse_args(["--projection", "sum"]).projection == "sum"
