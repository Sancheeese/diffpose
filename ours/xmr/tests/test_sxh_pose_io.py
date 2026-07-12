"""Tests for SXH registered pose CSV loading."""

from __future__ import annotations

from pathlib import Path

from xmr.case.sxh.pose_io import (
    default_registered_index,
    discover_registered_indices,
    load_registered_pose,
    parse_final_pose_row,
    registered_pose_csv_path,
)


def test_parse_final_pose_row_uses_last_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "sxh_xray006_se3_log_map.csv"
    csv_path.write_text(
        "alpha,beta,gamma,bx,by,bz,idx\n"
        "0.1,0.2,0.3,10,20,30,6\n"
        "-1.0,-0.1,1.8,245,38,141,6\n"
    )

    pose = parse_final_pose_row(csv_path)

    assert pose.index == 6
    assert pose.params == [-1.0, -0.1, 1.8, 245.0, 38.0, 141.0]


def test_load_registered_pose_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_registered_pose(99, tmp_path) is None


def test_load_registered_pose_reads_existing_csv(tmp_path: Path) -> None:
    csv_path = registered_pose_csv_path(42, tmp_path)
    csv_path.write_text(
        "alpha,beta,gamma,bx,by,bz,idx\n"
        "0.5,0.6,0.7,1,2,3,42\n"
    )

    pose = load_registered_pose(42, tmp_path)

    assert pose == [0.5, 0.6, 0.7, 1.0, 2.0, 3.0]


def test_discover_registered_indices_sorted(tmp_path: Path) -> None:
    (tmp_path / "sxh_xray010_se3_log_map.csv").write_text("alpha,beta,gamma,bx,by,bz\n0,0,0,0,0,0\n")
    (tmp_path / "sxh_xray003_se3_log_map.csv").write_text("alpha,beta,gamma,bx,by,bz\n0,0,0,0,0,0\n")

    assert discover_registered_indices(tmp_path) == [3, 10]
    assert default_registered_index(tmp_path) == 3
