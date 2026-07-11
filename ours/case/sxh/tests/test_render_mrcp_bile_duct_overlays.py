from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "diffpose" / "ours" / "case" / "sxh"))

from render_mrcp_bile_duct_overlays import discover_result_csvs, parse_final_pose_row  # noqa: E402


def test_parse_final_pose_row_uses_primary_pose_columns(tmp_path):
    csv_path = tmp_path / "sxh_xray042_se3_log_map.csv"
    pd.DataFrame(
        [
            {
                "alpha": 1.0,
                "beta": 2.0,
                "gamma": 3.0,
                "bx": 4.0,
                "by": 5.0,
                "bz": 6.0,
                "alpha2": 10.0,
                "beta2": 20.0,
                "gamma2": 30.0,
                "bx2": 40.0,
                "by2": 50.0,
                "bz2": 60.0,
                "idx": 42,
            },
            {
                "alpha": -1.0,
                "beta": -2.0,
                "gamma": -3.0,
                "bx": -4.0,
                "by": -5.0,
                "bz": -6.0,
                "alpha2": 0.0,
                "beta2": 0.0,
                "gamma2": 0.0,
                "bx2": 0.0,
                "by2": 0.0,
                "bz2": 0.0,
                "idx": 42,
            },
        ]
    ).to_csv(csv_path, index=False)

    pose = parse_final_pose_row(csv_path)

    assert pose.idx == 42
    assert pose.params == [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]


def test_discover_result_csvs_ignores_summary_and_cma(tmp_path):
    runs_dir = tmp_path / "runs" / "mask"
    cma_dir = runs_dir / "cma"
    cma_dir.mkdir(parents=True)
    (runs_dir / "sxh_xray003_se3_log_map.csv").write_text("alpha,beta,gamma,bx,by,bz,idx\n0,0,0,0,0,0,3\n")
    (runs_dir / "similarity_results.csv").write_text("x\n1\n")
    (cma_dir / "sxh_xray004_se3_log_map.csv").write_text("alpha,beta,gamma,bx,by,bz,idx\n0,0,0,0,0,0,4\n")

    files = discover_result_csvs(runs_dir)

    assert files == [runs_dir / "sxh_xray003_se3_log_map.csv"]
