import os
import glob

import math
import pandas as pd
import torch
import kornia

from matplotlib import pyplot as plt

from ours.utils.drr import DRR
from utils.drr_bone import DRR as DRR_Bone
from diffpose.deepfluoro import DeepFluoroDataset, Transforms
from diffpose.calibration import RigidTransform
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d


def toZeroOne(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# --------------------------------------------------
# Load dataset + DRR
# --------------------------------------------------
def load_model_and_data(id_number, device="cuda:0"):
    specimen = DeepFluoroDataset(
        id_number,
        filename="/home/zsr/project/diffpose/data/ipcai_2020_full_res_data.h5"
    )

    height = 256
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    ).to(device)

    drr_bone = DRR_Bone(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    ).to(device)

    return specimen, drr, drr_bone, height


# --------------------------------------------------
# Process single CSV
# --------------------------------------------------
def process_csv_file(csv_file, specimen, drr, height, device="cuda:0"):
    base_name = os.path.basename(csv_file)
    parts = base_name.split("_")
    id_number = int(parts[1][-3:])

    transforms = Transforms(drr.detector.height)
    criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])

    df = pd.read_csv(csv_file)
    last_row = df.iloc[-1]

    alpha = last_row["alpha"]
    beta  = last_row["beta"]
    gamma = last_row["gamma"]
    bx    = last_row["bx"]
    by    = last_row["by"]
    bz    = last_row["bz"]
    geo_r = last_row['geo_r'] / 510 * (180 / math.pi)
    geo_t = last_row['geo_t']

    x = torch.tensor(
        [alpha, beta, gamma, bx, by, bz],
        dtype=torch.float32, device=device
    )
    r = x[:3].unsqueeze(0)
    t = x[3:].unsqueeze(0)
    pose = RigidTransform(r, t, "euler_angles", "ZYX")

    # DRR
    p = drr(None, None, None, pose=pose)
    p = transforms(p).to(device)

    # GT X-ray
    img, _ = specimen[id_number]
    img = transforms(img).to(device)

    # Similarity
    ncc = criterion(img, p)
    ssim = 1 - 2 * kornia.losses.ssim_loss(
        toZeroOne(p), toZeroOne(img),
        window_size=11, reduction="mean"
    )

    # mPD
    true_fid, pred_fid = specimen.get_2d_fiducials(id_number, pose)
    mpd = torch.norm(true_fid - pred_fid, dim=2).mean()
    mpd *= 0.194
    # mTRE
    # tre = specimen.calc_tre(id_number, pose)
    tre = last_row["fiducial"]

    return {
        "file": csv_file,
        "id_number": id_number,
        "ncc": ncc.item(),
        "ssim": ssim.item(),
        "mtre": tre.item(),
        "mpd": mpd.item(),
        'geo_r': geo_r,
        'geo_t': geo_t,
        "alpha2": alpha,
        "beta2": beta,
        "gamma2": gamma,
        "bx2": bx,
        "by2": by,
        "bz2": bz,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(idx, directory, prefix="specimen", device="cuda:0"):
    pattern = os.path.join(directory, f"{prefix}*se3_log_map.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    print(f"Found {len(csv_files)} CSV files")

    specimen, drr, drr_bone, height = load_model_and_data(idx, device)

    results = []
    count_mtre = 0
    count_mpd = 0

    for csv_file in csv_files:
        result = process_csv_file(csv_file, specimen, drr, height, device)

        if result is not None:
            results.append(result)

            if result["mtre"] < 10:
                count_mtre += 1
            if result["mpd"] < 10:
                count_mpd += 1

            print(
                f"ID {result['id_number']:03d} | "
                f"mTRE={result['mtre']:.2f}, "
                f"mPD={result['mpd']:.2f}"
            )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    if results:
        print("\n=== Statistics ===")
        keys = ["ncc", "ssim", "mtre", "mpd"]

        for key in keys:
            values = [r[key] for r in results]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            print(f"{key.upper():5s}: {mean:.4f} ± {std:.4f}")

        print(f"mTRE_SR: {count_mtre / len(results):.3f}")
        print(f"mPD_SR : {count_mpd / len(results):.3f}")

        # Save CSV
        df = pd.DataFrame(results)
        df = df[
            [
                "id_number", "file",
                "ncc", "ssim",
                "mtre", "mpd",
                "geo_r", "geo_t",
                "alpha2", "beta2", "gamma2",
                "bx2", "by2", "bz2"
            ]
        ]

        # output_file = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/all/{idx}similarity_results.csv.csv"
        output_file = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/all_bi/{idx}similarity_results.csv.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    for i in range(3, 4):
        idx = i
        directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/bipose/{idx}"
        # directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/wsr/{idx}"
        # directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/mask/{idx}"
        # directory_path = f"/home/zsr/project/diffpose/ours/deepfluoro/runs/{idx}/norm"
        device = "cuda:0"
        prefix = "specimen"

        main(idx, directory_path, prefix=prefix, device=device)
