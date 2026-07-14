import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = WORKSPACE.parent / "experiment" / "pose_experienment"
OUT_DIR = Path(__file__).resolve().parent


DISPLAY_NAMES = {
    "experiment_full_degree_freedom_20260708_150826": "Full 6-DoF",
    "experiment_lock_x_rotation_20260708_152314": "Lock X rot.",
    "experiment_lock_x_translation_20260708_151438": "Lock X trans.",
    "experiment_lock_y_rotation_20260708_152609": "Lock Y rot.",
    "experiment_lock_y_translation_20260708_151656": "Lock Y trans.",
    "experiment_lock_z_rotation_20260708_152904": "Lock Z rot.",
    "experiment_lock_z_translation_20260708_152057": "Lock Z trans.",
}

SORT_ORDER = [
    "experiment_full_degree_freedom_20260708_150826",
    "experiment_lock_x_translation_20260708_151438",
    "experiment_lock_y_translation_20260708_151656",
    "experiment_lock_z_translation_20260708_152057",
    "experiment_lock_x_rotation_20260708_152314",
    "experiment_lock_y_rotation_20260708_152609",
    "experiment_lock_z_rotation_20260708_152904",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percentile(values, q):
    if not values:
        return math.nan
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return arr[lo]
    return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)


def summarize_time_diffs(test_pairs_path):
    values = []
    with test_pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                values.append(json.loads(line)["time_diff_s"] * 1000.0)
    return {
        "sync_mean_ms": float(np.mean(values)),
        "sync_p95_ms": percentile(values, 0.95),
        "sync_max_ms": max(values),
    }


def collect_rows():
    rows = []
    for name in SORT_ORDER:
        exp_dir = EXPERIMENT_ROOT / name
        report = load_json(exp_dir / "tag2flange_calibration_report.json")
        split = report["split"]
        test_res = report["test_residuals"]
        train_res = report["train_residuals"]
        sync = summarize_time_diffs(exp_dir / "dataset_split" / "test_pairs.jsonl")
        rows.append(
            {
                "experiment_dir": name,
                "experiment": DISPLAY_NAMES[name],
                "matched_samples": report["matched_samples"],
                "train_samples": split["train_samples"],
                "test_samples": split["test_samples"],
                "test_translation_mean_mm": test_res["translation_m_stats"]["mean"] * 1000.0,
                "test_translation_median_mm": test_res["translation_m_stats"]["median"] * 1000.0,
                "test_translation_p95_mm": test_res["translation_m_stats"]["p95"] * 1000.0,
                "test_rotation_mean_deg": test_res["rotation_deg_stats"]["mean"],
                "test_rotation_median_deg": test_res["rotation_deg_stats"]["median"],
                "test_rotation_p95_deg": test_res["rotation_deg_stats"]["p95"],
                "train_translation_mean_mm": train_res["translation_m_stats"]["mean"] * 1000.0,
                "train_rotation_mean_deg": train_res["rotation_deg_stats"]["mean"],
                "translation_gap_mm": report["evaluation"]["translation_mean_gap_mm"],
                "rotation_gap_deg": report["evaluation"]["rotation_mean_gap_deg"],
                **sync,
            }
        )
    return rows


def write_csv(rows):
    fields = list(rows[0].keys())
    path = OUT_DIR / "rigid_transform_validation_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[field]) for field in fields) + "\n")
    return path


def write_latex_table(rows):
    path = OUT_DIR / "rigid_transform_validation_table.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Rigid transformation validation across full and axis-constrained calibration motions. Translation errors are reported in millimeters and rotation errors in degrees on the held-out test split.}",
        r"\label{tab:rigid_transform_validation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Motion & Matched & Train/Test & Trans. mean & Trans. P95 & Rot. mean & Rot. P95 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {}/{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
                row["experiment"],
                row["matched_samples"],
                row["train_samples"],
                row["test_samples"],
                row["test_translation_mean_mm"],
                row["test_translation_p95_mm"],
                row["test_rotation_mean_deg"],
                row["test_rotation_p95_deg"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_manuscript_text(rows):
    total = sum(row["matched_samples"] for row in rows)
    mean_trans = np.mean([row["test_translation_mean_mm"] for row in rows])
    mean_rot = np.mean([row["test_rotation_mean_deg"] for row in rows])
    full = rows[0]
    mean_sync = np.mean([row["sync_mean_ms"] for row in rows])
    max_sync = max(row["sync_max_ms"] for row in rows)
    path = OUT_DIR / "rigid_transform_validation_manuscript_text.md"
    text = f"""# Rigid Transformation Validation - Manuscript Text

The rigid transformation experiment validates the pose-conversion step that maps marker motion measured by the external camera to the robot end-effector frame. Seven calibration sequences were collected: one full six-degree-of-freedom motion sequence and six axis-constrained sequences that isolated translational or rotational excitation along the robot-frame axes. For each sequence, AprilTag-based visual motion increments were paired with Franka end-effector motion increments according to host timestamps, and the resulting pairs were divided into training and held-out test sets using an 80/20 split. This procedure produced {total} matched visual-robot motion pairs across all sequences.

In the full six-degree-of-freedom sequence, the calibrated transformation achieved a test translational error of {full['test_translation_mean_mm']:.2f} mm and a test rotational error of {full['test_rotation_mean_deg']:.2f} degrees, with P95 errors of {full['test_translation_p95_mm']:.2f} mm and {full['test_rotation_p95_deg']:.2f} degrees, respectively. The corresponding train-test gaps were small ({full['translation_gap_mm']:.3f} mm for translation and {full['rotation_gap_deg']:.3f} degrees for rotation), indicating that the estimated transformation remained consistent on unseen motion pairs rather than fitting only the calibration trajectory.

The axis-constrained sequences provide a complementary check on the same transformation under motions that emphasize individual translation or rotation components. Across the seven sequences, the mean held-out translational error was {mean_trans:.2f} mm and the mean held-out rotational error was {mean_rot:.2f} degrees. The timestamp pairing was also stable: the mean visual-robot time difference was {mean_sync:.2f} ms, and the largest observed test-set difference was {max_sync:.2f} ms, well below the 50 ms pairing threshold. These results indicate that the calibrated marker-to-flange transformation is sufficiently stable to serve as a fixed conversion parameter when hospital-side marker trajectories are migrated into the robot end-effector representation.
"""
    path.write_text(text, encoding="utf-8")
    return path


def make_figure(rows):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    labels = [row["experiment"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.68, wspace=0.42)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(x - width / 2, [row["test_translation_mean_mm"] for row in rows], width, color=blue, label="Trans. mean")
    ax1.scatter(x - width / 2, [row["test_translation_p95_mm"] for row in rows], marker="_", s=260, linewidths=2.5, color="#003F5C", label="Trans. P95")
    ax1.set_ylabel("Translation error (mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=24, ha="right")
    ax1.set_ylim(0, 12)
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)

    ax1b = ax1.twinx()
    ax1b.bar(x + width / 2, [row["test_rotation_mean_deg"] for row in rows], width, color=orange, label="Rot. mean")
    ax1b.scatter(x + width / 2, [row["test_rotation_p95_deg"] for row in rows], marker="_", s=260, linewidths=2.5, color="#8C2D04", label="Rot. P95")
    ax1b.set_ylabel("Rotation error (deg)")
    ax1b.set_ylim(0, 4)
    ax1b.spines["top"].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, ncol=4, columnspacing=1.0, handlelength=1.8)
    ax1.text(-0.08, 1.05, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold")
    ax1.set_title("Held-out rigid transform residuals")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axhline(0, color="#4D4D4D", linewidth=1.0)
    ax2.bar(x - width / 2, [row["translation_gap_mm"] for row in rows], width, color=green, label="Translation gap")
    ax2.bar(x + width / 2, [row["rotation_gap_deg"] for row in rows], width, color=purple, label="Rotation gap")
    ax2.set_ylabel("Test - train gap")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax2.text(-0.16, 1.06, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold")
    ax2.set_title("Generalization gap")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(x, [row["sync_mean_ms"] for row in rows], width=0.55, color="#56B4E9", label="Mean")
    ax3.scatter(x, [row["sync_p95_ms"] for row in rows], marker="_", s=260, linewidths=2.5, color="#0072B2", label="P95")
    ax3.scatter(x, [row["sync_max_ms"] for row in rows], marker="o", s=34, color="#D55E00", label="Max")
    ax3.axhline(50, color="#4D4D4D", linestyle="--", linewidth=1.1, label="Matching threshold")
    ax3.set_ylabel("Time difference (ms)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=35, ha="right")
    ax3.set_ylim(0, 55)
    ax3.legend(frameon=False, loc="upper left")
    ax3.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax3.text(-0.16, 1.06, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold")
    ax3.set_title("Visual-robot timestamp pairing")

    fig.suptitle("Rigid transformation validation for pose conversion", fontsize=14, y=0.99)
    fig.savefig(OUT_DIR / "rigid_transform_validation_results_large_font.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "rigid_transform_validation_results_large_font.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "rigid_transform_validation_results_large_font.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    rows = collect_rows()
    write_csv(rows)
    write_latex_table(rows)
    write_manuscript_text(rows)
    make_figure(rows)


if __name__ == "__main__":
    main()
