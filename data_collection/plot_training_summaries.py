from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "training_summary_figures"

SUMMARIES = [
    {
        "name": "ACT",
        "path": Path(
            r"C:\Users\zhj80\xwechat_files\wxid_5kihrfy1233y22_a8e2\msg\file\2026-07\summary.json"
        ),
        "outfile": "act_training_curve.png",
    },
    {
        "name": "Diffusion",
        "path": Path(
            r"C:\Users\zhj80\xwechat_files\wxid_5kihrfy1233y22_a8e2\msg\file\2026-07\summary(1).json"
        ),
        "outfile": "diffusion_training_curve.png",
    },
]


def load_history(path: Path) -> tuple[list[int], list[float], list[float], dict]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    history = summary["history"]
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    return epochs, train_loss, val_loss, summary


def plot_summary(name: str, path: Path, outfile: str) -> Path:
    epochs, train_loss, val_loss, summary = load_history(path)
    best_val = summary.get("best_val_loss")
    test_loss = summary.get("test_loss")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Calibri"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.08, hspace=0.08)

    ax.plot(
        epochs,
        train_loss,
        color="#0072B2",
        linewidth=2.0,
        label="Train loss",
    )
    ax.plot(
        epochs,
        val_loss,
        color="#D55E00",
        linewidth=2.0,
        label="Validation loss",
    )

    best_epoch = min(range(len(val_loss)), key=val_loss.__getitem__)
    ax.scatter(
        [epochs[best_epoch]],
        [val_loss[best_epoch]],
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        s=48,
        zorder=5,
        label="Best validation",
    )

    subtitle = f"Best validation loss: {best_val:.6g}" if best_val is not None else ""
    if test_loss is not None:
        subtitle += f" | Test loss: {test_loss:.6g}"

    ax.set_title(f"{name} training curve\n{subtitle}", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(min(epochs), max(epochs))
    ax.margins(x=0.02, y=0.12)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.65)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.5, alpha=0.45)

    # Keep labels outside the data region so they do not cover the lines.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    for item in SUMMARIES:
        out_path = plot_summary(item["name"], item["path"], item["outfile"])
        print(out_path)


if __name__ == "__main__":
    main()
