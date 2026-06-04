#!/usr/bin/env python3
"""
aggregate_packet_loss.py — cross-probability summary for the packet-loss sweep.

Expected layout produced by run_pipeline_packet_loss.sh:

  <sweep_root>/
    p_0.10/
      mc_aggregate_stats.csv   <- written by aggregate_mc_results.py
    p_0.30/
      mc_aggregate_stats.csv
    ...

Outputs written to <output_dir>/:
  packet_loss_summary.csv
  packet_loss_plots/
    rov_ate_rms_vs_loss.png
    rov_mean_error_vs_loss.png
    asv_ate_rms_vs_loss.png
    asv_mean_error_vs_loss.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIOS = {
    1: "S1: Bearing-only",
    2: "S2: Bearing+Range",
    3: "S3: Bearing+Range+Depth",
}
PLATFORMS   = ["ROV", "ASV"]
ESTIMATORS  = ["ESKF", "FGO"]
MARKERS     = {"ESKF": "o", "FGO": "s"}
COLORS      = {"ESKF": "C0", "FGO": "C1"}
LINESTYLES  = {"ESKF": "--", "FGO": "-"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise packet-loss sweep results across probability levels."
    )
    p.add_argument(
        "--sweep-root",
        required=True,
        help="Root directory of the sweep (contains p_0.10/, p_0.30/, …)",
    )
    p.add_argument(
        "--loss-probs",
        required=True,
        nargs="+",
        type=float,
        help="Packet-loss probabilities that were swept, e.g. 0.1 0.3 0.5 0.7 0.9",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the summary CSV and plots into",
    )
    return p.parse_args()


def _p_dir(p: float) -> str:
    return f"p_{p:.2f}"


def load_sweep(sweep_root: Path, loss_probs: list[float]) -> pd.DataFrame:
    """
    Load per-probability aggregate CSVs and concatenate them with a
    loss_prob column added.
    """
    frames: list[pd.DataFrame] = []
    for p in loss_probs:
        csv_path = sweep_root / _p_dir(p) / "mc_aggregate_stats.csv"
        if not csv_path.exists():
            print(f"[WARN] missing {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        df["loss_prob"] = p
        frames.append(df)

    if not frames:
        print("ERROR: no aggregate CSVs found.", file=sys.stderr)
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per (loss_prob, estimator, platform, scenario_id) with
    the key metrics preserved.
    """
    keep_cols = [
        "loss_prob", "estimator", "platform", "scenario_id", "scenario",
        "n_seeds",
        "ate_rms_mean", "ate_rms_std",
        "mean_error_mean", "mean_error_std",
        "final_error_mean", "final_error_std",
        "p95_error_mean",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].sort_values(
        ["loss_prob", "estimator", "platform", "scenario_id"]
    ).reset_index(drop=True)


def plot_metric_vs_loss(
    df: pd.DataFrame,
    platform: str,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    plot_dir: Path,
    loss_probs: list[float],
) -> None:
    """
    One figure per (platform, metric).  Each subplot is one scenario.
    Lines show mean ± 1 std across MC seeds for each estimator.
    """
    scenario_ids = sorted(
        int(s) for s in df["scenario_id"].dropna().unique() if int(s) in SCENARIOS
    )
    if not scenario_ids:
        return

    n_scen = len(scenario_ids)
    fig, axes = plt.subplots(
        1, n_scen, figsize=(5 * n_scen, 4.5), sharey=False
    )
    if n_scen == 1:
        axes = [axes]

    for ax, sid in zip(axes, scenario_ids):
        for est in ESTIMATORS:
            mask = (
                (df["estimator"]   == est)      &
                (df["platform"]    == platform) &
                (df["scenario_id"] == sid)
            )
            sub = df.loc[mask].sort_values("loss_prob")
            if sub.empty:
                continue

            x    = sub["loss_prob"].values
            y    = sub[metric_mean].values
            yerr = sub[metric_std].values if metric_std in sub.columns else None

            ax.errorbar(
                x, y,
                yerr=yerr,
                marker=MARKERS[est],
                color=COLORS[est],
                linestyle=LINESTYLES[est],
                linewidth=1.8,
                markersize=6,
                capsize=4,
                label=est,
            )

        ax.set_title(SCENARIOS[sid], fontsize=10)
        ax.set_xlabel("Packet-loss probability $p$")
        ax.set_xticks(loss_probs)
        ax.set_xticklabels([str(p) for p in loss_probs])
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(ylabel)
    n_seeds_str = (
        str(int(df["n_seeds"].max())) if "n_seeds" in df.columns else "?"
    )
    fig.suptitle(
        f"{platform} — {ylabel} vs packet-loss probability "
        f"(N = {n_seeds_str} MC seeds per level)",
        fontsize=11,
    )
    fig.tight_layout()

    safe_metric = metric_mean.replace("_mean", "").replace("_", "")
    fname = f"{platform.lower()}_{safe_metric}_vs_loss.png"
    out_path = plot_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname}")


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output_dir)
    plot_dir   = output_dir / "packet_loss_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    loss_probs = sorted(args.loss_probs)

    print(f"Loading sweep from: {sweep_root}")
    print(f"Loss levels: {loss_probs}")

    df      = load_sweep(sweep_root, loss_probs)
    summary = make_summary(df)

    csv_path = output_dir / "packet_loss_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[OK] Summary CSV → {csv_path}")

    # ------------------------------------------------------------------
    # Plots: ATE-RMS and mean position error vs loss probability
    # ------------------------------------------------------------------
    metrics = [
        ("ate_rms_mean",     "ate_rms_std",     "ATE-RMS [m]"),
        ("mean_error_mean",  "mean_error_std",  "Mean position error [m]"),
        ("final_error_mean", "final_error_std", "Final position error [m]"),
    ]

    for platform in PLATFORMS:
        for metric_mean, metric_std, ylabel in metrics:
            if metric_mean not in df.columns:
                continue
            plot_metric_vs_loss(
                df=summary,
                platform=platform,
                metric_mean=metric_mean,
                metric_std=metric_std,
                ylabel=ylabel,
                plot_dir=plot_dir,
                loss_probs=loss_probs,
            )

    print(f"[OK] Plots → {plot_dir}")


if __name__ == "__main__":
    main()