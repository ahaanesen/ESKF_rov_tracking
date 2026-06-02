from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracking_and_navigation.plotting import (
    PlotterESKFJoint,
)
from tracking_and_navigation.states import (
    JointNominalState,
    JointIdx,
)


# =============================================================================
# Result container
# =============================================================================

@dataclass
class ExperimentResult:
    estimator: str
    experiment: str
    scenario: str
    vehicle: str = "ROV"

    sigma_a: float | None = None
    tdma_interval: float | None = None
    init_range_scale: float | None = None

    run_idx: int = 0

    rmse: float = np.nan
    ate: float = np.nan
    final_error: float = np.nan

    mean_nees: float = np.nan

    converged: bool = True


# =============================================================================
# Core extraction
# =============================================================================

def _results_from_statistics(
    plotter: PlotterESKFJoint,
    *,
    experiment: str,
    scenario: str,
    estimator: str,
    sigma_a: float | None,
    tdma_interval: float | None,
    init_range_scale: float | None,
    run_idx: int,
    divergence_threshold: float,
) -> list[ExperimentResult]:
    rows = plotter.collect_statistics_rows()
    results = []
    for row in rows:
        vehicle = str(row.get("platform", "ROV"))
        rmse = float(row.get("pos_ate_rms", np.nan))
        final_error = float(row.get("pos_final_error", np.nan))
        mean_nees = float(row.get("mean_tanees_pos", np.nan))
        converged = bool(
            np.isfinite(final_error)
            and final_error < divergence_threshold
        )

        results.append(
            ExperimentResult(
                estimator=estimator,
                experiment=experiment,
                scenario=scenario,
                vehicle=vehicle,
                sigma_a=sigma_a,
                tdma_interval=tdma_interval,
                init_range_scale=init_range_scale,
                run_idx=run_idx,
                rmse=rmse,
                ate=rmse,
                final_error=final_error,
                mean_nees=mean_nees,
                converged=converged,
            )
        )
    return results


def extract_eskf_result(
    plotter: PlotterESKFJoint,
    *,
    experiment: str,
    scenario: str,
    estimator: str = "ESKF",
    sigma_a: float = None,
    tdma_interval: float = None,
    init_range_scale: float = None,
    run_idx: int = 0,
    divergence_threshold: float = 10.0,
) -> list[ExperimentResult]:
    """
    Extract standardized experiment metrics from a completed ESKF run.
    """

    stats_results = _results_from_statistics(
        plotter,
        experiment=experiment,
        scenario=scenario,
        estimator=estimator,
        sigma_a=sigma_a,
        tdma_interval=tdma_interval,
        init_range_scale=init_range_scale,
        run_idx=run_idx,
        divergence_threshold=divergence_threshold,
    )
    if stats_results:
        return stats_results

    times = np.asarray(plotter.x_upds.times)

    gt_pos = np.stack([
        plotter.rov_gt.at_time(t).pos
        for t in times
    ])

    est_pos = plotter._rov_est_pos(plotter.x_upds)

    err = est_pos - gt_pos
    dist = np.linalg.norm(err, axis=1)

    rmse = float(np.sqrt(np.mean(dist**2)))
    ate = rmse

    final_error = float(dist[-1])

    converged = final_error < divergence_threshold

    # -------------------------------------------------------------------------
    # NEES
    # -------------------------------------------------------------------------

    nees_vals = []

    for t, x in plotter.x_upds.items():

        gt = JointNominalState(
            asv=plotter.asv_gt.at_time(t),
            rov=plotter.rov_gt.at_time(t),
        )

        err_gauss = x.get_err_gauss(gt)

        err_vec = np.asarray(err_gauss.mean)
        P = err_gauss.cov

        try:
            e = err_vec[JointIdx.ROV_POS]

            Pp = P[
                JointIdx.ROV_POS,
                JointIdx.ROV_POS
            ]

            nees = float(
                e @ np.linalg.solve(Pp, e)
            )

            nees_vals.append(nees)

        except np.linalg.LinAlgError:
            pass

    mean_nees = (
        float(np.mean(nees_vals))
        if nees_vals
        else np.nan
    )

    return [ExperimentResult(
        estimator=estimator,
        experiment=experiment,
        scenario=scenario,
        vehicle="ROV",

        sigma_a=sigma_a,
        tdma_interval=tdma_interval,
        init_range_scale=init_range_scale,

        run_idx=run_idx,

        rmse=rmse,
        ate=ate,
        final_error=final_error,

        mean_nees=mean_nees,

        converged=converged,
    )]


# =============================================================================
# Dataframe helpers
# =============================================================================

def results_to_dataframe(
    results: list[ExperimentResult]
) -> pd.DataFrame:

    return pd.DataFrame([
        asdict(r)
        for r in results
    ])


def save_results_csv(
    df: pd.DataFrame,
    filename: str,
):
    path = Path(filename)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(path, index=False)


# =============================================================================
# Experiment summaries
# =============================================================================

def summarize_sigma_experiment(
    df: pd.DataFrame
) -> pd.DataFrame:
    group_cols = ["sigma_a"]
    if "vehicle" in df.columns:
        group_cols.append("vehicle")

    out = (
        df.groupby(group_cols)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),

            nees_mean=("mean_nees", "mean"),
            nees_std=("mean_nees", "std"),

            convergence_rate=("converged", "mean"),
        )
        .reset_index()
        .sort_values("sigma_a")
    )

    return out


def summarize_initialization_experiment(
    df: pd.DataFrame
) -> pd.DataFrame:
    group_cols = ["init_range_scale"]
    if "vehicle" in df.columns:
        group_cols.append("vehicle")

    out = (
        df.groupby(group_cols)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),

            final_error_mean=("final_error", "mean"),
            final_error_std=("final_error", "std"),

            convergence_rate=("converged", "mean"),

            nees_mean=("mean_nees", "mean"),
        )
        .reset_index()
        .sort_values("init_range_scale")
    )

    return out


def summarize_tdma_experiment(
    df: pd.DataFrame
) -> pd.DataFrame:
    group_cols = ["tdma_interval", "scenario"]
    if "vehicle" in df.columns:
        group_cols.append("vehicle")

    out = (
        df.groupby(group_cols)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),

            nees_mean=("mean_nees", "mean"),

            convergence_rate=("converged", "mean"),
        )
        .reset_index()
        .sort_values("tdma_interval")
    )

    return out


# =============================================================================
# Plotting utilities
# =============================================================================

def _prepare_figure():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    return fig, ax


# =============================================================================
# Sigma tuning plots
# =============================================================================

def plot_sigma_rmse(
    df: pd.DataFrame,
):

    s = summarize_sigma_experiment(df)

    fig, ax = _prepare_figure()

    if "vehicle" in s.columns:
        for vehicle in s["vehicle"].unique():
            sub = s[s["vehicle"] == vehicle]
            x = sub["sigma_a"].values
            y = sub["rmse_mean"].values
            ystd = sub["rmse_std"].fillna(0).values

            ax.plot(x, y, marker="o", label=vehicle)

            ax.fill_between(
                x,
                y - ystd,
                y + ystd,
                alpha=0.2,
            )
        ax.legend()
    else:
        x = s["sigma_a"].values
        y = s["rmse_mean"].values
        ystd = s["rmse_std"].fillna(0).values

        ax.plot(x, y, marker="o")

        ax.fill_between(
            x,
            y - ystd,
            y + ystd,
            alpha=0.2,
        )

    ax.set_xscale("log")

    ax.set_xlabel(r"$\sigma_a$")
    ax.set_ylabel("ATE (RMSE) [m]")

    ax.set_title("RMSE vs Process Noise")

    return fig


def plot_sigma_nees(
    df: pd.DataFrame,
):

    s = summarize_sigma_experiment(df)

    fig, ax = _prepare_figure()

    if "vehicle" in s.columns:
        for vehicle in s["vehicle"].unique():
            sub = s[s["vehicle"] == vehicle]
            x = sub["sigma_a"].values
            y = sub["nees_mean"].values

            ax.plot(
                x,
                y,
                marker="o",
                label=f"{vehicle} mean NEES",
            )
    else:
        x = s["sigma_a"].values
        y = s["nees_mean"].values

        ax.plot(
            x,
            y,
            marker="o",
            label="Mean NEES",
        )

    ax.axhline(
        3.0,
        linestyle="--",
        color="k",
        label="Expected NEES",
    )

    ax.set_xscale("log")

    ax.set_xlabel(r"$\sigma_a$")
    ax.set_ylabel("Mean NEES")

    ax.set_title("NEES vs Process Noise")

    ax.legend()

    return fig


# =============================================================================
# Initialization robustness plots
# =============================================================================

def plot_init_convergence(
    df: pd.DataFrame,
):

    s = summarize_initialization_experiment(df)

    fig, ax = _prepare_figure()

    if "vehicle" in s.columns:
        for vehicle in s["vehicle"].unique():
            sub = s[s["vehicle"] == vehicle]
            ax.plot(
                sub["init_range_scale"].values,
                sub["convergence_rate"].values,
                marker="o",
                label=vehicle,
            )
        ax.legend()
    else:
        x = s["init_range_scale"].values
        y = s["convergence_rate"].values

        ax.plot(
            x,
            y,
            marker="o",
        )

    ax.set_xscale("log")

    ax.set_ylim([0, 1.05])

    ax.set_xlabel(
        r"Initialization scaling $\alpha$"
    )

    ax.set_ylabel(
        "Convergence Probability"
    )

    ax.set_title(
        "Convergence vs Initialization Error"
    )

    return fig


def plot_init_final_error(
    df: pd.DataFrame,
):

    s = summarize_initialization_experiment(df)

    fig, ax = _prepare_figure()

    if "vehicle" in s.columns:
        for vehicle in s["vehicle"].unique():
            sub = s[s["vehicle"] == vehicle]
            x = sub["init_range_scale"].values
            y = sub["final_error_mean"].values
            ystd = sub["final_error_std"].fillna(0).values

            ax.plot(
                x,
                y,
                marker="o",
                label=vehicle,
            )

            ax.fill_between(
                x,
                y - ystd,
                y + ystd,
                alpha=0.2,
            )
        ax.legend()
    else:
        x = s["init_range_scale"].values

        y = s["final_error_mean"].values

        ystd = s[
            "final_error_std"
        ].fillna(0).values

        ax.plot(
            x,
            y,
            marker="o",
        )

        ax.fill_between(
            x,
            y - ystd,
            y + ystd,
            alpha=0.2,
        )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"Initialization scaling $\alpha$"
    )

    ax.set_ylabel(
        "Final Position Error [m]"
    )

    ax.set_title(
        "Final Error vs Initialization Error"
    )

    return fig


# =============================================================================
# TDMA plots
# =============================================================================

def plot_tdma_rmse(
    df: pd.DataFrame,
):

    s = summarize_tdma_experiment(df)

    fig, ax = _prepare_figure()

    scenarios = s["scenario"].unique()
    vehicles = s["vehicle"].unique() if "vehicle" in s.columns else [None]

    for scenario in scenarios:
        for vehicle in vehicles:
            sub = s[s["scenario"] == scenario]
            label = scenario
            if vehicle is not None:
                sub = sub[sub["vehicle"] == vehicle]
                label = f"{scenario} ({vehicle})"

            ax.plot(
                sub["tdma_interval"],
                sub["rmse_mean"],
                marker="o",
                label=label,
            )

    ax.set_xlabel(
        "TDMA Interval [s]"
    )

    ax.set_ylabel(
        "RMSE [m]"
    )

    ax.set_title(
        "RMSE vs TDMA Interval"
    )

    ax.legend()

    return fig


def plot_tdma_divergence(
    df: pd.DataFrame,
):

    s = summarize_tdma_experiment(df)

    fig, ax = _prepare_figure()

    scenarios = s["scenario"].unique()
    vehicles = s["vehicle"].unique() if "vehicle" in s.columns else [None]

    for scenario in scenarios:
        for vehicle in vehicles:
            sub = s[s["scenario"] == scenario]
            label = scenario
            if vehicle is not None:
                sub = sub[sub["vehicle"] == vehicle]
                label = f"{scenario} ({vehicle})"

            divergence_rate = (
                1.0
                - sub["convergence_rate"]
            )

            ax.plot(
                sub["tdma_interval"],
                divergence_rate,
                marker="o",
                label=label,
            )

    ax.set_ylim([0, 1.05])

    ax.set_xlabel(
        "TDMA Interval [s]"
    )

    ax.set_ylabel(
        "Divergence Probability"
    )

    ax.set_title(
        "Divergence vs TDMA Interval"
    )

    ax.legend()

    return fig


def plot_tdma_nees(
    df: pd.DataFrame,
):

    s = summarize_tdma_experiment(df)

    fig, ax = _prepare_figure()

    scenarios = s["scenario"].unique()
    vehicles = s["vehicle"].unique() if "vehicle" in s.columns else [None]

    for scenario in scenarios:
        for vehicle in vehicles:
            sub = s[s["scenario"] == scenario]
            label = scenario
            if vehicle is not None:
                sub = sub[sub["vehicle"] == vehicle]
                label = f"{scenario} ({vehicle})"

            ax.plot(
                sub["tdma_interval"],
                sub["nees_mean"],
                marker="o",
                label=label,
            )

    ax.axhline(
        3.0,
        linestyle="--",
        color="k",
        label="Expected NEES",
    )

    ax.set_xlabel(
        "TDMA Interval [s]"
    )

    ax.set_ylabel(
        "Mean NEES"
    )

    ax.set_title(
        "NEES vs TDMA Interval"
    )

    ax.legend()

    return fig


# =============================================================================
# Convenience batch figure generation
# =============================================================================

def generate_sigma_figures(
    df: pd.DataFrame,
):

    figs = {
        "sigma_rmse":
            plot_sigma_rmse(df),

        "sigma_nees":
            plot_sigma_nees(df),
    }

    return figs


def generate_initialization_figures(
    df: pd.DataFrame,
):

    figs = {
        "init_convergence":
            plot_init_convergence(df),

        "init_final_error":
            plot_init_final_error(df),
    }

    return figs


def generate_tdma_figures(
    df: pd.DataFrame,
):

    figs = {
        "tdma_rmse":
            plot_tdma_rmse(df),

        "tdma_divergence":
            plot_tdma_divergence(df),

        "tdma_nees":
            plot_tdma_nees(df),
    }

    return figs


# =============================================================================
# Save figures
# =============================================================================

def save_figures(
    figures: dict,
    directory: str,
    dpi: int = 200,
):

    outdir = Path(directory)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, fig in figures.items():

        fig.savefig(
            outdir / f"{name}.png",
            dpi=dpi,
            bbox_inches="tight",
        )