import argparse
import csv
from enum import Enum
import io
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)

class TrajectoryType(str, Enum):
    """
    Trajectory mode selector for both USV and ROV.

    CIRCULAR   - ASV circles a fixed centre; ROV follows piecewise-linear
                 waypoints.  The CV model matches ROV truth exactly, making
                 this the most favourable scenario for the ESKF.

    FIGURE_8   - ASV traces a Lissajous figure-8 (2:1 frequency ratio),
                 producing frequent heading reversals that improve bearing-
                 only observability.  ROV makes a slow circular sweep with
                 sinusoidal depth — smooth but non-CV, introducing mild model
                 mismatch.
    LINEAR_TURNS - ASV moves linearly between waypoints, with smooth heading
                 changes at each waypoint.  ROV makes a slow linear dive.
                 This is the least favourable scenario for bearing-only, but
                 is simple and smooth, and still has some heading variation.
    """
    CIRCULAR   = "circular"
    FIGURE_8   = "figure_8"
    LINEAR_TURNS = "linear_turns"

TRAJECTORY_TYPE = TrajectoryType.LINEAR_TURNS
    
exp1_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp1,bearing-only,ROV,0.005,,,,0,3.9499063779355295,3.9499063779355295,2.1821256942821243,1.7640490052468927,True
ESKF,exp1,bearing-only,ASV,0.005,,,,0,1.273874340793982,1.273874340793982,1.239217223348542,2.330681330127901,True
ESKF,exp1,bearing-only,ROV,0.01,,,,0,3.223106168722343,3.223106168722343,1.683885406137093,1.3143779110986598,True
ESKF,exp1,bearing-only,ASV,0.01,,,,0,1.3979450006094776,1.3979450006094776,0.9504812138579132,2.7710695732275235,True
ESKF,exp1,bearing-only,ROV,0.02,,,,0,9.193567196153184,9.193567196153184,15.28360062328699,1.6535339367535375,False
ESKF,exp1,bearing-only,ASV,0.02,,,,0,1.3449974234079605,1.3449974234079605,2.404942056094326,2.4779787574765075,True
ESKF,exp1,bearing-only,ROV,0.05,,,,0,21.116167498295614,21.116167498295614,45.46711560985405,1.5990747463762078,False
ESKF,exp1,bearing-only,ASV,0.05,,,,0,1.3368735869405814,1.3368735869405814,0.6219840673766317,2.635653308776685,True
ESKF,exp1,bearing-only,ROV,0.1,,,,0,47.29803918181772,47.29803918181772,131.09903928609918,2.1629577640679494,False
ESKF,exp1,bearing-only,ASV,0.1,,,,0,1.3726462841180067,1.3726462841180067,0.9172763999045199,2.6213799180266366,True
ESKF,exp1,bearing-only,ROV,0.5,,,,0,895.4587503802459,895.4587503802459,1926.2936693966974,20.380234274248743,False
ESKF,exp1,bearing-only,ASV,0.5,,,,0,1.418151799174189,1.418151799174189,1.6444556257550798,2.812477793890941,True
ESKF,exp1,bearing-only,ROV,1.0,,,,0,1689.3033787105123,1689.3033787105123,3616.3344616393742,22.878195031810517,False
ESKF,exp1,bearing-only,ASV,1.0,,,,0,1.4517658011917463,1.4517658011917463,1.9418500058747457,3.1249811160203653,True
"""

exp1_fgo_results = """ 
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.005,5.0,GT init,7.688851813839472,7.688851813839472,3.116441844686936,5.200910223244082,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.005,5.0,GT init,11.810706147682364,11.810706147682364,1.4371889232780273,3.1013623435644777,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.01,5.0,GT init,8.652271568991777,8.652271568991777,3.760665279064407,5.14830056172258,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.01,5.0,GT init,1.7705836783640858,1.7705836783640858,1.588448087991858,3.0982607763321175,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,GT init,8.036619275383616,8.036619275383616,3.55571034960276,4.556133453534396,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,GT init,1.7701848175115966,1.7701848175115966,1.644147465777914,3.095903010053548,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.05,5.0,GT init,104.8188137129204,104.8188137129204,226.5639454508721,3.800394897812817,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.05,5.0,GT init,1.7852472969785582,1.7852472969785582,1.474601634768175,3.102676208640824,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.1,5.0,GT init,104.15775581699448,104.15775581699448,210.82527679960316,4.651716494606948,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.1,5.0,GT init,1.7797589258350277,1.7797589258350277,1.6923307336306264,3.1298988056260244,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.5,5.0,GT init,464.29772279996854,464.29772279996854,174.1501468258165,3.449530287484732,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.5,5.0,GT init,1.7093102261076405,1.7093102261076405,3.049075993018863,3.1329123291200105,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,1.0,5.0,GT init,498.8268069985549,498.8268069985549,252.06496401643133,3.5453874035570263,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,1.0,5.0,GT init,1.750792086983874,1.750792086983874,1.5250726388324054,3.0718470098740185,True
"""

SCENARIO_LABELS = ["B", "B+R", "B+R+D"]
ESTIMATOR_ORDER = ["ESKF", "FGO"]


def _scenario_label(raw_label: str) -> str | None:
    label = raw_label.strip().lower()
    if "scenario 1" in label or "bearing-only" in label or "bearing only" in label:
        return "B"
    if "scenario 2" in label or "bearing+range" in label or "bearing + range" in label or "range" in label:
        return "B+R"
    if "scenario 3" in label or "depth" in label or "dvl" in label:
        return "B+R+D"
    return None


def _clean_multiline_csv(text: str) -> str:
    cleaned = text.strip().replace("\\\n", "\n")
    return cleaned.replace("\\", "")


def _parse_results(text: str, vehicle_field: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(_clean_multiline_csv(text)))
    rows = []
    for row in reader:
        scenario = _scenario_label(row.get("scenario", ""))
        if scenario is None:
            continue
        vehicle = row.get(vehicle_field, "").strip()
        if not vehicle:
            continue
        rows.append(
            {
                "estimator": row.get("estimator", "").strip(),
                "scenario": scenario,
                "vehicle": vehicle,
                "sigma_a": float(row.get("sigma_a", "nan")),
                "ate": float(row.get("ate", "nan")),
                "mean_nees": float(row.get("mean_nees", "nan")),
            }
        )
    return rows


def _chi2_95_interval_3dof() -> tuple[float, float]:
    try:
        from scipy.stats import chi2

        lower = chi2.ppf(0.025, 3)
        upper = chi2.ppf(0.975, 3)
        return lower, upper
    except Exception:
        return 0.072, 3.116


def _plot_metric(ax, rows: list[dict], vehicle: str, metric: str, ylabel: str) -> None:
    marker_map = {"ESKF": "o", "FGO": "s"}
    for estimator in ESTIMATOR_ORDER:
        for scenario in SCENARIO_LABELS:
            series = [r for r in rows if r["vehicle"] == vehicle and r["estimator"] == estimator and r["scenario"] == scenario]
            if not series:
                continue
            series = sorted(series, key=lambda r: r["sigma_a"])
            sigmas = [r["sigma_a"] for r in series]
            values = [r[metric] for r in series]
            label = f"{estimator} {scenario}"
            ax.plot(sigmas, values, marker=marker_map.get(estimator, "o"), label=label)

    # ax.set_xlabel("sigma_a")
    ax.set_xlabel(r"$\sigma_a$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(vehicle)
    ax.grid(True, linestyle="--", alpha=0.4)


def main() -> None:
    gt_traj = "Circular/Figure 8" if TRAJECTORY_TYPE == TrajectoryType.FIGURE_8 else "Linear/Linear-turns"
    
    parser = argparse.ArgumentParser(description="Plot exp1 ATE and NEES results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"plots/{TRAJECTORY_TYPE.value}/exp1_cv"),
        help="Directory to save figures (optional).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively.",
    )
    args = parser.parse_args()

    eskf_rows = _parse_results(exp1_eskf_results, "vehicle")
    fgo_rows = _parse_results(exp1_fgo_results, "platform")
    rows = eskf_rows + fgo_rows

    fig_ate, axes_ate = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    _plot_metric(axes_ate[0], rows, "ROV", "ate", "ATE (RMSE)")
    _plot_metric(axes_ate[1], rows, "ASV", "ate", "ATE (RMSE)")
    axes_ate[1].legend(loc="upper right", fontsize=8)
    fig_ate.suptitle(rf"{gt_traj}: ATE (RMSE) vs $\sigma_a$", fontsize=14)
    fig_ate.tight_layout(rect=[0, 0, 1, 0.94])

    fig_nees, axes_nees = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    lower, upper = _chi2_95_interval_3dof()
    for ax in axes_nees:
        ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\,\chi^2_{0.95,3}$")
    _plot_metric(axes_nees[0], rows, "ROV", "mean_nees", "NEES")
    _plot_metric(axes_nees[1], rows, "ASV", "mean_nees", "NEES")
    axes_nees[1].legend(loc="upper right", fontsize=8)
    fig_nees.suptitle(rf"{gt_traj}: NEES vs $\sigma_a$", fontsize=14)
    fig_nees.tight_layout(rect=[0, 0, 1, 0.94])

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # fig_ate.savefig(args.output_dir / "exp1_ate_vs_sigma_a.png", dpi=300, bbox_inches="tight")
        # fig_nees.savefig(args.output_dir / "exp1_nees_vs_sigma_a.png", dpi=300, bbox_inches="tight")
        fig_ate.savefig(args.output_dir / "exp1_ate_vs_sigma_a.svg", bbox_inches="tight")
        fig_nees.savefig(args.output_dir / "exp1_nees_vs_sigma_a.svg", bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

