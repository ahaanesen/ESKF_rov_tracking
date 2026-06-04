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

TRAJECTORY_TYPE = TrajectoryType.FIGURE_8
    
exp1_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp1,bearing-only,ROV,0.005,,,,0,3.6152570827897037,3.6152570827897037,6.636447625834557,10.798121248266488,True
ESKF,exp1,bearing-only,ASV,0.005,,,,0,1.3309371073809362,1.3309371073809362,1.7303414606729304,3.168930079976048,True
ESKF,exp1,bearing-only,ROV,0.01,,,,0,2.7311454630535303,2.7311454630535303,0.6863889678463387,3.858437250786805,True
ESKF,exp1,bearing-only,ASV,0.01,,,,0,1.2637820996380167,1.2637820996380167,0.8290037881461129,2.765508977870145,True
ESKF,exp1,bearing-only,ROV,0.02,,,,0,3.559782015170971,3.559782015170971,0.6381261191899648,2.9654426019734577,True
ESKF,exp1,bearing-only,ASV,0.02,,,,0,1.332716962607181,1.332716962607181,0.9964197113307957,2.8466987357494657,True
ESKF,exp1,bearing-only,ROV,0.05,,,,0,4.796425989764966,4.796425989764966,4.496005329375376,2.9024089824738812,True
ESKF,exp1,bearing-only,ASV,0.05,,,,0,1.3950270002302119,1.3950270002302119,0.6916671299949243,2.9223925401983113,True
ESKF,exp1,bearing-only,ROV,0.1,,,,0,10.687787063290875,10.687787063290875,22.413201058580086,3.5368416448257403,False
ESKF,exp1,bearing-only,ASV,0.1,,,,0,1.4900516125993069,1.4900516125993069,1.1441865078359394,3.3815733317684753,True
ESKF,exp1,bearing-only,ROV,0.5,,,,0,266.6766659419769,266.6766659419769,535.4705433500873,130.21809179423659,False
ESKF,exp1,bearing-only,ASV,0.5,,,,0,1.2457231855348736,1.2457231855348736,1.5573442853875938,2.169301359913301,True
ESKF,exp1,bearing-only,ROV,1.0,,,,0,519.1510109383509,519.1510109383509,1114.712437618321,123.90957967409928,False
ESKF,exp1,bearing-only,ASV,1.0,,,,0,1.3369557649572605,1.3369557649572605,0.9224767956657481,2.615474606033414,True
"""

exp1_fgo_results = """ 
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.005,5.0,GT init,3.3867476356657074,3.3867476356657074,2.877077269256992,3.5372424337763553,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.005,5.0,GT init,1.6975756609980517,1.6975756609980517,1.1944402382894526,3.094857602223015,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.01,5.0,GT init,3.0137199022156644,3.0137199022156644,2.391592788330604,3.3981498971313204,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.01,5.0,GT init,1.692630818597605,1.692630818597605,1.3750437035954384,3.0924283879640933,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,GT init,2.916862534964018,2.916862534964018,3.3253016044601313,3.3529219792426983,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,GT init,1.7017996822887955,1.7017996822887955,1.2908739033708043,3.0872786090007716,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.05,5.0,GT init,2.7052708457495083,2.7052708457495083,6.4911267537170705,3.1458610158810107,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.05,5.0,GT init,1.724630490501401,1.724630490501401,1.2713366680368232,3.0795385632351167,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.1,5.0,GT init,5.898152050526624,5.898152050526624,10.294719979586558,3.454098326891233,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.1,5.0,GT init,1.738171829439968,1.738171829439968,1.3130692360249077,3.067349790303165,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.5,5.0,GT init,38.658262882654014,38.658262882654014,19.438081273486,3.1176644718189417,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.5,5.0,GT init,1.7142698718699554,1.7142698718699554,0.8814343724252003,3.059073107722251,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,1.0,5.0,GT init,25.538982540726685,25.538982540726685,29.34993696984155,4.003193665268199,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,1.0,5.0,GT init,1.7499419097446305,1.7499419097446305,1.7172751351989557,3.06675163945781,True
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


# def _plot_metric(ax, rows: list[dict], vehicle: str, metric: str, ylabel: str) -> None:
#     marker_map = {"ESKF": "o", "FGO": "s"}
#     for estimator in ESTIMATOR_ORDER:
#         for scenario in SCENARIO_LABELS:
#             series = [r for r in rows if r["vehicle"] == vehicle and r["estimator"] == estimator and r["scenario"] == scenario]
#             if not series:
#                 continue
#             series = sorted(series, key=lambda r: r["sigma_a"])
#             sigmas = [r["sigma_a"] for r in series]
#             values = [r[metric] for r in series]
#             label = f"{estimator} {scenario}"
#             ax.plot(sigmas, values, marker=marker_map.get(estimator, "o"), label=label)

#     # ax.set_xlabel("sigma_a")
#     ax.set_xlabel(r"$\sigma_a$")
#     ax.set_ylabel(ylabel)
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_title(vehicle)
#     ax.grid(True, linestyle="--", alpha=0.4)
def _plot_metric(ax, rows: list[dict], vehicle: str, metric: str, ylabel: str) -> None:
    marker_map = {"ESKF": "o", "FGO": "s"}

    # Map display label -> dataset label
    vehicle_map = {
        "UUV": "ROV",
        "USV": "ASV",
    }

    dataset_vehicle = vehicle_map.get(vehicle, vehicle)

    for estimator in ESTIMATOR_ORDER:
        for scenario in SCENARIO_LABELS:
            series = [
                r for r in rows
                if r["vehicle"] == dataset_vehicle
                and r["estimator"] == estimator
                and r["scenario"] == scenario
            ]
            if not series:
                continue

            series = sorted(series, key=lambda r: r["sigma_a"])
            sigmas = [r["sigma_a"] for r in series]
            values = [r[metric] for r in series]
            label = f"{estimator} {scenario}"

            ax.plot(sigmas, values,
                    marker=marker_map.get(estimator, "o"),
                    label=label)

    ax.set_xlabel(r"$\sigma_a$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 👇 This now shows UUV / USV
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

    fig_ate, axes_ate = plt.subplots(2, 1, figsize=(7.5, 7.5), sharey=True)
    _plot_metric(axes_ate[0], rows, "UUV", "ate", "ATE (RMSE)")
    _plot_metric(axes_ate[1], rows, "USV", "ate", "ATE (RMSE)")
    axes_ate[1].legend(loc="upper right", fontsize=8)
    fig_ate.suptitle(rf"{gt_traj}: ATE (RMSE) vs $\sigma_a$", fontsize=14)
    fig_ate.tight_layout(rect=[0, 0, 1, 0.94])

    fig_nees, axes_nees = plt.subplots(2, 1, figsize=(7.5, 7.5), sharey=True)
    lower, upper = _chi2_95_interval_3dof()
    for ax in axes_nees:
        ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\,\chi^2_{0.95,3}$")
    _plot_metric(axes_nees[0], rows, "UUV", "mean_nees", "NEES")
    _plot_metric(axes_nees[1], rows, "USV", "mean_nees", "NEES")
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

