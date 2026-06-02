import argparse
import csv
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

exp1_eskf_results = """
    estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,run_idx,rmse,ate,final_error,mean_nees,converged
    ESKF,exp1,bearing-only,ROV,0.005,,,0,4.108090071555802,4.108090071555802,3.141608595557201,14.240611086709698,True
    ESKF,exp1,bearing-only,ASV,0.005,,,0,1.479639433518815,1.479639433518815,1.2585531873364377,3.9350673963535057,True
    ESKF,exp1,bearing-only,ROV,0.01,,,0,2.639284718218277,2.639284718218277,2.6407154020197683,4.26418376595139,True
    ESKF,exp1,bearing-only,ASV,0.01,,,0,1.1923260882445827,1.1923260882445827,1.510040513020021,2.4216402914050716,True
    ESKF,exp1,bearing-only,ROV,0.02,,,0,2.5474647203866305,2.5474647203866305,1.8460604207830082,2.745736094424955,True
    ESKF,exp1,bearing-only,ASV,0.02,,,0,1.4203950680294228,1.4203950680294228,1.0680482632977084,3.4140304104183037,True
    ESKF,exp1,bearing-only,ROV,0.05,,,0,5.675567160498391,5.675567160498391,6.9782581083451145,2.6351698197017326,True
    ESKF,exp1,bearing-only,ASV,0.05,,,0,1.2436129015348765,1.2436129015348765,1.0822002750725754,2.272128537715435,True
    ESKF,exp1,bearing-only,ROV,0.1,,,0,16.386262311327467,16.386262311327467,22.02860260863204,10.420443898311984,False
    ESKF,exp1,bearing-only,ASV,0.1,,,0,1.154504882724549,1.154504882724549,0.84793295769563,1.8749918022391998,True
    ESKF,exp1,bearing-only,ROV,0.5,,,0,621.1800234378202,621.1800234378202,1305.0927421843762,374.4104293809302,False
    ESKF,exp1,bearing-only,ASV,0.5,,,0,1.318203157273905,1.318203157273905,0.46727963575783527,2.449813057636473,True
    ESKF,exp1,bearing-only,ROV,1.0,,,0,411.5578521760685,411.5578521760685,741.3015845076578,99.04007973488008,False
    ESKF,exp1,bearing-only,ASV,1.0,,,0,1.268556224719796,1.268556224719796,1.165658121446783,2.328837980924512,True
"""

exp1_fgo_results = """ 
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.005,5.0,GT init,3.1334222453126337,3.1334222453126337,2.313379226905214,5.309872444576585,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.005,5.0,GT init,0.5289545241407887,0.5289545241407887,0.2798738024412492,3.0020598588709557,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.01,5.0,GT init,2.3647831436261533,2.3647831436261533,1.4176419941207874,4.023750098382871,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.01,5.0,GT init,0.4637536635954723,0.4637536635954723,0.3999052779475783,3.012266967669119,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,GT init,2.088067313147829,2.088067313147829,1.1910839953426011,3.503786286259959,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,GT init,0.437416364970001,0.437416364970001,0.3794120026576633,3.0149881474328835,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.05,5.0,GT init,2.3042656367847276,2.3042656367847276,0.5294015532268908,3.0211118260206726,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.05,5.0,GT init,0.4159304582006954,0.4159304582006954,0.2521754807135593,3.0117731777994345,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.1,5.0,GT init,7.771598904192512,7.771598904192512,4.900738274754367,3.7417468948373287,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.1,5.0,GT init,0.4048454659036315,0.4048454659036315,0.1826426199646599,3.011989453509631,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,0.5,5.0,GT init,16.495441269972556,16.495441269972556,38.01498714083401,4.322090094544777,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,0.5,5.0,GT init,0.3923804153542282,0.3923804153542282,0.1553183010319277,3.017793031199126,True
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ROV,1.0,5.0,GT init,18.57576920062749,18.57576920062749,39.64167512834981,4.15924383977008,False
FGO,exp1_noise_sweep,FGO - Scenario 1: Bearing-only,ASV,1.0,5.0,GT init,0.4010090577996043,0.4010090577996043,0.1607518963556575,3.0115394954393,True
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
    parser = argparse.ArgumentParser(description="Plot exp1 ATE and NEES results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/exp1_cv"),
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
    fig_ate.suptitle(r"CV tuning: ATE (RMSE) vs $\sigma_a$")
    fig_ate.tight_layout(rect=[0, 0, 1, 0.94])

    fig_nees, axes_nees = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    lower, upper = _chi2_95_interval_3dof()
    for ax in axes_nees:
        ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$95\%\,\chi^2_{3}$")
    _plot_metric(axes_nees[0], rows, "ROV", "mean_nees", "NEES")
    _plot_metric(axes_nees[1], rows, "ASV", "mean_nees", "NEES")
    axes_nees[1].legend(loc="upper right", fontsize=8)
    fig_nees.suptitle(r"CV tuning: NEES vs $\sigma_a$")
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

