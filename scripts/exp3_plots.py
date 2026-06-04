
import argparse
import csv
import io
from pathlib import Path

from enum import Enum
import matplotlib.pyplot as plt
from exp1_plots import TrajectoryType  # reuse the same TrajectoryType enum

# class TrajectoryType(str, Enum):
#     """
#     Trajectory mode selector for both USV and ROV.

#     CIRCULAR   - ASV circles a fixed centre; ROV follows piecewise-linear
#                  waypoints.  The CV model matches ROV truth exactly, making
#                  this the most favourable scenario for the ESKF.

#     FIGURE_8   - ASV traces a Lissajous figure-8 (2:1 frequency ratio),
#                  producing frequent heading reversals that improve bearing-
#                  only observability.  ROV makes a slow circular sweep with
#                  sinusoidal depth — smooth but non-CV, introducing mild model
#                  mismatch.
#     LINEAR_TURNS - ASV moves linearly between waypoints, with smooth heading
#                  changes at each waypoint.  ROV makes a slow linear dive.
#                  This is the least favourable scenario for bearing-only, but
#                  is simple and smooth, and still has some heading variation.
#     """
#     CIRCULAR   = "circular"
#     FIGURE_8   = "figure_8"
#     LINEAR_TURNS = "linear_turns"
	
plt.rcParams.update(
	{
		"font.family": "serif",
		"font.serif": ["CMU Serif", "Computer Modern", "cmr10", "DejaVu Serif"],
		"mathtext.fontset": "cm",
	}
)

TRAJECTORY_TYPE = TrajectoryType.FIGURE_8  # set in main()

exp3_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp3,scenario1,ROV,0.02,5.0,1.0,,0,6.956316168719903,6.956316168719903,1.6143105012684074,11.873516761717253,True
ESKF,exp3,scenario1,ASV,0.02,5.0,1.0,,0,1.7405675500488194,1.7405675500488194,1.0067916019250012,4.674398477838349,True
ESKF,exp3,scenario2,ROV,0.02,5.0,1.0,,0,1.6327620172310326,1.6327620172310326,1.2457240829357128,4.31773884866077,True
ESKF,exp3,scenario2,ASV,0.02,5.0,1.0,,0,1.5835670311207808,1.5835670311207808,0.7770410397982845,4.386636310752138,True
ESKF,exp3,scenario3,ROV,0.02,5.0,1.0,,0,1.519803732282129,1.519803732282129,1.333334052320439,4.199533433642186,True
ESKF,exp3,scenario3,ASV,0.02,5.0,1.0,,0,1.5656258865991688,1.5656258865991688,0.8415449533463223,4.4687578103589685,True
ESKF,exp3,scenario1,ROV,0.02,10.0,1.0,,0,4.528901215213702,4.528901215213702,5.603518449206289,4.158885569365465,True
ESKF,exp3,scenario1,ASV,0.02,10.0,1.0,,0,1.5745496676210007,1.5745496676210007,1.105978303970808,3.696954285328185,True
ESKF,exp3,scenario2,ROV,0.02,10.0,1.0,,0,1.9443004404523971,1.9443004404523971,2.1062236787251662,4.186708419729756,True
ESKF,exp3,scenario2,ASV,0.02,10.0,1.0,,0,1.6521634600373731,1.6521634600373731,1.110128628997229,4.21451618744963,True
ESKF,exp3,scenario3,ROV,0.02,10.0,1.0,,0,1.9296176265434264,1.9296176265434264,2.16156381013216,4.337492701794092,True
ESKF,exp3,scenario3,ASV,0.02,10.0,1.0,,0,1.6656575404011118,1.6656575404011118,1.1156414666616603,4.4299521117466645,True
ESKF,exp3,scenario1,ROV,0.02,20.0,1.0,,0,6.029818153114706,6.029818153114706,17.07062649017038,6.861282992404374,False
ESKF,exp3,scenario1,ASV,0.02,20.0,1.0,,0,1.5941021826462651,1.5941021826462651,1.10500342963515,3.6257382525534054,True
ESKF,exp3,scenario2,ROV,0.02,20.0,1.0,,0,2.2009947124985527,2.2009947124985527,1.3775371100329894,2.8294219816243973,True
ESKF,exp3,scenario2,ASV,0.02,20.0,1.0,,0,1.5918557523252617,1.5918557523252617,1.0966831306146623,3.636065898750679,True
ESKF,exp3,scenario3,ROV,0.02,20.0,1.0,,0,2.1276814880948742,2.1276814880948742,1.0646039514652383,2.7144756635289964,True
ESKF,exp3,scenario3,ASV,0.02,20.0,1.0,,0,1.5722637297431257,1.5722637297431257,1.1059844616627688,3.5756191381496896,True
ESKF,exp3,scenario1,ROV,0.02,30.0,1.0,,0,14.326088622710074,14.326088622710074,37.5710660803588,24.922871301976297,False
ESKF,exp3,scenario1,ASV,0.02,30.0,1.0,,0,1.6293079478506154,1.6293079478506154,1.0061090256688014,3.758868296133131,True
ESKF,exp3,scenario2,ROV,0.02,30.0,1.0,,0,2.459220736558618,2.459220736558618,2.5601937383923743,1.98579335428253,True
ESKF,exp3,scenario2,ASV,0.02,30.0,1.0,,0,1.6011979632829878,1.6011979632829878,1.1144245697465456,3.614394192410628,True
ESKF,exp3,scenario3,ROV,0.02,30.0,1.0,,0,2.5267251263180786,2.5267251263180786,3.499028184533351,2.254920464386083,True
ESKF,exp3,scenario3,ASV,0.02,30.0,1.0,,0,1.598109902306981,1.598109902306981,1.1118810102283154,3.618184303146741,True
ESKF,exp3,scenario1,ROV,0.02,60.0,1.0,,0,16.78811522093313,16.78811522093313,37.62831197635309,3.0963521966032794,False
ESKF,exp3,scenario1,ASV,0.02,60.0,1.0,,0,1.6021535910126476,1.6021535910126476,1.0905582310206623,3.6229179469445643,True
ESKF,exp3,scenario2,ROV,0.02,60.0,1.0,,0,6.627611963293504,6.627611963293504,11.379262716382838,3.13498931712105,False
ESKF,exp3,scenario2,ASV,0.02,60.0,1.0,,0,1.6052695455942645,1.6052695455942645,1.0904606577825207,3.6122624896093765,True
ESKF,exp3,scenario3,ROV,0.02,60.0,1.0,,0,6.197534047720446,6.197534047720446,9.827432181262038,2.681143285073392,True
ESKF,exp3,scenario3,ASV,0.02,60.0,1.0,,0,1.5946435215261945,1.5946435215261945,1.0932302145801658,3.5578926629570904,True
ESKF,exp3,scenario1,ROV,0.02,120.0,1.0,,0,27.476177443989062,27.476177443989062,61.723860067499224,4.695515146646203,False
ESKF,exp3,scenario1,ASV,0.02,120.0,1.0,,0,1.6051282042921808,1.6051282042921808,1.0903429641650135,3.6350211354096613,True
ESKF,exp3,scenario2,ROV,0.02,120.0,1.0,,0,17.521461006425163,17.521461006425163,24.94640161561468,5.54856924101044,False
ESKF,exp3,scenario2,ASV,0.02,120.0,1.0,,0,1.609075474446377,1.609075474446377,1.0905742779391105,3.627947673454915,True
ESKF,exp3,scenario3,ROV,0.02,120.0,1.0,,0,19.356488085519626,19.356488085519626,35.14060463970058,17.74451595978637,False
ESKF,exp3,scenario3,ASV,0.02,120.0,1.0,,0,1.6789070780827038,1.6789070780827038,1.0832373423351176,4.142783902171591,True
"""

exp3_fgo_results = """
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,1,4.25091735797207,4.25091735797207,2.396930333093752,3.1047088808966805,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,1,1.630385792730915,1.630385792730915,0.8877298070188966,3.0607949368009617,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,1,1.1849934921234295,1.1849934921234295,0.6597572858700025,3.176014103282331,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,1,1.6802427417107848,1.6802427417107848,0.8784773659126609,3.0577111383451796,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,1,2.8879461947919873,2.8879461947919873,0.6321588457826595,3.1588107116580786,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,1,2.496730934096547,2.496730934096547,0.7643086679784943,3.1061219097229635,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,10.0,1,4.471675935863016,4.471675935863016,3.854476716734457,3.3081207446526797,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,10.0,1,1.7473788006589153,1.7473788006589153,1.601596177529777,3.0786878326645715,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,10.0,1,1.087162416779215,1.087162416779215,2.516155929100459,3.0696061286974508,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,10.0,1,1.7666716221078365,1.7666716221078365,0.8662400427163284,3.0427148766553027,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,10.0,1,1.0821426668248273,1.0821426668248273,2.362498302567685,3.06238530721209,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,10.0,1,1.765452074466623,1.765452074466623,0.8634881403166393,3.040770539639256,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,20.0,1,4.951985706465951,4.951985706465951,6.446362548429906,3.387790420900318,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,20.0,1,1.8053828570017163,1.8053828570017163,1.1581718162327383,3.038604270154805,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,20.0,1,1.1572619710875631,1.1572619710875631,2.475260673705901,3.499646767155419,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,20.0,1,1.8052202476012005,1.8052202476012005,1.152246114335281,3.037447232978424,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,20.0,1,1.0967749727711276,1.0967749727711276,2.4055401621308397,3.1369354293626452,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,20.0,1,1.853668084644724,1.853668084644724,0.7655763371175857,3.0390377131282342,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,30.0,1,1127.3834700468915,1127.3834700468915,3802.7226746755887,3.929099999560406,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,30.0,1,1.792366890591073,1.792366890591073,0.938538008480439,3.0390168110015012,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,30.0,1,1.3352134157041802,1.3352134157041802,1.6370648588073975,3.193036003456416,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,30.0,1,1.812663302660842,1.812663302660842,1.2173162977151066,3.038728522693536,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,30.0,1,1.3259280292694955,1.3259280292694955,1.56431148308901,3.0854179659104264,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,30.0,1,1.809806127613656,1.809806127613656,1.1729183829142005,3.034760580632907,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,60.0,1,16.834856374387222,16.834856374387222,1.2384806031792408,11.229101800903065,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,60.0,1,1.7688079563801675,1.7688079563801675,1.676313813625966,3.113840688231079,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,60.0,1,1.1079713873044017,1.1079713873044017,2.0565156683947974,4.899996452601714,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,60.0,1,1.8229152156702653,1.8229152156702653,1.3346280247978286,3.0389626725871866,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,60.0,1,1.0485035305724255,1.0485035305724255,1.94412221393748,3.7314261058413143,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,60.0,1,1.901614683548134,1.901614683548134,1.6576104762055188,3.0348199679836383,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,120.0,1,19.003648676088208,19.003648676088208,9.14100620410527,1069.7618944657827,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,120.0,1,1.8069189874418528,1.8069189874418528,1.1717382433887986,3.038700939391218,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,120.0,1,3.6318665234184953,3.6318665234184953,1.52819926439427,66.8513102913592,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,120.0,1,1.7670355016710455,1.7670355016710455,1.7257583681852204,3.0708643443636414,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,120.0,1,3.245080768874696,3.245080768874696,1.50976886182969,87.79698277430482,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,120.0,1,1.8078134099008425,1.8078134099008425,1.101754965494839,3.0457927751902334,True
"""

SCENARIO_LABELS = ["B", "B+R", "B+R+D"]
ESTIMATOR_ORDER = ["ESKF", "FGO"]


def _scenario_label(raw_label: str) -> str | None:
	label = raw_label.strip().lower()
	if "scenario1" in label or "scenario 1" in label or "bearing-only" in label or "bearing only" in label:
		return "B"
	if "scenario3" in label or "scenario 3" in label or "depth" in label:
		return "B+R+D"
	if "scenario2" in label or "scenario 2" in label or "bearing + range" in label or "bearing+range" in label:
		return "B+R"
	return None


def _parse_results(text: str, vehicle_field: str) -> list[dict]:
	reader = csv.DictReader(io.StringIO(text.strip()))
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
				"tdma_interval": float(row.get("tdma_interval", "nan")),
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
			series = sorted(series, key=lambda r: r["tdma_interval"])
			intervals = [r["tdma_interval"] for r in series]
			values = [r[metric] for r in series]
			label = f"{estimator} {scenario}"
			ax.plot(intervals, values, marker=marker_map.get(estimator, "o"), label=label)

	ax.set_xlabel("TDMA interval (s)")
	ax.set_ylabel(ylabel)
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_title(vehicle)
	ax.grid(True, linestyle="--", alpha=0.4)


def main() -> None:
	gt_traj = "Circular/Figure 8" if TRAJECTORY_TYPE == TrajectoryType.FIGURE_8 else "Linear/Linear-turns"
	parser = argparse.ArgumentParser(description="Plot exp3 TDMA sweep results.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(f"plots/{TRAJECTORY_TYPE.value}/exp3_tdma"),
		help="Directory to save figures (optional).",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Do not display figures interactively.",
	)
	args = parser.parse_args()

	eskf_rows = _parse_results(exp3_eskf_results, "vehicle")
	fgo_rows = _parse_results(exp3_fgo_results, "platform")
	rows = eskf_rows + fgo_rows

	fig_ate, axes_ate = plt.subplots(2, 1, figsize=(7.5, 7.5), sharey=True)
	_plot_metric(axes_ate[0], rows, "UUV", "ate", "ATE (RMSE)")
	_plot_metric(axes_ate[1], rows, "USV", "ate", "ATE (RMSE)")
	axes_ate[1].legend(loc="upper right", fontsize=8)
	fig_ate.suptitle(rf"{gt_traj}: ATE (RMSE) vs TDMA interval", fontsize=14)
	fig_ate.tight_layout()

	fig_nees, axes_nees = plt.subplots(2, 1, figsize=(7.5, 7.5), sharey=True)
	lower, upper = _chi2_95_interval_3dof()
	for ax in axes_nees:
		ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\chi^2_{0.95,3}$")
	_plot_metric(axes_nees[0], rows, "UUV", "mean_nees", "NEES")
	_plot_metric(axes_nees[1], rows, "USV", "mean_nees", "NEES")
	axes_nees[1].legend(loc="upper right", fontsize=8)
	fig_nees.suptitle(rf"{gt_traj}: NEES vs TDMA interval", fontsize=14)
	fig_nees.tight_layout()

	if args.output_dir is not None:
		args.output_dir.mkdir(parents=True, exist_ok=True)
		fig_ate.savefig(args.output_dir / "exp3_ate_vs_tdma_interval.svg", bbox_inches="tight")
		fig_nees.savefig(args.output_dir / "exp3_nees_vs_tdma_interval.svg", bbox_inches="tight")

	if not args.no_show:
		plt.show()


if __name__ == "__main__":
	main()
