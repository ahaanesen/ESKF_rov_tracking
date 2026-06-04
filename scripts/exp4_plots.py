
import argparse
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt

from exp1_plots import TrajectoryType  # reuse the same TrajectoryType enum

plt.rcParams.update(
	{
		"font.family": "serif",
		"font.serif": ["CMU Serif", "Computer Modern", "cmr10", "DejaVu Serif"],
		"mathtext.fontset": "cm",
	}
)

TRAJECTORY_TYPE = TrajectoryType.LINEAR_TURNS  # change to your actual trajectory type if needed 

exp4_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.0,0,12.486924416534297,12.486924416534297,20.502871743389495,2.6578723115988696,False
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.0,0,1.4832188677138682,1.4832188677138682,1.400101194911641,2.755784415828441,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.0,0,1.1948662278538351,1.1948662278538351,1.239450597223261,2.661634487841207,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.0,0,1.1095249096105966,1.1095249096105966,1.097994264321855,2.4079531886523324,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.0,0,1.0436967720243036,1.0436967720243036,1.132732052048328,2.8879903090081767,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.0,0,1.0402958045087687,1.0402958045087687,1.0929764976531489,2.4422663801256213,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.1,0,11.876812053387395,11.876812053387395,26.355259024113334,2.234358419979336,False
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.1,0,1.49809866057372,1.49809866057372,1.4511991970327602,2.8537191784112,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.1,0,1.1440159186216718,1.1440159186216718,0.808293988222121,2.3265295907374095,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.1,0,1.1413778060502244,1.1413778060502244,1.2332102659458977,2.540801518255304,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.1,0,0.9399857311932636,0.9399857311932636,0.8086718864898085,2.2934012267131614,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.1,0,1.0755474892778327,1.0755474892778327,1.2528279813325793,2.623062121180174,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.3,0,11.635722521602327,11.635722521602327,16.081438199444673,2.9284387701070913,False
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.3,0,1.5298770859323565,1.5298770859323565,1.1685307567799084,3.024576681771509,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.3,0,1.4003682428161932,1.4003682428161932,1.082901390169496,3.0033461895470737,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.3,0,1.2030478546379402,1.2030478546379402,1.053105092442258,2.7039411103320456,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.3,0,1.228654245496834,1.228654245496834,0.990485792403272,3.0845256000853842,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.3,0,1.1407858862418425,1.1407858862418425,1.0157501579358623,2.7393512461845786,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.5,0,10.788535800627985,10.788535800627985,15.624843006496098,2.698502855397199,False
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.5,0,1.5307136485614765,1.5307136485614765,1.2746449289048214,2.897231084215616,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.5,0,1.419478487559505,1.419478487559505,1.3370057044763983,2.8956104369898346,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.5,0,1.1817332567327699,1.1817332567327699,1.1378144355957915,2.562535850697453,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.5,0,1.1709108102321044,1.1709108102321044,1.03154347000231,3.0694169067637533,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.5,0,1.121183108088346,1.121183108088346,1.1323548696261625,2.4869035991823405,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.7,0,12.194744104748478,12.194744104748478,17.30659546718675,3.0071026229553586,False
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.7,0,1.5447922522903592,1.5447922522903592,1.1804920241840804,2.951063336113197,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.7,0,1.7892838697251001,1.7892838697251001,0.8379302909642669,2.843950792070373,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.7,0,1.2076947150162172,1.2076947150162172,0.979039188246149,2.5066252934498148,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.7,0,1.4566317769125114,1.4566317769125114,0.9048340369623572,2.782480265197163,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.7,0,1.221736785277545,1.221736785277545,1.0182006345159111,2.868233187743869,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.9,0,9.812042185387448,9.812042185387448,10.024394087867101,1.9131247835682448,False
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.9,0,1.4606797829555105,1.4606797829555105,1.2049088176750555,2.8309282025467604,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.9,0,2.5428612046485246,2.5428612046485246,1.9065871342973133,3.523569403144588,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.9,0,1.3351303477226828,1.3351303477226828,1.1401216115655313,2.9174603485808484,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.9,0,2.544007470488948,2.544007470488948,1.9005256592336783,3.624183201027224,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.9,0,1.3368840227251995,1.3368840227251995,1.1422806239899157,2.887303087811413,True
"""

exp4_fgo_results = """
estimator,experiment,scenario,platform,sigma_a,tdma_interval,packet_loss_prob,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.0,1,12.499216953541849,12.499216953541849,4.225605638275798,6.781471326205747,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.0,1,1.9203583921752556,1.9203583921752556,1.2365374251885703,3.07703194660809,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.0,1,2.0392636915468048,2.0392636915468048,2.695140281500117,4.163085632851569,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.0,1,1.7433654579250368,1.7433654579250368,1.3033492928759003,3.0858728292081667,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.0,1,1.9652196440490912,1.9652196440490912,2.922825121341213,3.424063232593386,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.0,1,1.7650829583623628,1.7650829583623628,1.4499877267560075,3.1449122048846587,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.1,1,15.985559841316451,15.985559841316451,23.58278450504067,7.9376689305342,False
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.1,1,1.7797002150509424,1.7797002150509424,1.4598273735697065,3.153115374879684,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.1,1,2.8964526938248065,2.8964526938248065,1.5194597585050442,4.012686250396108,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.1,1,1.9630847776573075,1.9630847776573075,0.7761346820984263,3.211574452025482,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.1,1,2.845752907874533,2.845752907874533,2.188381990288122,4.701772604049246,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.1,1,1.984886330196378,1.984886330196378,0.8867892044914849,3.1634644056159376,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.3,1,13.065254782113309,13.065254782113309,4.137361725624273,3.500004163701154,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.3,1,1.7842095994847484,1.7842095994847484,1.5339130712640618,3.163334464128881,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.3,1,3.2410395014153757,3.2410395014153757,4.128207175550642,3.415902096232692,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.3,1,1.9671920146713544,1.9671920146713544,1.0487029972941813,3.1801180412215144,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.3,1,2.7751848728756516,2.7751848728756516,3.743018670698538,3.3823460689543055,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.3,1,1.778875649663805,1.778875649663805,0.735595318511256,3.042455038818133,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.5,1,12.773908955053978,12.773908955053978,11.502729356813594,4.202836078830386,False
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.5,1,1.7952868533517023,1.7952868533517023,1.5348369841856933,3.149986063167186,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.5,1,4.7634186779997485,4.7634186779997485,7.348398865153085,3.671402768928469,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.5,1,2.170136754342654,2.170136754342654,1.692811289856612,3.062020385406674,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.5,1,4.533271186419198,4.533271186419198,5.505812191360789,3.568671526007651,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.5,1,2.3089541940714997,2.3089541940714997,3.0165828926191103,3.088902206430281,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.7,1,50.00441377211431,50.00441377211431,54.407473659430146,10.723717467940412,False
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.7,1,1.8019780400285288,1.8019780400285288,1.522374916854487,3.1593475912620903,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.7,1,5.354550721970101,5.354550721970101,1.6586736942189706,3.380747404925574,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.7,1,1.7899177616376474,1.7899177616376474,1.4198422366009855,3.1555867063211287,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.7,1,5.219050659808985,5.219050659808985,1.399069419780705,3.3830458106646324,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.7,1,1.7863993581825564,1.7863993581825564,1.4120469579314996,3.144576755706696,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.9,1,96.04338731448097,96.04338731448097,110.87128880214964,-7495690737124451.0,False
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.9,1,1.785819739181558,1.785819739181558,0.6506821286479979,3.179981369887347,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.9,1,8.804106120228365,8.804106120228365,6.888244049767075,-2.6276165043590924e+16,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.9,1,1.7960338200109136,1.7960338200109136,1.5261465126625953,3.1418058372273077,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.9,1,7.640988944465215,7.640988944465215,5.323996729252455,-4408066531496337.0,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.9,1,1.7529934387420705,1.7529934387420705,1.131726752503197,3.0622086325551434,True
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
		packet_loss_value = row.get("packet_loss_prob", row.get("packet_loss_probability", "nan"))
		rows.append(
			{
				"estimator": row.get("estimator", "").strip(),
				"scenario": scenario,
				"vehicle": vehicle,
				"packet_loss_probability": float(packet_loss_value),
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
# 	marker_map = {"ESKF": "o", "FGO": "s"}
# 	for estimator in ESTIMATOR_ORDER:
# 		for scenario in SCENARIO_LABELS:
# 			series = [
# 				r
# 				for r in rows
# 				if r["vehicle"] == vehicle and r["estimator"] == estimator and r["scenario"] == scenario
# 			]
# 			if not series:
# 				continue
# 			series = sorted(series, key=lambda r: r["packet_loss_probability"])
# 			intervals = [r["packet_loss_probability"] for r in series]
# 			values = [r[metric] for r in series]
# 			label = f"{estimator} {scenario}"
# 			ax.plot(intervals, values, marker=marker_map.get(estimator, "o"), label=label)

# 	ax.set_xlabel("Packet Loss Probability")
# 	ax.set_ylabel(ylabel)
# 	# ax.set_xscale("log")
	# ax.set_yscale("log")
# 	ax.set_xlim(0.0, 1.0)
# 	ax.set_title(vehicle)
# 	ax.grid(True, linestyle="--", alpha=0.4)
def _plot_metric(ax, rows: list[dict], vehicle: str, metric: str, ylabel: str) -> None:
    marker_map = {"ESKF": "o", "FGO": "s"}

    for estimator in ESTIMATOR_ORDER:
        for scenario in SCENARIO_LABELS:
            series = [
                r
                for r in rows
                if r["vehicle"] == vehicle
                and r["estimator"] == estimator
                and r["scenario"] == scenario
            ]

            if not series:
                continue

            series = sorted(series, key=lambda r: r["packet_loss_probability"])

            intervals = [r["packet_loss_probability"] for r in series]
            values = [r[metric] for r in series]

            label = f"{estimator} {scenario}"
            ax.plot(intervals, values, marker=marker_map.get(estimator, "o"), label=label)

    ax.set_xlabel("Packet Loss Probability")
    ax.set_ylabel(ylabel)
	
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])

    ax.set_title(vehicle)
    ax.grid(True, linestyle="--", alpha=0.4)


def main() -> None:
	gt_traj = "Circular/Figure 8" if TRAJECTORY_TYPE == TrajectoryType.FIGURE_8 else "Linear/Linear-turns"
	parser = argparse.ArgumentParser(description="Plot exp4 packet loss sweep results.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(f"plots/{TRAJECTORY_TYPE.value}/exp4_packet_loss"),
		help="Directory to save figures (optional).",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Do not display figures interactively.",
	)
	args = parser.parse_args()

	eskf_rows = _parse_results(exp4_eskf_results, "vehicle")
	fgo_rows = _parse_results(exp4_fgo_results, "platform")
	rows = eskf_rows + fgo_rows

	fig_ate, axes_ate = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
	_plot_metric(axes_ate[0], rows, "ROV", "ate", "ATE (RMSE)")
	_plot_metric(axes_ate[1], rows, "ASV", "ate", "ATE (RMSE)")
	axes_ate[1].legend(loc="upper right", fontsize=8)
	fig_ate.suptitle(rf"{gt_traj}: ATE (RMSE) vs Packet Loss Probability", fontsize=14)
	fig_ate.tight_layout()

	fig_nees, axes_nees = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
	lower, upper = _chi2_95_interval_3dof()
	for ax in axes_nees:
		ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\chi^2_{0.95,3}$")
	_plot_metric(axes_nees[0], rows, "ROV", "mean_nees", "NEES")
	_plot_metric(axes_nees[1], rows, "ASV", "mean_nees", "NEES")
	axes_nees[1].legend(loc="upper right", fontsize=8)
	fig_nees.suptitle(rf"{gt_traj}: NEES vs Packet Loss Probability", fontsize=14)
	fig_nees.tight_layout()

	if args.output_dir is not None:
		args.output_dir.mkdir(parents=True, exist_ok=True)
		fig_ate.savefig(args.output_dir / "exp4_ate_vs_packet_loss.svg", bbox_inches="tight")
		fig_nees.savefig(args.output_dir / "exp4_nees_vs_packet_loss.svg", bbox_inches="tight")

	if not args.no_show:
		plt.show()


if __name__ == "__main__":
	main()
