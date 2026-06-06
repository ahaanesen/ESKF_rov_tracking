
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

TRAJECTORY_TYPE = TrajectoryType.FIGURE_8  # change to your actual trajectory type if needed 

exp4_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.0,0,4.469258268647386,4.469258268647386,2.7195221439817954,5.810787776715161,True
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.0,0,1.53961461279601,1.53961461279601,0.8279224276232627,3.981262013539334,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.0,0,1.5929049145676797,1.5929049145676797,0.6019632784080222,5.957128240616962,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.0,0,1.6353085071182012,1.6353085071182012,0.6954778343618611,6.158436358714454,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.0,0,1.4497766684169269,1.4497766684169269,0.31243857583387646,5.704829216079992,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.0,0,1.4960773607047433,1.4960773607047433,0.32517375885127786,5.8040406680165155,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.1,0,4.221557905683483,4.221557905683483,4.055371573963803,5.574121913111889,True
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.1,0,1.5337908514318646,1.5337908514318646,0.6994212775558902,3.992499357665621,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.1,0,1.623739856347446,1.623739856347446,0.7822565511545773,6.162999107488878,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.1,0,1.6708153592690071,1.6708153592690071,0.666145057876533,6.39389448472691,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.1,0,1.4714064199549772,1.4714064199549772,0.22112505513222872,5.636872412761088,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.1,0,1.5301814633046533,1.5301814633046533,0.25840330964566416,6.061338992983707,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.3,0,3.807249198942201,3.807249198942201,3.298189584685533,5.007702190578423,True
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.3,0,1.4853698778054238,1.4853698778054238,1.2436104084342727,3.7480414987016326,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.3,0,1.5857146918102463,1.5857146918102463,0.5755287678545882,5.78559286091407,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.3,0,1.6034691154653786,1.6034691154653786,0.8227311889167022,5.790273752108902,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.3,0,1.4331440634322274,1.4331440634322274,0.14737410299065778,5.359071609005154,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.3,0,1.4619706921437021,1.4619706921437021,0.4271628896555428,5.280457612415968,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.5,0,4.574106420600024,4.574106420600024,3.42984047244208,6.787901198432679,True
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.5,0,1.6040951973428135,1.6040951973428135,1.0095101304429592,4.265646150615929,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.5,0,1.65448961326084,1.65448961326084,0.42975774662972344,6.2871360979605475,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.5,0,1.6953942091186693,1.6953942091186693,0.7741083895953972,6.419670597937647,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.5,0,1.5202627063056773,1.5202627063056773,0.3600936658662582,6.056084301002278,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.5,0,1.5693547224882862,1.5693547224882862,0.16388509603395424,6.217891662982916,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.7,0,6.959026070968304,6.959026070968304,3.166836014759185,15.103971823405995,True
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.7,0,1.7383697606918316,1.7383697606918316,0.8318478244754246,4.841299517282437,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.7,0,1.3426752904421309,1.3426752904421309,0.5767761078510704,4.078231051437151,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.7,0,1.3999095840713878,1.3999095840713878,0.7475267881722534,4.225540398612071,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.7,0,1.2231461415749905,1.2231461415749905,0.2214702710015163,4.0083075017854455,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.7,0,1.3036915208156004,1.3036915208156004,0.2210778718692428,4.309397209418613,True
ESKF,exp4_packet_loss,scenario1,ROV,0.02,0.2,1.0,0.9,0,5.927950282650659,5.927950282650659,4.405483410992795,8.987077530472684,True
ESKF,exp4_packet_loss,scenario1,ASV,0.02,0.2,1.0,0.9,0,1.5980679947087078,1.5980679947087078,1.0417124088897298,4.033123142613071,True
ESKF,exp4_packet_loss,scenario2,ROV,0.02,0.2,1.0,0.9,0,1.6293022806815618,1.6293022806815618,0.7740671438061344,4.277434343175105,True
ESKF,exp4_packet_loss,scenario2,ASV,0.02,0.2,1.0,0.9,0,1.4640066391325623,1.4640066391325623,0.8626724834743806,4.145688321826693,True
ESKF,exp4_packet_loss,scenario3,ROV,0.02,0.2,1.0,0.9,0,1.3103461391152047,1.3103461391152047,0.6483686344553343,3.488501786947625,True
ESKF,exp4_packet_loss,scenario3,ASV,0.02,0.2,1.0,0.9,0,1.3217068791975326,1.3217068791975326,0.6906250078201572,3.580319966179497,True
"""

exp4_fgo_results = """
estimator,experiment,scenario,platform,sigma_a,tdma_interval,packet_loss_prob,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.0,1,2.2678867545371655,2.2678867545371655,1.3126509564595985,3.5207776529315984,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.0,1,0.4661019088087787,0.4661019088087787,0.3938395006716971,3.0032631790675492,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.0,1,0.7399153658154367,0.7399153658154367,0.3892545395829744,3.333577901705435,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.0,1,0.505280095524501,0.505280095524501,0.2617263252616982,3.020364298849506,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.0,1,0.6787213728513455,0.6787213728513455,0.4338598553266469,3.339899567198372,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.0,1,0.5185625141789018,0.5185625141789018,0.2586820827844996,3.0086881348946135,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.1,1,1.8223798358059495,1.8223798358059495,0.808742980218508,3.1151617509388596,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.1,1,0.438111871961547,0.438111871961547,0.3581934487763794,3.0169323346140944,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.1,1,0.7427747386811233,0.7427747386811233,0.8368931503429494,3.4252898941372876,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.1,1,0.6759889977422573,0.6759889977422573,0.5964474073163424,3.016616475576426,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.1,1,0.700402275193639,0.700402275193639,0.7321629196363805,3.3696576543134875,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.1,1,0.4812680177779019,0.4812680177779019,0.5490274758603555,3.0032258649161743,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.3,1,3.0264387006868634,3.0264387006868634,2.5336703432053445,3.01285872852836,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.3,1,0.4292320941018916,0.4292320941018916,0.4599743034628675,3.0077422906590328,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.3,1,0.7332630998622296,0.7332630998622296,0.5137582225224734,3.5719183626277435,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.3,1,0.4539208254791913,0.4539208254791913,0.5489680010117757,3.0258599057469566,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.3,1,0.6881604457111195,0.6881604457111195,0.4030961834279813,3.553472501015739,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.3,1,0.464976918581335,0.464976918581335,0.6073619718937454,3.014578499084672,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.5,1,2.7160679075825125,2.7160679075825125,0.179764474586195,3.1663454957822905,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.5,1,0.4559506022315681,0.4559506022315681,0.2587832340177534,3.013515991281217,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.5,1,0.8834859747473295,0.8834859747473295,0.1887082787985069,3.2963622776202817,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.5,1,0.4696400620689611,0.4696400620689611,0.1894127472830737,3.031811172197757,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.5,1,0.8354679378284366,0.8354679378284366,0.1504837344807003,3.28697329370236,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.5,1,0.4709735375014206,0.4709735375014206,0.1563063755027912,3.0387450681051535,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.7,1,4.668833884140708,4.668833884140708,3.1248722309330836,3.597063227757525,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.7,1,0.4263503771671282,0.4263503771671282,0.1838370719963686,3.007814625577311,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.7,1,1.26922649454022,1.26922649454022,1.137581065826096,3.0310012539183933,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.7,1,0.4257567126782944,0.4257567126782944,0.1969056520692448,3.0078059727989483,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.7,1,1.2130677377387498,1.2130677377387498,1.138840478809264,3.0326846181030165,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.7,1,0.4162635043578881,0.4162635043578881,0.2016358372685194,3.0041029530934766,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,0.9,1,9.131502247581649,9.131502247581649,9.477137519133358,3.874592780727117,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,0.9,1,0.42018009155656,0.42018009155656,0.1268392257571802,3.0092973961767417,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,0.9,1,1.6947479134080232,1.6947479134080232,0.9858669336827056,5.266647229919916,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,0.9,1,0.3927162464529236,0.3927162464529236,0.2314460405332046,3.006995259638327,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,0.9,1,1.652607250687752,1.652607250687752,0.8389527559242257,4.959590007433611,True
FGO,exp4_packet_loss_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,0.9,1,0.374361383129909,0.374361383129909,0.3602297248753057,3.006607165472785,True
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

	fig_ate, axes_ate = plt.subplots(2, 1, figsize=(7.5, 7.5), sharey=True)
	_plot_metric(axes_ate[0], rows, "UUV", "ate", "ATE (RMSE)")
	_plot_metric(axes_ate[1], rows, "USV", "ate", "ATE (RMSE)")
	axes_ate[1].legend(loc="upper right", fontsize=8)
	fig_ate.suptitle(rf"{gt_traj}: ATE (RMSE) vs Packet Loss Probability", fontsize=14)
	fig_ate.tight_layout()

	fig_nees, axes_nees = plt.subplots(2, 1, figsize=(7.5, 7.5), sharey=True)
	lower, upper = _chi2_95_interval_3dof()
	for ax in axes_nees:
		ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\chi^2_{0.95,3}$")
	_plot_metric(axes_nees[0], rows, "UUV", "mean_nees", "NEES")
	_plot_metric(axes_nees[1], rows, "USV", "mean_nees", "NEES")
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
