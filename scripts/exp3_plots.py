
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

TRAJECTORY_TYPE = TrajectoryType.LINEAR_TURNS  # set in main()

exp3_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp3,scenario1,ROV,0.02,5.0,1.0,,0,81.04586946233731,81.04586946233731,45.99726392087533,3583.1707374829903,False
ESKF,exp3,scenario1,ASV,0.02,5.0,1.0,,0,1.6837808565810146,1.6837808565810146,0.6687238696071345,4.280560534520253,True
ESKF,exp3,scenario2,ROV,0.02,5.0,1.0,,0,2.7686450820348747,2.7686450820348747,1.8317623056472627,2.941481175656741,True
ESKF,exp3,scenario2,ASV,0.02,5.0,1.0,,0,1.374886567642643,1.374886567642643,0.6540351106046274,3.120695627231762,True
ESKF,exp3,scenario3,ROV,0.02,5.0,1.0,,0,2.3859680273015806,2.3859680273015806,1.7261068087439102,2.839485381512924,True
ESKF,exp3,scenario3,ASV,0.02,5.0,1.0,,0,1.3583968892744158,1.3583968892744158,0.6461487028537286,3.096406173421828,True
ESKF,exp3,scenario1,ROV,0.02,10.0,1.0,,0,134.30314335192335,134.30314335192335,156.09253596683916,3449.081264688809,False
ESKF,exp3,scenario1,ASV,0.02,10.0,1.0,,0,1.7151543451404818,1.7151543451404818,0.8566368605722909,4.163326467292389,True
ESKF,exp3,scenario2,ROV,0.02,10.0,1.0,,0,3.8952434814338295,3.8952434814338295,4.199529316919088,3.538508423633868,True
ESKF,exp3,scenario2,ASV,0.02,10.0,1.0,,0,1.3914747924198252,1.3914747924198252,0.8577243182160137,2.968367640178215,True
ESKF,exp3,scenario3,ROV,0.02,10.0,1.0,,0,2.659035873585314,2.659035873585314,3.8918263117654366,2.1792420798585677,True
ESKF,exp3,scenario3,ASV,0.02,10.0,1.0,,0,1.4128365545146133,1.4128365545146133,0.8484477976960266,3.1030408481499405,True
ESKF,exp3,scenario1,ROV,0.02,20.0,1.0,,0,182.33134573407597,182.33134573407597,244.7186635707327,1704.0566692097464,False
ESKF,exp3,scenario1,ASV,0.02,20.0,1.0,,0,1.4896508669184232,1.4896508669184232,0.808602830533697,3.193047978563364,True
ESKF,exp3,scenario2,ROV,0.02,20.0,1.0,,0,4.597433793162377,4.597433793162377,1.909063634561693,3.2937665092883797,True
ESKF,exp3,scenario2,ASV,0.02,20.0,1.0,,0,1.4480022307593197,1.4480022307593197,0.8206742382178376,3.0388543320720607,True
ESKF,exp3,scenario3,ROV,0.02,20.0,1.0,,0,4.033035881724957,4.033035881724957,2.0569849389466413,3.005315382180301,True
ESKF,exp3,scenario3,ASV,0.02,20.0,1.0,,0,1.448670194606013,1.448670194606013,0.8197158039871095,3.073189674023631,True
ESKF,exp3,scenario1,ROV,0.02,30.0,1.0,,0,459.9195698576411,459.9195698576411,803.470110660692,1685.0876385165504,False
ESKF,exp3,scenario1,ASV,0.02,30.0,1.0,,0,1.4661093925613369,1.4661093925613369,0.8305878144615558,3.090118388163225,True
ESKF,exp3,scenario2,ROV,0.02,30.0,1.0,,0,4.433543232352164,4.433543232352164,2.07113002613772,1.9508227515345036,True
ESKF,exp3,scenario2,ASV,0.02,30.0,1.0,,0,1.4520172737288677,1.4520172737288677,0.8107908797966842,3.0515110102877383,True
ESKF,exp3,scenario3,ROV,0.02,30.0,1.0,,0,3.8842481901536,3.8842481901536,1.4609350020256378,1.6326637409569726,True
ESKF,exp3,scenario3,ASV,0.02,30.0,1.0,,0,1.4442073869540026,1.4442073869540026,0.8105368166117474,3.008790712409683,True
ESKF,exp3,scenario1,ROV,0.02,60.0,1.0,,0,1276.1940211630588,1276.1940211630588,2593.135007713517,5588.904078073464,False
ESKF,exp3,scenario1,ASV,0.02,60.0,1.0,,0,1.4369503136067823,1.4369503136067823,0.8047770604301038,2.9812639094367883,True
ESKF,exp3,scenario2,ROV,0.02,60.0,1.0,,0,9.520532774702156,9.520532774702156,7.555281836318296,2.358615460290108,True
ESKF,exp3,scenario2,ASV,0.02,60.0,1.0,,0,1.4529971801551813,1.4529971801551813,0.8056240376890679,3.045889142860225,True
ESKF,exp3,scenario3,ROV,0.02,60.0,1.0,,0,9.664023272126016,9.664023272126016,5.26498964830472,2.7330182898888546,True
ESKF,exp3,scenario3,ASV,0.02,60.0,1.0,,0,1.4433307460964107,1.4433307460964107,0.8072235910282931,3.0011680397464207,True
ESKF,exp3,scenario1,ROV,0.02,120.0,1.0,,0,580.0212411569124,580.0212411569124,1007.9970489228775,3454.865631883439,False
ESKF,exp3,scenario1,ASV,0.02,120.0,1.0,,0,1.4519222820470794,1.4519222820470794,0.80415265081292,3.057553786556428,True
ESKF,exp3,scenario2,ROV,0.02,120.0,1.0,,0,54.177319090546874,54.177319090546874,38.53142373862789,51.86129134730811,False
ESKF,exp3,scenario2,ASV,0.02,120.0,1.0,,0,1.456188768385131,1.456188768385131,0.8044409581063227,3.0593900719658382,True
ESKF,exp3,scenario3,ROV,0.02,120.0,1.0,,0,75.53689865725978,75.53689865725978,45.55427750087363,70.99662126060159,False
ESKF,exp3,scenario3,ASV,0.02,120.0,1.0,,0,1.4801934879319438,1.4801934879319438,0.8084772575697818,3.209494359045436,True
"""

exp3_fgo_results = """
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,1,26.038833901308,26.038833901308,15.690213319599769,10.19017679546724,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,1,1.7717112397174086,1.7717112397174086,0.7746973914366491,3.149957351637334,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,1,2.242426214652015,2.242426214652015,2.955709278606935,3.946025814857654,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,1,1.7794047998662172,1.7794047998662172,1.477544669382863,3.1820327833763136,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,1,1.939569180002896,1.939569180002896,2.944193545306777,3.60893729544526,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,1,1.7882160948995804,1.7882160948995804,1.4696041532540578,3.203063439795673,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,10.0,1,15.675607389528311,15.675607389528311,27.362954831707764,3.902759660351105,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,10.0,1,1.8020713552825265,1.8020713552825265,1.4739031537524832,3.174118753452841,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,10.0,1,2.6459095080156154,2.6459095080156154,6.149528281071183,3.574997106889842,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,10.0,1,1.791754666260829,1.791754666260829,1.4864629151560371,3.1640380270494073,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,10.0,1,2.4744092097537567,2.4744092097537567,4.715356665305871,3.5929954318317496,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,10.0,1,1.788425850372831,1.788425850372831,1.490159705757433,3.1519812449613394,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,20.0,1,28.874062857830204,28.874062857830204,16.87037929513981,4.757146942864672,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,20.0,1,1.8010220527542091,1.8010220527542091,1.4952279576420544,3.1509591499750926,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,20.0,1,2.339638595890493,2.339638595890493,1.5782929646582775,3.4462015982764624,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,20.0,1,1.793597322533049,1.793597322533049,0.7958880490638577,3.06914283913539,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,20.0,1,2.2854278430757913,2.2854278430757913,2.8618658009800377,3.817758031388742,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,20.0,1,1.803755507092119,1.803755507092119,1.455963206592182,3.1591351640786987,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,30.0,1,43.31109709073649,43.31109709073649,43.622092789841304,42.16751470203935,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,30.0,1,1.8634719960834063,1.8634719960834063,1.8648315811195495,3.2753544667187926,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,30.0,1,2.415519316158711,2.415519316158711,3.224792273594864,3.4540777412218486,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,30.0,1,1.803429095930699,1.803429095930699,1.5058340914635169,3.156128869721024,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,30.0,1,2.335247730502415,2.335247730502415,2.9028464999129033,3.666476138437127,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,30.0,1,1.7986241306198256,1.7986241306198256,0.8186002363338385,3.063616174537666,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,60.0,1,49.8619078593052,49.8619078593052,31.19894277400452,32.25167837629638,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,60.0,1,1.7974254675006534,1.7974254675006534,1.5262450843008015,3.1452882557463746,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,60.0,1,2.0454803511454767,2.0454803511454767,4.44713219336618,6.207440529403967,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,60.0,1,1.8387574057461944,1.8387574057461944,1.9927397264521751,3.116342511120147,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,60.0,1,1.8710682039019213,1.8710682039019213,3.786641182696305,4.8072909199989775,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,60.0,1,1.7973675135469087,1.7973675135469087,1.524577736018884,3.137669936820628,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,120.0,1,101.98705587048404,101.98705587048404,43.77145425458408,3.1255129626730652e+16,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,120.0,1,1.780035543768224,1.780035543768224,0.6595428744563672,3.1687506700431025,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,120.0,1,1.5983223073520985,1.5983223073520985,2.771547540216416,9.418543202916442e+16,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,120.0,1,1.7175411013009854,1.7175411013009854,2.1274650556525345,3.0785755838421087,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,120.0,1,2.236869651308605,2.236869651308605,3.6295759701381334,143038958291683.38,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,120.0,1,1.7960461587609309,1.7960461587609309,1.5387676740922864,3.138731130567284,True
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
	for estimator in ESTIMATOR_ORDER:
		for scenario in SCENARIO_LABELS:
			series = [
				r
				for r in rows
				if r["vehicle"] == vehicle and r["estimator"] == estimator and r["scenario"] == scenario
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

	fig_ate, axes_ate = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
	_plot_metric(axes_ate[0], rows, "ROV", "ate", "ATE (RMSE)")
	_plot_metric(axes_ate[1], rows, "ASV", "ate", "ATE (RMSE)")
	axes_ate[1].legend(loc="upper right", fontsize=8)
	fig_ate.suptitle(rf"{gt_traj}: ATE (RMSE) vs TDMA interval", fontsize=14)
	fig_ate.tight_layout()

	fig_nees, axes_nees = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
	lower, upper = _chi2_95_interval_3dof()
	for ax in axes_nees:
		ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\chi^2_{0.95,3}$")
	_plot_metric(axes_nees[0], rows, "ROV", "mean_nees", "NEES")
	_plot_metric(axes_nees[1], rows, "ASV", "mean_nees", "NEES")
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
