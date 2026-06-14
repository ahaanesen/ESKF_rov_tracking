
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
	
NORMAL_FONTSIZE = 10  # use 11 or 12 if your thesis text is 11pt or 12pt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",

        "font.size": NORMAL_FONTSIZE,
        "axes.titlesize": NORMAL_FONTSIZE,
        "axes.labelsize": NORMAL_FONTSIZE,
        "xtick.labelsize": NORMAL_FONTSIZE - 1,
        "ytick.labelsize": NORMAL_FONTSIZE - 1,
        "legend.fontsize": NORMAL_FONTSIZE - 1,
        "figure.titlesize": NORMAL_FONTSIZE,

        "axes.unicode_minus": False,
    }
)

TRAJECTORY_TYPE = TrajectoryType.LINEAR_TURNS  # set in main()

exp3_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,packet_loss_prob,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp3,scenario1,ROV,0.02,5.0,1.0,,0,16.172272025693946,16.172272025693946,12.635658806211817,3.08582080434726,False
ESKF,exp3,scenario1,ASV,0.02,5.0,1.0,,0,1.35221331487032,1.35221331487032,1.5147185680608166,2.4563184971812837,True
ESKF,exp3,scenario2,ROV,0.02,5.0,1.0,,0,2.612690776381963,2.612690776381963,1.7535352272791853,3.1047727179220725,True
ESKF,exp3,scenario2,ASV,0.02,5.0,1.0,,0,1.294337837379559,1.294337837379559,1.4799682946554942,2.515845628107432,True
ESKF,exp3,scenario3,ROV,0.02,5.0,1.0,,0,2.2900927288046993,2.2900927288046993,1.547801191811592,2.8845302037847684,True
ESKF,exp3,scenario3,ASV,0.02,5.0,1.0,,0,1.286136523731033,1.286136523731033,1.4818209043419672,2.503411261841935,True
ESKF,exp3,scenario1,ROV,0.02,10.0,1.0,,0,19.753308016832474,19.753308016832474,7.41896976275806,4.087372911083323,True
ESKF,exp3,scenario1,ASV,0.02,10.0,1.0,,0,1.3659470312282054,1.3659470312282054,1.5110635511158057,2.492021891813291,True
ESKF,exp3,scenario2,ROV,0.02,10.0,1.0,,0,3.3380899898853964,3.3380899898853964,4.931308007060394,2.998867574210927,True
ESKF,exp3,scenario2,ASV,0.02,10.0,1.0,,0,1.4016742650749279,1.4016742650749279,1.549653652468389,2.742183359265344,True
ESKF,exp3,scenario3,ROV,0.02,10.0,1.0,,0,2.8437191860257363,2.8437191860257363,4.765941090554096,2.8023317258198257,True
ESKF,exp3,scenario3,ASV,0.02,10.0,1.0,,0,1.3944713044189123,1.3944713044189123,1.5502998364928908,2.726721039562043,True
ESKF,exp3,scenario1,ROV,0.02,20.0,1.0,,0,15.539549055399656,15.539549055399656,22.276280311344397,3.3854942785004924,False
ESKF,exp3,scenario1,ASV,0.02,20.0,1.0,,0,1.355669795045615,1.355669795045615,1.4786131278549428,2.453064769962737,True
ESKF,exp3,scenario2,ROV,0.02,20.0,1.0,,0,4.694736630914474,4.694736630914474,9.75485549242703,3.2053491284665143,True
ESKF,exp3,scenario2,ASV,0.02,20.0,1.0,,0,1.3629447643491297,1.3629447643491297,1.4863331807636204,2.4681719921716896,True
ESKF,exp3,scenario3,ROV,0.02,20.0,1.0,,0,3.932455842319245,3.932455842319245,9.284129917246332,2.854990932436848,True
ESKF,exp3,scenario3,ASV,0.02,20.0,1.0,,0,1.3613593141914253,1.3613593141914253,1.4833735606140708,2.4706283380365006,True
ESKF,exp3,scenario1,ROV,0.02,30.0,1.0,,0,25.918485175954505,25.918485175954505,25.40966619018933,8.815103745836288,False
ESKF,exp3,scenario1,ASV,0.02,30.0,1.0,,0,1.3640968463101226,1.3640968463101226,1.4787136616192864,2.4896722676728196,True
ESKF,exp3,scenario2,ROV,0.02,30.0,1.0,,0,5.876167268212874,5.876167268212874,6.105115750569585,3.781216540725072,True
ESKF,exp3,scenario2,ASV,0.02,30.0,1.0,,0,1.3671332010403512,1.3671332010403512,1.4779691237267212,2.480456667706903,True
ESKF,exp3,scenario3,ROV,0.02,30.0,1.0,,0,5.44176228505712,5.44176228505712,4.36519549594931,3.487393492754541,True
ESKF,exp3,scenario3,ASV,0.02,30.0,1.0,,0,1.3590126333144905,1.3590126333144905,1.4754347266989583,2.439033817906492,True
ESKF,exp3,scenario1,ROV,0.02,60.0,1.0,,0,25.01217747518196,25.01217747518196,27.097698039671055,5.188938993074334,False
ESKF,exp3,scenario1,ASV,0.02,60.0,1.0,,0,1.353955601963468,1.353955601963468,1.4747915988810392,2.4467298701121325,True
ESKF,exp3,scenario2,ROV,0.02,60.0,1.0,,0,10.008012810274211,10.008012810274211,3.931185645897941,3.2650190679213122,True
ESKF,exp3,scenario2,ASV,0.02,60.0,1.0,,0,1.3629075331201643,1.3629075331201643,1.475673093036418,2.459095128195322,True
ESKF,exp3,scenario3,ROV,0.02,60.0,1.0,,0,9.3845538023342,9.3845538023342,2.7132323991524787,2.731905102536629,True
ESKF,exp3,scenario3,ASV,0.02,60.0,1.0,,0,1.361892355412898,1.361892355412898,1.4772988905127704,2.4612669561092044,True
ESKF,exp3,scenario1,ROV,0.02,120.0,1.0,,0,71.86158635220885,71.86158635220885,100.45385815335935,39.155622662253364,False
ESKF,exp3,scenario1,ASV,0.02,120.0,1.0,,0,1.3577367249724646,1.3577367249724646,1.4758323992503544,2.460072186931705,True
ESKF,exp3,scenario2,ROV,0.02,120.0,1.0,,0,51.72301458070271,51.72301458070271,38.63618698500948,47.634298281838944,False
ESKF,exp3,scenario2,ASV,0.02,120.0,1.0,,0,1.3663840942196073,1.3663840942196073,1.4766273494530398,2.4696410730254175,True
ESKF,exp3,scenario3,ROV,0.02,120.0,1.0,,0,87.94423275880347,87.94423275880347,58.050459882869944,72.2741045962402,False
ESKF,exp3,scenario3,ASV,0.02,120.0,1.0,,0,1.3732108841692834,1.3732108841692834,1.4764461922481387,2.515222826095832,True
"""

exp3_fgo_results = """
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,1,11.998011461629744,11.998011461629744,5.94164636488514,2.966699030733412,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,1,0.4653012262842345,0.4653012262842345,0.423686398905869,0.1827841095332082,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,1,2.054255677947461,2.054255677947461,2.8825012067414373,1.650745651890946,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,1,0.4607092598274242,0.4607092598274242,0.3435787468300479,0.1908798900532654,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,1,1.7084588719685996,1.7084588719685996,2.85160993322141,1.2100553776907752,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,1,0.4824986307357285,0.4824986307357285,0.3518384898428089,0.2270599962464855,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,10.0,1,17.84640815173948,17.84640815173948,44.40431709638423,4.759569271817808,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,10.0,1,0.4368162635469946,0.4368162635469946,0.520833710620783,0.150526705342179,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,10.0,1,2.4148505897234607,2.4148505897234607,5.55755214599732,1.4233514777237006,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,10.0,1,0.4358552766105302,0.4358552766105302,0.5084906529384119,0.1514804410314268,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,10.0,1,2.339458651488523,2.339458651488523,4.367961656950816,2.221156392752851,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,10.0,1,0.7889940529906043,0.7889940529906043,0.3750451847171227,0.6452340234305302,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,20.0,1,28.791645928880666,28.791645928880666,23.49093276365291,8.78709094258813,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,20.0,1,0.5475361412355877,0.5475361412355877,0.997896811316746,0.3576842493416324,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,20.0,1,2.302638289781168,2.302638289781168,1.7333005523630118,1.5570953244538046,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,20.0,1,0.4404667551973249,0.4404667551973249,0.5747697940775759,0.1442136102964493,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,20.0,1,1.9848397752553295,1.9848397752553295,1.6918505421610524,1.3485740635046837,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,20.0,1,0.4424397194364225,0.4424397194364225,0.5711487999668495,0.147600815333144,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,30.0,1,43.66240027825076,43.66240027825076,41.2914392551216,11.167516893345084,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,30.0,1,0.4191015717393904,0.4191015717393904,0.476598908820947,0.1325776625937347,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,30.0,1,2.385624798520359,2.385624798520359,2.998342827025616,1.1467283419888072,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,30.0,1,0.4254278451441132,0.4254278451441132,0.5030737868971743,0.13540099722262,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,30.0,1,2.044866911184948,2.044866911184948,1.8431936998666916,0.9186064000510832,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,30.0,1,0.4268645732406673,0.4268645732406673,0.4962188949842085,0.1384714696505158,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,60.0,1,50.417566246263874,50.417566246263874,32.60537946533516,93.64002597829712,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,60.0,1,0.4132465435142067,0.4132465435142067,0.4645022264494282,0.1288945211925847,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,60.0,1,1.738358510622904,1.738358510622904,4.5507843625369295,0.7294340526369836,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,60.0,1,0.4161287976122047,0.4161287976122047,0.4777994349842055,0.1305446237866897,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,60.0,1,1.6033373021136863,1.6033373021136863,4.646838126809038,0.4298024295295806,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,60.0,1,0.3565572110359762,0.3565572110359762,0.3622779281009432,0.099535957539479,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,120.0,1,101.8697100374878,101.8697100374878,43.38949191810506,2060.981160684345,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,120.0,1,0.4177811430442806,0.4177811430442806,0.482682590219673,0.1313624891503711,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,120.0,1,1.4786306183374611,1.4786306183374611,2.42155299701755,0.5060244756699842,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,120.0,1,0.4169113929758101,0.4169113929758101,0.4826424422654664,0.1308780997110843,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,120.0,1,1.4072793378861332,1.4072793378861332,2.382841029603913,0.5416241129293097,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,120.0,1,0.4170934733476393,0.4170934733476393,0.4812132403121354,0.1320643149526816,True
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

	ax.set_xlabel("TDMA cycle (s)")
	ax.set_ylabel(ylabel)
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_title(vehicle)
	ax.grid(True, alpha=0.3)


def main() -> None:
	gt_traj = "Circular/Figure 8" if TRAJECTORY_TYPE == TrajectoryType.FIGURE_8 else "Linear/Linear-turns"
	parser = argparse.ArgumentParser(description="Plot exp3 TDMA sweep results.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(f"plots_new/{TRAJECTORY_TYPE.value}"),
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

	fig_ate, axes_ate = plt.subplots(
		2, 1,
		figsize=(3.4, 3.8),
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	_plot_metric(axes_ate[0], rows, "UUV", "ate", "ATE (RMSE)")
	_plot_metric(axes_ate[1], rows, "USV", "ate", "ATE (RMSE)")
	axes_ate[0].set_xlabel("")
	# axes_ate[1].legend(loc="upper right", fontsize=8, framealpha=0.4)
	axes_ate[1].legend(
		loc="upper center",
		bbox_to_anchor=(0.5, 1.0),
		ncol=2,
		fontsize=8,
		framealpha=0.4,
		handlelength=1.2,
		columnspacing=0.8,
		labelspacing=0.25,
	)

	fig_nees, axes_nees = plt.subplots(
		2, 1,
		figsize=(3.4, 3.8),
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	lower, upper = _chi2_95_interval_3dof()
	for ax in axes_nees:
		ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$\chi^2_{0.95,3}$")
	_plot_metric(axes_nees[0], rows, "UUV", "mean_nees", "NEES")
	_plot_metric(axes_nees[1], rows, "USV", "mean_nees", "NEES")
	axes_nees[0].set_xlabel("")
	# axes_nees[1].legend(loc="upper right", fontsize=8, framealpha=0.4)
	axes_nees[1].legend(
		loc="upper center",
		bbox_to_anchor=(0.5, 1.0),
		ncol=2,
		fontsize=8,
		framealpha=0.4,
		handlelength=1.2,
		columnspacing=0.8,
		labelspacing=0.25,
	)

	pre_fig = "fig8" if TRAJECTORY_TYPE == TrajectoryType.FIGURE_8 else "linear"

	if args.output_dir is not None:
		args.output_dir.mkdir(parents=True, exist_ok=True)
		fig_ate.savefig(args.output_dir / f"{pre_fig}_exp3_ate_vs_tdma_interval_new.svg", bbox_inches="tight")
		fig_nees.savefig(args.output_dir / f"{pre_fig}_exp3_nees_vs_tdma_interval_new.svg", bbox_inches="tight")

	if not args.no_show:
		plt.show()


if __name__ == "__main__":
	main()
