
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

exp3_eskf_results = """
estimator,experiment,scenario,vehicle,sigma_a,tdma_interval,init_range_scale,run_idx,rmse,ate,final_error,mean_nees,converged
ESKF,exp3,scenario1,ROV,0.02,5.0,1.0,0,6.289407015003677,6.289407015003677,1.0015270627846844,9.23220609406022,True
ESKF,exp3,scenario1,ASV,0.02,5.0,1.0,0,1.5581408489307398,1.5581408489307398,1.8019599351985296,3.706216785823892,True
ESKF,exp3,scenario2,ROV,0.02,5.0,1.0,0,1.5231171886859212,1.5231171886859212,0.5226942593644692,3.885583485813994,True
ESKF,exp3,scenario2,ASV,0.02,5.0,1.0,0,1.5571674193548934,1.5571674193548934,1.2134391134825961,4.148101129817294,True
ESKF,exp3,scenario3,ROV,0.02,5.0,1.0,0,1.4521395620963586,1.4521395620963586,0.8319306519921407,3.705172117459357,True
ESKF,exp3,scenario3,ASV,0.02,5.0,1.0,0,1.5127449642999755,1.5127449642999755,1.2880383067768497,3.9735784707946196,True
ESKF,exp3,scenario1,ROV,0.02,10.0,1.0,0,4.891529161548781,4.891529161548781,4.897746306945873,4.567461588315841,True
ESKF,exp3,scenario1,ASV,0.02,10.0,1.0,0,1.5000198714303212,1.5000198714303212,1.7320084762809975,3.2072367438011304,True
ESKF,exp3,scenario2,ROV,0.02,10.0,1.0,0,1.8013375896355148,1.8013375896355148,1.0330132392411255,3.3936915456689953,True
ESKF,exp3,scenario2,ASV,0.02,10.0,1.0,0,1.5618296540910106,1.5618296540910106,1.6308838484629204,3.636054163933656,True
ESKF,exp3,scenario3,ROV,0.02,10.0,1.0,0,1.7673487731905202,1.7673487731905202,0.9053955211844857,3.486666797175459,True
ESKF,exp3,scenario3,ASV,0.02,10.0,1.0,0,1.5529716030930192,1.5529716030930192,1.6327663560453005,3.648636883565712,True
ESKF,exp3,scenario1,ROV,0.02,20.0,1.0,0,4.267612307967239,4.267612307967239,5.049599253436974,3.5532795439521014,True
ESKF,exp3,scenario1,ASV,0.02,20.0,1.0,0,1.5413453018996988,1.5413453018996988,1.765639146098466,3.2667701464233745,True
ESKF,exp3,scenario2,ROV,0.02,20.0,1.0,0,2.0068614774411753,2.0068614774411753,3.2700697447721403,2.2593384238627263,True
ESKF,exp3,scenario2,ASV,0.02,20.0,1.0,0,1.5421820277876614,1.5421820277876614,1.8003401536613903,3.2722759243327895,True
ESKF,exp3,scenario3,ROV,0.02,20.0,1.0,0,1.9897540573507664,1.9897540573507664,3.4234475891523033,2.3557975455883486,True
ESKF,exp3,scenario3,ASV,0.02,20.0,1.0,0,1.5305933017161106,1.5305933017161106,1.8007756960938737,3.2350423504892807,True
ESKF,exp3,scenario1,ROV,0.02,30.0,1.0,0,6.346072225227417,6.346072225227417,2.3174236389920284,4.265351517291776,True
ESKF,exp3,scenario1,ASV,0.02,30.0,1.0,0,1.564968181379117,1.564968181379117,1.800038495510343,3.373382494012897,True
ESKF,exp3,scenario2,ROV,0.02,30.0,1.0,0,2.6166662324481065,2.6166662324481065,2.2778343360809825,2.015723728763492,True
ESKF,exp3,scenario2,ASV,0.02,30.0,1.0,0,1.5622655457866748,1.5622655457866748,1.7624237012339792,3.3393678551046464,True
ESKF,exp3,scenario3,ROV,0.02,30.0,1.0,0,2.614882569344693,2.614882569344693,2.318305850497761,2.0991365815645056,True
ESKF,exp3,scenario3,ASV,0.02,30.0,1.0,0,1.562583209858052,1.562583209858052,1.7661415074184332,3.3833252545725556,True
ESKF,exp3,scenario1,ROV,0.02,60.0,1.0,0,12.52089989176147,12.52089989176147,28.793506534411808,2.5828472941847007,False
ESKF,exp3,scenario1,ASV,0.02,60.0,1.0,0,1.5483049640693232,1.5483049640693232,1.7656381103779457,3.275972742671321,True
ESKF,exp3,scenario2,ROV,0.02,60.0,1.0,0,6.222100717872697,6.222100717872697,6.713481346817509,2.71408365699752,True
ESKF,exp3,scenario2,ASV,0.02,60.0,1.0,0,1.5600238547889242,1.5600238547889242,1.7647457218919576,3.3052685307832106,True
ESKF,exp3,scenario3,ROV,0.02,60.0,1.0,0,6.10537790202347,6.10537790202347,6.270088729788133,2.543620634832379,True
ESKF,exp3,scenario3,ASV,0.02,60.0,1.0,0,1.5526608694435964,1.5526608694435964,1.7664509530671002,3.2790161683220336,True
ESKF,exp3,scenario1,ROV,0.02,120.0,1.0,0,26.907092368690876,26.907092368690876,60.3577461543281,6.19743314880573,False
ESKF,exp3,scenario1,ASV,0.02,120.0,1.0,0,1.5508441919023745,1.5508441919023745,1.7660353872140564,3.2856934401649167,True
ESKF,exp3,scenario2,ROV,0.02,120.0,1.0,0,18.769194163294795,18.769194163294795,28.89410570172401,7.897257494831194,False
ESKF,exp3,scenario2,ASV,0.02,120.0,1.0,0,1.5613846107731826,1.5613846107731826,1.7661449336319344,3.309537568380306,True
ESKF,exp3,scenario3,ROV,0.02,120.0,1.0,0,40.9598130971695,40.9598130971695,106.69855832595258,40.33016057320837,False
ESKF,exp3,scenario3,ASV,0.02,120.0,1.0,0,1.550050817880135,1.550050817880135,1.7649864635859855,3.2473907144169774,True
"""

exp3_fgo_results = """
estimator,experiment,scenario,platform,sigma_a,tdma_interval,init_range_scale,rmse,ate,final_error,mean_nees,converged
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,5.0,1,2.0962096517270754,2.0962096517270754,1.3343004589684604,3.470428642819391,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,5.0,1,0.4469804174059428,0.4469804174059428,0.3753181568362349,3.0009269581291127,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,5.0,1,0.7400575489357742,0.7400575489357742,0.3888224547746777,3.328745528095466,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,5.0,1,0.5061920063818663,0.5061920063818663,0.2630435070864341,3.0205025227015723,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,5.0,1,0.6788059346546809,0.6788059346546809,0.3846842231983632,3.2804664108589647,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,5.0,1,0.4967778004390646,0.4967778004390646,0.2735083658801621,3.0124736735999114,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,10.0,1,2.351879221068937,2.351879221068937,2.862828209696309,3.03796304509586,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,10.0,1,0.4449334014273001,0.4449334014273001,0.1486942005298262,3.024002626057878,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,10.0,1,0.7096132322922615,0.7096132322922615,1.678164819252088,3.0522729697187527,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,10.0,1,0.4555911036683599,0.4555911036683599,0.1441988368435971,3.021187383972924,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,10.0,1,0.6708392202306469,0.6708392202306469,1.411381144850812,3.0620493997576905,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,10.0,1,0.4629823131492451,0.4629823131492451,0.1817950245611033,3.016929611450429,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,20.0,1,5.397295321412012,5.397295321412012,3.868299403714067,3.003978375423745,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,20.0,1,0.423150756058709,0.423150756058709,0.128105079257936,3.005797039721473,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,20.0,1,0.8584095027278631,0.8584095027278631,1.5411672494161273,4.244323308621014,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,20.0,1,0.4148528422658401,0.4148528422658401,0.1569507667184944,3.009472846715156,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,20.0,1,0.836317896070633,0.836317896070633,1.4512710383194891,4.203798032697988,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,20.0,1,0.4179509826775365,0.4179509826775365,0.1554216244629293,3.0099872078673195,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,30.0,1,2214.2621203583262,2214.2621203583262,5359.930381383147,3.540415665956483,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,30.0,1,0.5508597055549364,0.5508597055549364,0.2896227744529846,3.0249141515636193,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,30.0,1,0.9045543665596554,0.9045543665596554,0.8479311354817946,3.0397208328195284,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,30.0,1,0.4220745834412719,0.4220745834412719,0.1252824567848051,3.0113233220946416,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,30.0,1,0.8693564232015499,0.8693564232015499,0.7800788661315754,3.0158031829753065,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,30.0,1,0.423377736155167,0.423377736155167,0.1249087762929646,3.010458783824702,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,60.0,1,22.86963188063409,22.86963188063409,2.1638863765212,10.831586207230009,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,60.0,1,0.4163338043222125,0.4163338043222125,0.1369990887863984,3.0092976535396234,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,60.0,1,0.9649454257493096,0.9649454257493096,0.9109624161734634,6.5632213912940545,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,60.0,1,0.4164411401443101,0.4164411401443101,0.1280709888712131,3.0091597954084683,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,60.0,1,0.93524388014606,0.93524388014606,0.9038951998185776,5.962802853658804,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,60.0,1,0.4173293132085954,0.4173293132085954,0.1281156719779481,3.0100320349155725,True
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ROV,0.02,120.0,1,25.289773265611316,25.289773265611316,15.206878331295972,2322.7503567216763,False
FGO,exp3_tdma_sweep,FGO - Scenario 1: Bearing-only,ASV,0.02,120.0,1,0.4154025799272818,0.4154025799272818,0.130037886800617,3.0091122916642203,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ROV,0.02,120.0,1,3.932007096700638,3.932007096700638,0.6961600606082111,15.883172273599351,True
FGO,exp3_tdma_sweep,FGO - Scenario 2: Bearing + range,ASV,0.02,120.0,1,0.4151777149694788,0.4151777149694788,0.1324294963125156,3.00850440212305,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ROV,0.02,120.0,1,3.925096789865788,3.925096789865788,0.6657174693861738,20.014045837798857,True
FGO,exp3_tdma_sweep,FGO - Scenario 3: Bearing + range + depth,ASV,0.02,120.0,1,0.4203152691723417,0.4203152691723417,0.1320294614654307,3.0079131780266093,True
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
	parser = argparse.ArgumentParser(description="Plot exp3 TDMA sweep results.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("plots/exp3_tdma"),
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
	fig_ate.suptitle(r"TDMA sensitivity: ATE (RMSE) vs $\sigma_a$")
	fig_ate.tight_layout()

	fig_nees, axes_nees = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
	lower, upper = _chi2_95_interval_3dof()
	for ax in axes_nees:
		ax.axhspan(lower, upper, color="grey", alpha=0.15, label=r"$95\%\,\chi^2_{3}$")
	_plot_metric(axes_nees[0], rows, "ROV", "mean_nees", "NEES")
	_plot_metric(axes_nees[1], rows, "ASV", "mean_nees", "NEES")
	axes_nees[1].legend(loc="upper right", fontsize=8)
	fig_nees.suptitle(r"TDMA sensitivity: NEES vs $\sigma_a$") 
	fig_nees.tight_layout()

	if args.output_dir is not None:
		args.output_dir.mkdir(parents=True, exist_ok=True)
		fig_ate.savefig(args.output_dir / "exp3_ate_vs_tdma_interval.svg", bbox_inches="tight")
		fig_nees.savefig(args.output_dir / "exp3_nees_vs_tdma_interval.svg", bbox_inches="tight")

	if not args.no_show:
		plt.show()


if __name__ == "__main__":
	main()
