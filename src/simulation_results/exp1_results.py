# scripts/exp1_cv_tuning_pipeline.py

from __future__ import annotations

from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count

# Ensure workspace src/ is on sys.path when running this file directly.
WORKSPACE_SRC = Path(__file__).resolve().parents[1]
if str(WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_SRC))

from tracking_and_navigation.generate_trajectories import TrajectoryType
from tracking_and_navigation.run_simulations import run_simulations_s1

from simulation_results.eskf_experiments import (
    extract_eskf_result,
    results_to_dataframe,
    save_results_csv,
    generate_sigma_figures,
    save_figures,
)

TRAJECTORY_TYPE = TrajectoryType.FIGURE_8

ACOUSTIC_DELAY = True
JITTER_STD     = 0.0
MISS_PROB      = 0.0
SOUND_SPEED    = 1500.0
TDMA_FREQ      = 0.2  # 1 msg / 5 s

SIGMA_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]


# ✅ Worker function (runs in separate process)
def run_single_sigma(sigma_a: float):
    print(f"[exp1] Running sigma_a = {sigma_a}")

    # IMPORTANT: import inside function for process isolation safety
    from tracking_and_navigation.tuning_sim import eskf_sim

    eskf_sim.modelCvRov.sigma_a = float(sigma_a)

    outdir = Path("results/exp1_cv_tuning")
    save_dir = outdir / f"raw/cv_{sigma_a:.3f}"
    save_dir.mkdir(parents=True, exist_ok=True)

    plotter = run_simulations_s1(
        TRAJECTORY_TYPE=TRAJECTORY_TYPE,
        ACOUSTIC_DELAY=ACOUSTIC_DELAY,
        JITTER_STD=JITTER_STD,
        MISS_PROB=MISS_PROB,
        SOUND_SPEED=SOUND_SPEED,
        TDMA_FREQ=TDMA_FREQ,
        SAVE_DIR=str(save_dir),
        ESKF_SIM=eskf_sim,
        INIT_FROM_GT=True,
    )

    res = extract_eskf_result(
        plotter,
        experiment="exp1",
        scenario="bearing-only",
        estimator="ESKF",
        sigma_a=float(sigma_a),
        run_idx=0,
        divergence_threshold=10.0,
    )

    return res


def main():
    outdir = Path("results/exp1_cv_tuning")
    outdir.mkdir(parents=True, exist_ok=True)

    # ✅ Choose number of workers
    n_workers = min(len(SIGMA_VALUES), cpu_count())
    print(f"Using {n_workers} parallel workers")

    # ✅ Run in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single_sigma, SIGMA_VALUES)

    # ✅ Aggregate results
    df = results_to_dataframe(results)
    save_results_csv(df, str(outdir / "exp1_results.csv"))

    # ✅ Generate figures
    figs = generate_sigma_figures(df)
    save_figures(figs, str(outdir / "figures"))

    print(f"Wrote {outdir / 'exp1_results.csv'}")
    print(f"Wrote figures to {outdir / 'figures'}")


if __name__ == "__main__":
    main()