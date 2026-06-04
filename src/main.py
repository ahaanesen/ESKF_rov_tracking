
import numpy as np

from tracking_and_navigation.generate_trajectories import TrajectoryType


from tracking_and_navigation.run_simulations import run_simulations_s1
from tracking_and_navigation.tuning_sim import (
    eskf_sim,
)

# =============================================================================
# Scenario configuration — edit here to switch between modes
# =============================================================================
#
# TRAJECTORY_TYPE controls the USV/ROV motion geometry:
#   TrajectoryType.CIRCULAR   — circular ASV, piecewise-linear ROV (ideal for ESKF)
#   TrajectoryType.FIGURE_8   — lemniscate ASV, curved ROV (better bearing geometry)
#   TrajectoryType.SINUSOIDAL — S-curve ASV, maneuvering ROV (hardest for ESKF)
#
TRAJECTORY_TYPE = TrajectoryType.FIGURE_8

#
# Measurement realism settings (all False/0 = ideal, synchronous measurements):
#   ACOUSTIC_DELAY — shift reception timestamp by one-way TOF = range/SOUND_SPEED
#                    (the key scenario where FGO's delayed-measurement handling helps)
#   JITTER_STD     — Gaussian timing jitter std [s] on acoustic reception
#   MISS_PROB      — probability of dropping each acoustic measurement [0, 1]
#   SOUND_SPEED    — acoustic propagation speed [m/s]
#
ACOUSTIC_DELAY = True
JITTER_STD     = 0.0    # ±500 ms 1-sigma
MISS_PROB      = 0.0    # 10 % dropout
SOUND_SPEED    = 1500.0  # m/s
TDMA_FREQ      = 0.2     # 1 measurement every 5 seconds on average

true_range = np.sqrt(269)

def main():
    # cv_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

    # for cv in cv_values:
    #     print(f"Running simulations with ROV CV velocity std = {cv} m/s")
    #     eskf_sim.modelCvRov.sigma_a = cv
    #     run_simulations_s1(
    #         TRAJECTORY_TYPE=TRAJECTORY_TYPE,
    #         ACOUSTIC_DELAY=ACOUSTIC_DELAY,
    #         JITTER_STD=JITTER_STD,
    #         MISS_PROB=MISS_PROB,
    #         SOUND_SPEED=SOUND_SPEED,
    #         TDMA_FREQ=TDMA_FREQ,
    #         SAVE_DIR=f"results/exp1_cv_tuning/cv_{cv:.3f}",
    #         ESKF_SIM=eskf_sim,
    #         INIT_FROM_GT=True,
    #     )
    eskf_sim.modelCvRov.sigma_a = 0.02
    # for range_scale in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]:
    # #for range_scale in [0.25]:
    #     print(f"Running simulations with initial range guess = {range_scale*100:.1f} % of true range")
    plotter = run_simulations_s1(
        TRAJECTORY_TYPE=TRAJECTORY_TYPE,
        ACOUSTIC_DELAY=ACOUSTIC_DELAY,
        JITTER_STD=JITTER_STD,
        MISS_PROB=MISS_PROB,
        SOUND_SPEED=SOUND_SPEED,
        TDMA_FREQ=TDMA_FREQ,
        SAVE_DIR=f"results_exp1_ne0es",
        ESKF_SIM=eskf_sim,
        INIT_FROM_GT=False,
        INITIAL_RANGE_GUESS=true_range * 1.0,
    )
    plotter.show()

if __name__ == '__main__':
    main()