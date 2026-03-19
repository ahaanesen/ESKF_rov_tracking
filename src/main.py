from tracking_only.run_simulations import run_simulations as run_simulations_tracking
from tracking_and_navigation.run_simulations import run_simulations as run_simulations_nav
from config import RUN


def main():
    if RUN == 'tracking_only':
        run_simulations_tracking()

    elif RUN == 'tracking_and_navigation':
        run_simulations_nav()


if __name__ == '__main__':
    main()
