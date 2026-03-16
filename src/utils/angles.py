import numpy as np

TAU = 2.0 * np.pi

def wrap_to_2pi(angle_rad: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    return angle_rad % TAU

def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle_rad + np.pi) % TAU - np.pi