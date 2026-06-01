from __future__ import annotations

import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    """Return a 2D counter-clockwise rotation matrix."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=np.float32,
    )


def to_relative_coords(
    positions: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Transform global xy positions into an origin-centered heading frame."""
    positions_array = np.asarray(positions, dtype=np.float32)
    origin_array = np.asarray(origin, dtype=np.float32)
    rotated = (positions_array - origin_array) @ rotation_matrix(-theta).T
    return rotated.astype(np.float32)


def to_global_coords(
    rel_positions: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Transform relative xy positions back into the global frame."""
    rel_array = np.asarray(rel_positions, dtype=np.float32)
    origin_array = np.asarray(origin, dtype=np.float32)
    global_positions = rel_array @ rotation_matrix(theta).T + origin_array
    return global_positions.astype(np.float32)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the [-pi, pi) interval."""
    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped.astype(np.float32)
