import numpy as np

from src.utils.geometry import rotation_matrix, to_global_coords, to_relative_coords, wrap_angle


def test_rotation_matrix_rotates_unit_x_to_unit_y() -> None:
    rotated = rotation_matrix(np.pi / 2.0) @ np.array([1.0, 0.0], dtype=np.float32)

    assert np.allclose(rotated, np.array([0.0, 1.0]), atol=1e-6)


def test_to_relative_coords_applies_translation() -> None:
    positions = np.array([[2.0, 3.0], [4.0, 6.0]], dtype=np.float32)
    origin = np.array([2.0, 3.0], dtype=np.float32)

    rel = to_relative_coords(positions, origin, theta=0.0)

    assert np.allclose(rel, np.array([[0.0, 0.0], [2.0, 3.0]], dtype=np.float32))


def test_to_relative_coords_applies_heading_rotation() -> None:
    positions = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    origin = np.array([0.0, 0.0], dtype=np.float32)

    rel = to_relative_coords(positions, origin, theta=np.pi / 2.0)

    assert np.allclose(rel, np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_relative_and_global_transforms_are_inverse() -> None:
    positions = np.array([[5.0, -2.0], [8.0, 1.0], [10.0, 3.0]], dtype=np.float32)
    origin = positions[1]
    theta = 0.7

    rel = to_relative_coords(positions, origin, theta)
    recovered = to_global_coords(rel, origin, theta)

    assert np.allclose(recovered, positions, atol=1e-5)


def test_last_observation_becomes_relative_origin() -> None:
    positions = np.array([[1.0, 1.0], [2.0, 1.5], [4.0, 3.0]], dtype=np.float32)
    origin = positions[-1]

    rel = to_relative_coords(positions, origin, theta=0.3)

    assert np.allclose(rel[-1], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_wrap_angle_scalar_and_array() -> None:
    assert np.isclose(wrap_angle(3.0 * np.pi), -np.pi)

    wrapped = wrap_angle(np.array([-3.5 * np.pi, 0.0, 3.5 * np.pi], dtype=np.float32))

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert np.allclose(wrapped, np.array([0.5 * np.pi, 0.0, -0.5 * np.pi], dtype=np.float32))
