"""Регрессии пользовательского ввода и независимые проверки метрик сил."""

import json
import warnings

import numpy as np
import pytest

from nbody import cli
from nbody.experiments import force_error_metrics


@pytest.mark.parametrize("invalid_input", ["scalar_mass_npz", "npy"])
def test_invalid_input_returns_an_error_before_writing_results(
    tmp_path, monkeypatch, invalid_input
):
    monkeypatch.setattr(cli, "warm_up", lambda: None)
    if invalid_input == "scalar_mass_npz":
        source = tmp_path / "invalid.npz"
        np.savez(
            source,
            positions=np.zeros((1, 2)),
            masses=np.array(1.0),
            velocities=np.zeros((1, 2)),
        )
    else:
        source = tmp_path / "array.npy"
        np.save(source, np.zeros((2, 2)))
    destination = tmp_path / "results"

    status = cli.main(
        ["simulate", "--input", str(source), "--out", str(destination), "--steps", "1"]
    )

    assert status == 2
    assert not (destination / "initial_state.npz").exists()
    assert not (destination / "simulation_summary.json").exists()


def test_custom_initial_state_with_unequal_masses_is_preserved_in_outputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "warm_up", lambda: None)
    positions = np.array([[-2.0, 0.0], [1.0, 0.0]])
    masses = np.array([2.0, 5.0])
    velocities = np.array([[0.1, -0.2], [-0.04, 0.08]])
    source = tmp_path / "custom.npz"
    np.savez(source, positions=positions, masses=masses, velocities=velocities)
    destination = tmp_path / "results"
    dt, gravitational_constant, softening = 0.005, 1.7, 0.04

    status = cli.main(
        [
            "simulate", "--input", str(source), "--out", str(destination),
            "--method", "both", "--steps", "2", "--sample-every", "1",
            "--dt", str(dt), "--g", str(gravitational_constant),
            "--softening", str(softening), "--n", "999", "--scenario", "binary",
        ]
    )

    assert status == 0
    with np.load(destination / "initial_state.npz", allow_pickle=False) as initial:
        np.testing.assert_array_equal(initial["positions"], positions)
        np.testing.assert_array_equal(initial["masses"], masses)
        np.testing.assert_array_equal(initial["velocities"], velocities)

    # Первый drift leapfrog проверяется по аналитическому ускорению пары.
    factor = gravitational_constant * 3.0 / (9.0 + softening**2) ** 1.5
    acceleration = np.array([[5.0 * factor, 0.0], [-2.0 * factor, 0.0]])
    expected_first_positions = positions + dt * velocities + 0.5 * dt**2 * acceleration
    for method in ("direct", "barnes-hut"):
        with np.load(destination / f"trajectory_{method}.npz", allow_pickle=False) as history:
            np.testing.assert_array_equal(history["masses"], masses)
            np.testing.assert_array_equal(history["positions"][0], positions)
            np.testing.assert_array_equal(history["velocities"][0], velocities)
            np.testing.assert_allclose(history["times"], [0.0, dt, 2 * dt])
            np.testing.assert_allclose(history["positions"][1], expected_first_positions)
            assert history["positions"].shape == (3, 2, 2)
            assert np.isfinite(history["energy"]).all()
            assert float(history["gravitational_constant"]) == gravitational_constant
            assert float(history["softening"]) == softening
    summary = json.loads((destination / "simulation_summary.json").read_text(encoding="utf-8"))
    assert summary["direct"]["n"] == summary["barnes-hut"]["n"] == 2
    assert summary["final_position_rms_difference"] < 1e-12


def test_force_metrics_include_unequal_particle_masses():
    # Эталонные силы равны (1,0), (3,0); ошибка сил — (1,0), (0,0).
    result = force_error_metrics(
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[2.0, 0.0], [1.0, 0.0]]),
        np.array([1.0, 3.0]),
    )

    assert result["relative_l2"] == pytest.approx(1 / np.sqrt(10))
    assert result["absolute_rms"] == pytest.approx(1 / np.sqrt(2))
    assert result["mean_relative"] == pytest.approx(0.5)
    assert result["p95_relative"] == pytest.approx(0.95)
    assert result["max_relative"] == pytest.approx(1.0)


def test_force_metrics_of_identical_zero_forces_are_zero():
    result = force_error_metrics(np.zeros((3, 2)), np.zeros((3, 2)), np.array([1.0, 2.0, 7.0]))

    assert all(value == 0.0 for value in result.values())


def test_large_finite_forces_keep_finite_relative_errors_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = force_error_metrics(
            np.array([[1e308, 0.0]]), np.array([[-1e308, 0.0]]), np.array([1.0])
        )

    for metric in ("relative_l2", "mean_relative", "p95_relative", "max_relative"):
        assert result[metric] == pytest.approx(2.0)
    # Абсолютная ошибка 2e308 превышает диапазон float64.
    assert np.isinf(result["absolute_rms"])
