"""Новые начальные условия: физические свойства и доступность через CLI."""

import numpy as np
import pytest

from nbody.cli import build_parser
from nbody.simulation import initial_conditions, simulate


@pytest.mark.parametrize("n", [2, 3, 257, 512])
def test_galaxies_are_finite_and_centered_with_equal_particle_masses(n):
    positions, masses, velocities = initial_conditions(n, scenario="galaxies")
    assert positions.shape == velocities.shape == (n, 2)
    assert masses.shape == (n,)
    assert np.isfinite(positions).all() and np.isfinite(velocities).all()
    np.testing.assert_allclose(masses, np.full(n, 1 / n), rtol=0, atol=0)
    np.testing.assert_allclose(masses.sum(), 1.0, rtol=1e-14)
    np.testing.assert_allclose(np.sum(masses[:, None] * positions, axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(np.sum(masses[:, None] * velocities, axis=0), 0, atol=1e-14)


def test_galaxy_groups_rotate_and_approach_on_an_orbit_with_angular_momentum():
    x, m, v = initial_conditions(512, seed=53, scenario="galaxies")
    centers, bulk_velocities = [], []
    for group in (slice(0, 256), slice(256, 512)):
        center = np.average(x[group], axis=0, weights=m[group])
        bulk = np.average(v[group], axis=0, weights=m[group])
        local_x, local_v = x[group] - center, v[group] - bulk
        internal_angular_momentum = np.sum(
            m[group] * (local_x[:, 0] * local_v[:, 1] - local_x[:, 1] * local_v[:, 0])
        )
        assert internal_angular_momentum > 0.01
        assert np.max(np.linalg.norm(local_x, axis=1)) < 0.85
        centers.append(center)
        bulk_velocities.append(bulk)
    separation = centers[1] - centers[0]
    relative_velocity = bulk_velocities[1] - bulk_velocities[0]
    assert np.dot(separation, relative_velocity) < 0
    orbital_cross = separation[0] * relative_velocity[1] - separation[1] * relative_velocity[0]
    assert abs(orbital_cross) > 0.1


def test_galaxies_are_reproducible_but_different_seeds_change_the_sample():
    first = initial_conditions(73, seed=31, scenario="galaxies")
    repeated = initial_conditions(73, seed=31, scenario="galaxies")
    changed = initial_conditions(73, seed=32, scenario="galaxies")
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], changed[0])


def test_galaxies_require_at_least_two_particles():
    with pytest.raises(ValueError, match="n>=2"):
        initial_conditions(1, scenario="galaxies")


def test_new_galaxy_state_can_be_integrated_by_the_existing_force_model():
    x, m, v = initial_conditions(37, seed=7, scenario="galaxies")
    calculation = simulate(x, m, v, method="barnes-hut", theta=0.6,
                            steps=12, dt=0.005, sample_every=4)
    assert calculation["positions"].shape == (4, 37, 2)
    assert np.isfinite(calculation["energy"]).all()
    assert np.linalg.norm(calculation["positions"][-1] - x) > 0


def test_cli_exposes_the_visual_options_and_galaxy_scenario():
    args = build_parser().parse_args(
        ["simulate", "--scenario", "galaxies", "--theme", "paper", "--fps", "20"]
    )
    assert args.scenario == "galaxies"
    assert args.theme == "paper"
    assert args.fps == 20
    defaults = build_parser().parse_args(["simulate"])
    assert defaults.theme == "cosmic" and defaults.fps == 24


@pytest.mark.parametrize("fps", ["0", "-1", "51"])
def test_cli_rejects_gif_frame_rates_outside_the_supported_range(fps):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["simulate", "--fps", fps])
