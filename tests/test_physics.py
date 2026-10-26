"""Проверки модели на аналитических решениях и физических инвариантах."""

import numpy as np
import pytest

from nbody.forces import barnes_hut_accelerations, direct_accelerations
from nbody.simulation import initial_conditions, invariants, simulate


FORCE_METHODS = (direct_accelerations, barnes_hut_accelerations)


@pytest.mark.parametrize("force", FORCE_METHODS)
def test_two_unequal_masses_against_analytic_force(force):
    """Сила направлена к соседу, ускорение зависит от массы соседа."""
    positions = np.array([[-1.0, 0.0], [2.0, 0.0]])
    masses = np.array([2.0, 5.0])
    gravitational_constant, softening = 3.0, 0.4
    factor = gravitational_constant * 3.0 / (9.0 + softening**2) ** 1.5
    expected = np.array([[5.0 * factor, 0.0], [-2.0 * factor, 0.0]])

    actual = force(
        positions,
        masses,
        gravitational_constant=gravitational_constant,
        softening=softening,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_direct_method_respects_newtons_third_law():
    rng = np.random.default_rng(173)
    positions = rng.normal(size=(47, 2))
    masses = rng.uniform(0.1, 3.0, len(positions))
    acceleration = direct_accelerations(positions, masses, softening=0.07)
    forces = masses[:, None] * acceleration
    residual = np.linalg.norm(forces.sum(axis=0))
    force_scale = np.linalg.norm(forces, axis=1).sum()

    assert residual <= 2e-14 * force_scale


@pytest.mark.parametrize("leaf_capacity,max_depth", [(1, 48), (6, 48), (1, 2)])
def test_zero_theta_reproduces_direct_sum(leaf_capacity, max_depth):
    """Без группировки квадродерево должно давать точную сумму сил."""
    rng = np.random.default_rng(76)
    positions = rng.normal(size=(83, 2))
    masses = rng.lognormal(mean=0.0, sigma=1.0, size=len(positions))
    expected = direct_accelerations(positions, masses, softening=0.04)
    actual = barnes_hut_accelerations(
        positions,
        masses,
        theta=0.0,
        softening=0.04,
        leaf_capacity=leaf_capacity,
        max_depth=max_depth,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("force", FORCE_METHODS)
def test_isolated_particle_has_no_self_acceleration(force):
    acceleration = force(np.array([[7.0, -2.0]]), np.array([13.0]), softening=0.0)
    np.testing.assert_array_equal(acceleration, np.zeros((1, 2)))


def test_target_containing_branch_is_opened_even_for_large_theta():
    """Большая собственная масса не должна попасть в агрегированную силу."""
    positions = np.array([[0.0, 0.0], [0.1, 0.0], [100.0, 0.0]])
    masses = np.array([1e9, 1.0, 1.0])
    exact = direct_accelerations(positions, masses, softening=0.01)
    approximate = barnes_hut_accelerations(
        positions, masses, theta=1e6, softening=0.01
    )

    # Для двух близких тел всё дерево, содержащее цель, раскрывается до листьев.
    np.testing.assert_allclose(approximate[:2], exact[:2], rtol=1e-13, atol=1e-12)


def test_coincident_particles_with_softening_are_finite_and_exact():
    positions = np.array([[0.0, 0.0]] * 12 + [[1.0, -0.5]])
    masses = np.linspace(0.2, 2.0, len(positions))
    expected = direct_accelerations(positions, masses, softening=0.1)
    actual = barnes_hut_accelerations(
        positions, masses, theta=0.0, softening=0.1, max_depth=12
    )

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("force", FORCE_METHODS)
def test_unsoftened_coincident_particles_are_rejected(force):
    with pytest.raises(ValueError):
        force(np.zeros((2, 2)), np.ones(2), softening=0.0)


@pytest.mark.parametrize("force", FORCE_METHODS)
@pytest.mark.parametrize(
    "positions,masses,parameters",
    [
        (np.empty((0, 2)), np.empty(0), {}),
        (np.zeros((2, 3)), np.ones(2), {}),
        (np.zeros((2, 2)), np.ones(1), {}),
        (np.array([[np.nan, 0.0]]), np.ones(1), {}),
        (np.zeros((1, 2)), np.array([np.inf]), {}),
        (np.zeros((1, 2)), np.array([-1.0]), {}),
        (np.zeros((1, 2)), np.array([0.0]), {}),
        (np.zeros((1, 2)), np.ones(1), {"softening": -0.1}),
        (np.zeros((1, 2)), np.ones(1), {"softening": np.nan}),
        (np.zeros((1, 2)), np.ones(1), {"gravitational_constant": -1.0}),
    ],
)
def test_invalid_physical_inputs_are_rejected(force, positions, masses, parameters):
    with pytest.raises(ValueError):
        force(positions, masses, **parameters)


@pytest.mark.parametrize("theta", [-0.1, np.nan, np.inf])
def test_invalid_opening_angle_is_rejected(theta):
    with pytest.raises(ValueError):
        barnes_hut_accelerations(np.zeros((1, 2)), np.ones(1), theta=theta)


def test_binary_initial_state_is_a_softened_circular_orbit():
    gravitational_constant, softening = 1.7, 0.08
    positions, masses, velocities = initial_conditions(
        2,
        scenario="binary",
        softening=softening,
        gravitational_constant=gravitational_constant,
    )
    omega = np.sqrt(gravitational_constant / (1.0 + softening**2) ** 1.5)

    np.testing.assert_allclose(masses, [0.5, 0.5])
    np.testing.assert_allclose(np.linalg.norm(positions[1] - positions[0]), 1.0)
    np.testing.assert_allclose((masses[:, None] * positions).sum(axis=0), 0.0)
    np.testing.assert_allclose((masses[:, None] * velocities).sum(axis=0), 0.0)
    np.testing.assert_allclose(np.linalg.norm(velocities, axis=1), 0.5 * omega)
    np.testing.assert_allclose(np.sum(positions * velocities, axis=1), 0.0)


def test_energy_diagnostic_matches_the_analytic_softened_pair():
    positions = np.array([[-0.5, 0.0], [0.5, 0.0]])
    masses = np.array([2.0, 3.0])
    velocities = np.array([[0.0, -0.3], [0.0, 0.2]])
    state = invariants(
        positions, masses, velocities, gravitational_constant=1.4, softening=0.2
    )
    expected_energy = 0.5 * (2 * 0.3**2 + 3 * 0.2**2) - 1.4 * 6 / np.sqrt(1.04)

    np.testing.assert_allclose(state["energy"], expected_energy, rtol=1e-14)
    np.testing.assert_allclose(state["momentum"], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(state["angular_momentum"], 0.6, rtol=1e-14)


@pytest.mark.parametrize("method", ["direct", "barnes-hut"])
def test_leapfrog_keeps_a_binary_orbit_and_invariants(method):
    softening = 0.03
    positions, masses, velocities = initial_conditions(2, scenario="binary", softening=softening)
    omega = np.sqrt(1.0 / (1.0 + softening**2) ** 1.5)
    period = 2 * np.pi / omega
    result = simulate(
        positions,
        masses,
        velocities,
        method=method,
        steps=600,
        dt=period / 600,
        softening=softening,
        sample_every=13,
    )
    radius = np.linalg.norm(result["positions"][:, 1] - result["positions"][:, 0], axis=1)
    energy = np.asarray(result["energy"])

    assert abs(result["times"][-1] - period) < 1e-12
    assert np.max(np.abs(radius - 1.0)) < 2e-4
    assert np.max(np.abs(energy - energy[0])) / abs(energy[0]) < 1e-6
    np.testing.assert_allclose(result["momentum"], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result["angular_momentum"], result["angular_momentum"][0], rtol=1e-12
    )


def test_leapfrog_has_second_order_global_convergence():
    """Погрешность фазы круговой орбиты при dt/2 уменьшается примерно в 4 раза."""
    softening = 0.03
    positions, masses, velocities = initial_conditions(2, scenario="binary", softening=softening)
    omega = np.sqrt(1.0 / (1.0 + softening**2) ** 1.5)
    final_time = 2.0
    angle = omega * final_time
    direction = np.sign(
        positions[0, 0] * velocities[0, 1] - positions[0, 1] * velocities[0, 0]
    )
    rotation = np.array(
        [[np.cos(angle), -direction * np.sin(angle)],
         [direction * np.sin(angle), np.cos(angle)]]
    )
    exact_positions = positions @ rotation.T
    errors = []
    for steps in (40, 80):
        result = simulate(
            positions,
            masses,
            velocities,
            method="direct",
            steps=steps,
            dt=final_time / steps,
            softening=softening,
            sample_every=steps,
        )
        errors.append(np.linalg.norm(result["positions"][-1] - exact_positions))

    assert 3.5 < errors[0] / errors[1] < 4.5


def test_simulation_preserves_input_arrays_and_samples_the_final_step():
    positions, masses, velocities = initial_conditions(2, scenario="binary")
    original = [array.copy() for array in (positions, masses, velocities)]
    result = simulate(
        positions, masses, velocities, method="direct", steps=17, dt=0.01, sample_every=6
    )

    for actual, expected in zip((positions, masses, velocities), original):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(result["times"], [0.0, 0.06, 0.12, 0.17])
    assert np.asarray(result["positions"]).shape == (4, 2, 2)
