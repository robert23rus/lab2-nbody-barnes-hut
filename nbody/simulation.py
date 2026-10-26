"""Начальные условия, leapfrog и независимые физические диагностики."""

from time import perf_counter

import numpy as np

from .forces import _validate, barnes_hut_accelerations, direct_accelerations


def validate_state(positions, masses, velocities, *, softening=0.03,
                   gravitational_constant=1.0):
    """Проверить и скопировать все три массива до начала расчёта/записи."""
    x, m, _, _ = _validate(np.array(positions, dtype=np.float64, copy=True),
                           np.array(masses, dtype=np.float64, copy=True),
                           gravitational_constant, softening)
    v = np.array(velocities, dtype=np.float64, copy=True)
    if v.shape != x.shape or not np.all(np.isfinite(v)):
        raise ValueError("Скорости должны быть конечным массивом формы (N, 2).")
    return x, m, v


def invariants(positions, masses, velocities, *, softening=0.03,
               gravitational_constant=1.0):
    """Точные энергия, импульс и момент импульса для смягчённой модели.

    Потенциальная энергия считается по всем парам, без приближения деревом.
    Временная сложность O(N²), дополнительная память O(N).
    """
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    potential = 0.0
    for i in range(len(masses) - 1):
        delta = positions[i + 1:] - positions[i]
        distances = np.sqrt(np.sum(delta**2, axis=1) + softening**2)
        if np.any(distances == 0):
            raise ValueError("Совпавшие частицы при softening=0: сила сингулярна.")
        potential -= gravitational_constant * masses[i] * np.sum(
            masses[i + 1:] / distances
        )
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    angular = np.sum(masses * (
        positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0]
    ))
    return {"kinetic": float(kinetic), "potential": float(potential),
            "energy": float(kinetic + potential), "momentum": momentum,
            "angular_momentum": float(angular)}


def initial_conditions(n, seed=42, scenario="disk", softening=0.03,
                       gravitational_constant=1.0):
    """Безразмерные условия: суммарная масса 1; размер порядка 1.

    disk: равномерный по площади диск с вращением и малым шумом скорости.
    clusters: два гауссовых скопления со встречным движением.
    binary: точная круговая орбита двух равных масс для данной epsilon.
    В первых двух случаях скорости масштабируются до 2T/|U| = 1.
    Это начальная вириализация, а не доказательство равновесия системы.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n должно быть положительным целым числом.")
    if not np.isfinite(softening) or softening < 0:
        raise ValueError("softening должно быть конечным и неотрицательным.")
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("G должно быть конечным положительным числом.")
    if scenario not in {"disk", "clusters", "binary"}:
        raise ValueError("Неизвестный сценарий начальных условий.")
    masses = np.full(n, 1.0 / n)
    if scenario == "binary":
        if n != 2:
            raise ValueError("Для сценария binary необходимо n=2.")
        positions = np.array([[-0.5, 0.0], [0.5, 0.0]])
        omega = np.sqrt(gravitational_constant / (1.0 + softening**2)**1.5)
        velocities = np.array([[0.0, -0.5 * omega], [0.0, 0.5 * omega]])
        return positions, masses, velocities
    rng = np.random.default_rng(seed)
    if scenario == "disk":
        radii = np.sqrt(rng.uniform(size=n))
        angles = rng.uniform(0, 2 * np.pi, size=n)
        positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        velocities = np.column_stack((-positions[:, 1], positions[:, 0]))
        velocities += rng.normal(0, 0.05, (n, 2))
    else:
        signs = np.where(np.arange(n) < n // 2, -1.0, 1.0)
        positions = rng.normal(0, 0.16, (n, 2))
        positions[:, 0] += 0.7 * signs
        velocities = rng.normal(0, 0.07, (n, 2))
        velocities[:, 1] += 0.3 * signs
    positions -= np.average(positions, axis=0, weights=masses)
    velocities -= np.average(velocities, axis=0, weights=masses)
    diag = invariants(positions, masses, velocities, softening=softening,
                      gravitational_constant=gravitational_constant)
    if diag["kinetic"] > 0:
        velocities *= np.sqrt(abs(diag["potential"]) / (2 * diag["kinetic"]))
    return positions, masses, velocities


def simulate(positions, masses, velocities, *, method="barnes-hut", steps=200,
             dt=0.01, theta=0.5, softening=0.03, gravitational_constant=1.0,
             sample_every=10):
    """Схема kick–drift–kick (leapfrog), скорости в целые моменты времени.

    Дерево перестраивается при каждом новом расчёте ускорения. Результат
    содержит начальное состояние, отсчёты с шагом sample_every и финал.
    Входные массивы не изменяются.
    """
    if method not in {"direct", "barnes-hut"}:
        raise ValueError("Метод должен быть direct или barnes-hut.")
    for name, value in (("steps", steps), ("sample_every", sample_every)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} должно быть положительным целым числом.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt должно быть конечным положительным числом.")
    x, m, v = validate_state(positions, masses, velocities, softening=softening,
                             gravitational_constant=gravitational_constant)
    solver = direct_accelerations if method == "direct" else barnes_hut_accelerations
    kwargs = {"softening": softening, "gravitational_constant": gravitational_constant}
    if method == "barnes-hut":
        kwargs["theta"] = theta
    start = perf_counter()
    force_start = perf_counter()
    acceleration = solver(x, m, **kwargs)  # Также проверяет координаты и массы.
    force_seconds = perf_counter() - force_start
    history = {key: [] for key in (
        "times", "positions", "velocities", "energy", "momentum", "angular_momentum"
    )}

    def save(step):
        diag = invariants(x, m, v, softening=softening,
                          gravitational_constant=gravitational_constant)
        history["times"].append(step * dt)
        history["positions"].append(x.copy())
        history["velocities"].append(v.copy())
        for key in ("energy", "momentum", "angular_momentum"):
            history[key].append(diag[key])
        return diag

    initial = save(0)
    for step in range(1, steps + 1):
        v += 0.5 * dt * acceleration
        x += dt * v
        force_start = perf_counter()
        acceleration = solver(x, m, **kwargs)
        force_seconds += perf_counter() - force_start
        v += 0.5 * dt * acceleration
        if not np.all(np.isfinite(v)):
            raise ValueError("Скорости переполнились; уменьшите dt или увеличьте epsilon.")
        if step % sample_every == 0 or step == steps:
            save(step)
    result = {key: np.asarray(value) for key, value in history.items()}
    result.update(masses=m, method=method, dt=dt, theta=theta,
                  softening=softening, gravitational_constant=gravitational_constant,
                  force_seconds=force_seconds, elapsed_seconds=perf_counter() - start,
                  energy_scale=max(abs(initial["energy"]),
                                   1e-15 * (initial["kinetic"] + abs(initial["potential"])),
                                   np.finfo(float).tiny))
    return result
