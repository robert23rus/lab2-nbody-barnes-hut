"""Воспроизводимое сравнение времени и погрешности двух расчётов сил."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

import numpy as np

from .forces import (BACKEND, barnes_hut_accelerations,
                     direct_accelerations, warm_up)
from .simulation import initial_conditions


def force_error_metrics(reference_acceleration, approx_acceleration, masses):
    """Сравнить силы F_i=m_i*a_i на одном и том же снимке частиц.

    relative_l2 = ||F_BH-F_direct||_2 / ||F_direct||_2 по всем компонентам.
    Индивидуальные ошибки используют max(|F_i|, 1e-14*max_j|F_j|, tiny).
    absolute_rms = sqrt(mean_i |F_BH_i-F_direct_i|**2).
    """
    reference = np.asarray(reference_acceleration, dtype=np.float64)
    approximate = np.asarray(approx_acceleration, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[0] == 0
            or reference.shape[1] != 2 or approximate.shape != reference.shape
            or mass.shape != (len(reference),)):
        raise ValueError("Ускорения должны иметь форму (N, 2), массы — (N,), N >= 1.")
    if (not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(approximate))
            or not np.all(np.isfinite(mass)) or np.any(mass <= 0)):
        raise ValueError("Ускорения и массы должны быть конечными; массы > 0.")
    with np.errstate(over="ignore", invalid="ignore"):
        reference_force = reference * mass[:, None]
        approximate_force = approximate * mass[:, None]
    if (not np.all(np.isfinite(reference_force))
            or not np.all(np.isfinite(approximate_force))):
        raise ValueError("Силы вышли за диапазон float64.")
    # Все разности и нормы вычисляются после общего масштабирования.
    # Например, 1e308 - (-1e308) переполняется, хотя относительная ошибка = 2.
    scale = max(float(np.max(np.abs(reference_force))),
                float(np.max(np.abs(approximate_force))))
    if scale == 0:
        individual = np.zeros(len(mass))
        relative_l2 = 0.0
        absolute_rms = 0.0
    else:
        reference_scaled = reference_force / scale
        difference_scaled = approximate_force / scale - reference_scaled
        reference_norm = np.hypot(reference_scaled[:, 0], reference_scaled[:, 1])
        difference_norm = np.hypot(difference_scaled[:, 0], difference_scaled[:, 1])
        floor = max(1e-14 * float(reference_norm.max()), np.finfo(float).tiny / scale)
        denominators = np.maximum(reference_norm, floor)
        individual = np.zeros(len(mass))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(difference_norm, denominators, out=individual,
                      where=denominators > 0)
        individual[(denominators == 0) & (difference_norm > 0)] = np.inf
        numerator = float(np.linalg.norm(difference_scaled))
        denominator = float(np.linalg.norm(reference_scaled))
        relative_l2 = numerator / denominator if denominator else float("inf")
        absolute_rms = scale * (numerator / math.sqrt(len(mass)))
    # np.percentile([inf, inf], 95) даёт NaN из-за inf-inf при интерполяции.
    ordered = np.sort(individual)
    percentile_index = 0.95 * (len(ordered) - 1)
    left = int(math.floor(percentile_index))
    right = int(math.ceil(percentile_index))
    if left == right or not math.isfinite(float(ordered[right])):
        p95 = float(ordered[right])
    else:
        fraction = percentile_index - left
        p95 = float((1 - fraction) * ordered[left] + fraction * ordered[right])
    maximum = float(np.max(individual))
    mean = maximum * float(np.mean(individual / maximum)) if 0 < maximum < math.inf else maximum
    return {
        "relative_l2": relative_l2,
        "mean_relative": mean,
        "p95_relative": p95,
        "max_relative": maximum,
        "absolute_rms": absolute_rms,
    }


def _json_safe(value):
    """JSON остаётся стандартным: бесконечности представлены строками."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, data):
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _slope(rows):
    """Эмпирический наклон на трёх наибольших измеренных N."""
    last = sorted(rows, key=lambda pair: pair[0])[-3:]
    if len(last) < 3:
        return {"n": [int(pair[0]) for pair in last], "log_log_slope": None}
    slope = np.polyfit(np.log([pair[0] for pair in last]),
                       np.log([pair[1] for pair in last]), 1)[0]
    return {"n": [int(pair[0]) for pair in last], "log_log_slope": float(slope)}


def _plot_benchmark(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({row["n"] for row in rows})
    thetas = sorted({row["theta"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_time, ax_speed, ax_error, ax_tradeoff = axes.flat
    direct = [next(row["direct_seconds"] for row in rows if row["n"] == n)
              for n in sizes]
    ax_time.loglog(sizes, direct, "ko-", label="Прямой метод")
    for theta in thetas:
        selected = sorted((row for row in rows if row["theta"] == theta),
                          key=lambda row: row["n"])
        label = f"Барнс–Хат, θ={theta:g}"
        ax_time.loglog([row["n"] for row in selected],
                       [row["barnes_hut_seconds"] for row in selected], "o-", label=label)
        ax_speed.semilogx([row["n"] for row in selected],
                         [row["speedup"] for row in selected], "o-", label=f"θ={theta:g}")
    plot_floor = 1e-16
    for n in sizes:
        selected = sorted((row for row in rows if row["n"] == n),
                          key=lambda row: row["theta"])
        ax_error.semilogy([row["theta"] for row in selected],
                          [max(row["relative_l2"], plot_floor) for row in selected],
                          "o-", label=f"N={n}")
    largest = sorted((row for row in rows if row["n"] == max(sizes)),
                     key=lambda row: row["theta"])
    ax_tradeoff.semilogx([max(row["relative_l2"], plot_floor) for row in largest],
                        [row["speedup"] for row in largest], "o-")
    for row in largest:
        ax_tradeoff.annotate(f"θ={row['theta']:g}",
                            (max(row["relative_l2"], plot_floor), row["speedup"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax_time.set(title="Время расчёта сил (дерево включено)", xlabel="Число частиц N",
                ylabel="Медианное время, с")
    ax_speed.set(title="Ускорение вычислений", xlabel="Число частиц N",
                 ylabel="t прямого / t Барнса–Хата")
    ax_speed.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax_error.set(title="Погрешность сил на одном снимке", xlabel="Порог раскрытия θ",
                 ylabel="Относительная ошибка L2")
    ax_tradeoff.set(title=f"Скорость и точность, N={max(sizes)}",
                    xlabel="Относительная ошибка L2", ylabel="Ускорение вычислений")
    ax_tradeoff.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax_tradeoff.margins(x=0.16, y=0.12)
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.25)
    for ax in (ax_time, ax_speed, ax_error):
        ax.legend(fontsize=8)
    fig.suptitle("Сравнение прямого метода и алгоритма Барнса–Хата\n"
                 "Нулевые ошибки показаны на уровне 10⁻¹⁶", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def benchmark(out_dir, *, sizes=(128, 256, 512, 1024, 2048, 4096, 8192),
              thetas=(0.2, 0.5, 0.8, 1.0), repeats=3, seed=42,
              scenario="disk", softening=0.03, gravitational_constant=1.0):
    """Сохранить CSV, метаданные, графики и сводку измерений; вернуть строки.

    Все времена измеряются perf_counter после прогрева, публикуется медиана.
    BH каждый раз строит новое дерево. Начальные условия, JIT, вычисление
    ошибок и построение графиков не входят во время расчёта сил.
    """
    sizes = tuple(sizes)
    thetas = tuple(thetas)
    if not sizes or any(not isinstance(n, Integral) or isinstance(n, bool) or n < 1
                        for n in sizes):
        raise ValueError("sizes должен содержать положительные целые N.")
    if (not isinstance(repeats, Integral) or isinstance(repeats, bool) or repeats < 1):
        raise ValueError("repeats должен быть положительным целым числом.")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed должен быть неотрицательным целым числом.")
    if not thetas:
        raise ValueError("Нужно хотя бы одно значение theta.")
    try:
        thetas = tuple(float(theta) for theta in thetas)
        softening = float(softening)
        gravitational_constant = float(gravitational_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta, softening и G должны быть числами.") from exc
    if any(not math.isfinite(theta) or theta < 0 for theta in thetas):
        raise ValueError("theta должен быть конечным и >= 0.")
    if not math.isfinite(softening) or softening < 0:
        raise ValueError("softening должен быть конечным и >= 0.")
    if not math.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("G должен быть конечным и > 0.")
    sizes = tuple(sorted(set(int(n) for n in sizes)))
    thetas = tuple(sorted(set(thetas)))
    if scenario not in {"disk", "clusters", "binary"}:
        raise ValueError("Сценарий должен быть disk, clusters или binary.")
    if scenario == "binary" and sizes != (2,):
        raise ValueError("Для сценария binary список sizes должен содержать только 2.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_up()
    force_kwargs = {"softening": softening,
                    "gravitational_constant": gravitational_constant}
    rows = []
    timings = []
    seeds = {}
    for n in sizes:
        # Seed зависит только от базового seed и N, а не от порядка списка sizes.
        state_seed = int(np.random.SeedSequence([int(seed), n]).generate_state(1)[0])
        seeds[str(n)] = state_seed
        positions, masses, _ = initial_conditions(
            n, seed=state_seed, scenario=scenario, **force_kwargs)
        reference = direct_accelerations(positions, masses, **force_kwargs)
        direct_samples = []
        for _ in range(repeats):
            start = perf_counter()
            direct_accelerations(positions, masses, **force_kwargs)
            direct_samples.append(perf_counter() - start)
        direct_seconds = float(median(direct_samples))
        for theta in thetas:
            bh_samples = []
            for _ in range(repeats):
                start = perf_counter()
                approximate = barnes_hut_accelerations(
                    positions, masses, theta=theta, **force_kwargs)
                bh_samples.append(perf_counter() - start)
            bh_seconds = float(median(bh_samples))
            row = {"n": n, "theta": theta,
                   "direct_seconds": direct_seconds,
                   "barnes_hut_seconds": bh_seconds,
                   "speedup": direct_seconds / max(bh_seconds, np.finfo(float).tiny)}
            row.update(force_error_metrics(reference, approximate, masses))
            rows.append(row)
            timings.append({"n": n, "theta": theta,
                            "direct_samples_seconds": direct_samples.copy(),
                            "barnes_hut_samples_seconds": bh_samples})
        print(f"N={n}: прямой метод {direct_seconds:.6f} с; "
              f"Барнс–Хат: " + "; ".join(
                  f"theta={row['theta']:g}, ускорение {row['speedup']:.2f}, "
                  f"ошибка L2 {row['relative_l2']:.3e}"
                  for row in rows if row["n"] == n), flush=True)
    with (out_dir / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "numpy": np.__version__, "matplotlib": _package_version("matplotlib"),
                        "numba": _package_version("numba")},
        "backend": BACKEND,
        "parameters": {"sizes": sizes, "thetas": thetas, "repeats": int(repeats),
                       "seed": int(seed), "scenario": scenario, **force_kwargs,
                       "leaf_capacity": 1, "max_depth": 48},
        "seed_for_each_n": seeds,
        "method": {
            "timer": "time.perf_counter",
            "statistic": "median of repeats",
            "direct": "Each unordered pair is evaluated once; same backend as BH.",
            "barnes_hut": "Each call includes tree construction, traversal and input validation.",
            "excluded_from_timing": ["initial conditions", "JIT compilation", "force error metrics", "plotting"],
            "comparison": "All theta use identical positions and masses at each N; F_i=m_i*a_i.",
            "relative_l2": "||F_BH-F_direct||_2 / ||F_direct||_2",
            "individual_relative": "|delta F_i| / max(|F_direct_i|, 1e-14*max_j|F_direct_j|, float64.tiny)",
            "absolute_rms": "sqrt(mean_i |F_BH_i-F_direct_i|^2)",
            "zero_reference": "Both zero: relative L2=0. Zero reference and nonzero approximation: Infinity.",
            "nonfinite_json": "Nonfinite numbers are explicit strings: Infinity, -Infinity, NaN.",
            "plot_error_floor": 1e-16,
        },
        "raw_timing_samples": timings,
    }
    _write_json(out_dir / "benchmark_metadata.json", metadata)
    summary = {
        "interpretation": "Measured local log-log slopes, not a proof of asymptotic complexity. "
                          "Speedup crossings refer only to the measured sizes and this environment.",
        "direct_fit": _slope([(n, next(row["direct_seconds"] for row in rows if row["n"] == n))
                               for n in sizes]),
        "barnes_hut_by_theta": {},
    }
    for theta in thetas:
        selected = [row for row in rows if row["theta"] == theta]
        result = _slope([(row["n"], row["barnes_hut_seconds"]) for row in selected])
        accelerated = [row["n"] for row in selected if row["speedup"] > 1]
        result["smallest_measured_n_with_speedup_gt_1"] = min(accelerated) if accelerated else None
        result["largest_n"] = selected[-1]
        summary["barnes_hut_by_theta"][f"{theta:g}"] = result
    _write_json(out_dir / "benchmark_summary.json", summary)
    _plot_benchmark(rows, out_dir / "benchmark.png")
    return rows
