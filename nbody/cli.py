"""Пользовательский интерфейс лабораторной работы."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .forces import BACKEND, warm_up
from .plots import save_simulations
from .simulation import initial_conditions, simulate, validate_state


def _positive_int(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("Требуется целое число >= 1.")
    return number


def _positive_float(text):
    number = float(text)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Требуется конечное число > 0.")
    return number


def _nonnegative_float(text):
    number = float(text)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Требуется конечное число >= 0.")
    return number


def _physical_options(parser):
    parser.add_argument("--softening", type=_nonnegative_float, default=0.03, help="Смягчение ε (по умолчанию 0.03)")
    parser.add_argument("--g", type=_positive_float, default=1.0, help="Гравитационная постоянная (по умолчанию 1)")


def _simulation_options(parser):
    parser.add_argument("--n", type=_positive_int, default=256, help="Количество частиц")
    parser.add_argument("--method", choices=["direct", "barnes-hut", "both"], default="both")
    parser.add_argument("--scenario", choices=["disk", "clusters", "binary"], default="disk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=_positive_int, default=300)
    parser.add_argument("--dt", type=_positive_float, default=0.003)
    parser.add_argument("--theta", type=_nonnegative_float, default=0.5)
    parser.add_argument("--sample-every", type=_positive_int, default=5, help="Шаг сохранения снимков и диагностик")
    parser.add_argument("--input", type=Path, help="NPZ с positions(N,2), masses(N), velocities(N,2); заменяет генератор")
    parser.add_argument("--animate", action="store_true", help="Дополнительно создать animation.gif")
    _physical_options(parser)


def build_parser():
    parser = argparse.ArgumentParser(description="ЛР № 2. N тел: прямой метод и алгоритм Барнса–Хата.")
    commands = parser.add_subparsers(dest="command")
    sim = commands.add_parser("simulate", help="Движение частиц и законы сохранения")
    _simulation_options(sim)
    sim.add_argument("--out", type=Path, default=Path("results/simulation"))
    bench = commands.add_parser("benchmark", help="Скорость и ошибка сил в зависимости от N и θ")
    bench.add_argument("--sizes", nargs="+", type=_positive_int, default=[128, 256, 512, 1024, 2048, 4096, 8192])
    bench.add_argument("--thetas", nargs="+", type=_nonnegative_float, default=[0.2, 0.5, 0.8, 1.0])
    bench.add_argument("--repeats", type=_positive_int, default=3)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--scenario", choices=["disk", "clusters"], default="disk")
    bench.add_argument("--out", type=Path, default=Path("results/benchmark"))
    _physical_options(bench)
    validate = commands.add_parser("validate", help="Проверка круговой орбиты и порядка интегратора")
    validate.add_argument("--out", type=Path, default=Path("results/validation"))
    _physical_options(validate)
    demo = commands.add_parser("demo", help="Все эксперименты с готовыми параметрами; действие по умолчанию")
    demo.add_argument("--out", type=Path, default=Path("results/demo"))
    demo.add_argument("--animate", action="store_true")
    return parser


def run_simulation(args):
    if args.input:
        data = np.load(args.input, allow_pickle=False)
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError("--input ожидает NPZ с positions, masses и velocities.")
        with data:
            x, masses, v = (data[key] for key in ("positions", "masses", "velocities"))
    else:
        x, masses, v = initial_conditions(args.n, seed=args.seed, scenario=args.scenario,
                                          softening=args.softening, gravitational_constant=args.g)
    x, masses, v = validate_state(x, masses, v, softening=args.softening,
                                  gravitational_constant=args.g)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "initial_state.npz", positions=x, masses=masses, velocities=v)
    methods = ["direct", "barnes-hut"] if args.method == "both" else [args.method]
    results = []
    for method in methods:
        print(f"Моделирование: {method}, N={len(masses)}, шагов={args.steps}...", flush=True)
        result = simulate(x, masses, v, method=method, steps=args.steps, dt=args.dt,
                          theta=args.theta, softening=args.softening, gravitational_constant=args.g,
                          sample_every=args.sample_every)
        results.append(result)
    summary = save_simulations(results, args.out, animate=args.animate)
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    config["backend"] = BACKEND
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for method in methods:
        row = summary[method]
        print(f"  {method}: расчёт {row['elapsed_seconds']:.3f} с; "
              f"max |ΔE/E₀| = {row['max_abs_relative_energy_drift']:.3e}")
    return summary


def run_validation(out, *, softening=0.03, gravitational_constant=1.0):
    out.mkdir(parents=True, exist_ok=True)
    x, masses, v = initial_conditions(2, scenario="binary", softening=softening,
                                      gravitational_constant=gravitational_constant)
    omega = np.sqrt(gravitational_constant / (1 + softening**2)**1.5)
    period = 2 * np.pi / omega
    result = simulate(x, masses, v, method="direct", steps=1000, dt=period / 1000,
                      softening=softening, gravitational_constant=gravitational_constant,
                      sample_every=5)
    summary = save_simulations([result], out)
    distance = np.linalg.norm(result["positions"][:, 1] - result["positions"][:, 0], axis=1)
    final_time = 2.0 / omega
    angle = omega * final_time
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    exact = x @ rotation.T
    errors = []
    for steps in (100, 200):
        calculation = simulate(x, masses, v, method="direct", steps=steps, dt=final_time / steps,
                               softening=softening, gravitational_constant=gravitational_constant,
                               sample_every=steps)
        errors.append(float(np.linalg.norm(calculation["positions"][-1] - exact)))
    ratio = errors[0] / errors[1]
    values = {"period": float(period), "max_distance_error": float(np.max(np.abs(distance - 1))),
              "max_abs_relative_energy_drift": summary["direct"]["max_abs_relative_energy_drift"],
              "coordinate_error_dt": errors[0], "coordinate_error_dt_over_2": errors[1],
              "convergence_ratio": ratio, "expected_ratio": 4.0,
              "passed": bool(np.max(np.abs(distance - 1)) < 1e-3
                             and summary["direct"]["max_abs_relative_energy_drift"] < 1e-5
                             and 3.5 < ratio < 4.5)}
    (out / "validation.json").write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Круговая орбита: max |r−1|={values['max_distance_error']:.3e}; "
          f"ошибка(dt)/ошибка(dt/2)={ratio:.3f}; проверка {'пройдена' if values['passed'] else 'НЕ пройдена'}.")
    return values["passed"]


def main(argv=None):
    # Windows при перенаправлении вывода выбирает cp1251, где нет θ и ε.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments if arguments else ["demo"])
    print(f"Вычислительный режим: {BACKEND}.", flush=True)
    if BACKEND == "python":
        print("Для быстрых экспериментов запустите программу в .venv с зависимостями из requirements.txt.", flush=True)
    try:
        warm_up()
        if args.command == "simulate":
            run_simulation(args)
        elif args.command == "benchmark":
            from .experiments import benchmark
            benchmark(args.out, sizes=args.sizes, thetas=args.thetas, repeats=args.repeats,
                      seed=args.seed, scenario=args.scenario, softening=args.softening,
                      gravitational_constant=args.g)
        elif args.command == "validate":
            if not run_validation(args.out, softening=args.softening, gravitational_constant=args.g):
                return 1
        elif args.command == "demo":
            from .experiments import benchmark
            benchmark(args.out / "benchmark")
            sim_args = parser.parse_args(["simulate", "--out", str(args.out / "simulation")])
            sim_args.animate = args.animate
            run_simulation(sim_args)
            if not run_validation(args.out / "validation"):
                return 1
        print(f"Результаты: {args.out.resolve()}")
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
