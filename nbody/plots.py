"""Сохранение расчёта, графиков и необязательной GIF-анимации."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


LABELS = {"direct": "Прямой метод", "barnes-hut": "Барнс–Хат"}


def save_simulations(results, out_dir, *, animate=False):
    """Записать каждый расчёт в NPZ/CSV и совместно построить графики."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for result in results:
        method = result["method"]
        np.savez_compressed(out_dir / f"trajectory_{method}.npz", **result)
        times = result["times"]
        energy = result["energy"]
        drift = (energy - energy[0]) / result["energy_scale"]
        momentum_drift = np.linalg.norm(result["momentum"] - result["momentum"][0], axis=1)
        angular_drift = result["angular_momentum"] - result["angular_momentum"][0]
        with (out_dir / f"diagnostics_{method}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "energy", "relative_energy_drift", "px", "py",
                             "momentum_drift_norm", "angular_momentum", "angular_momentum_drift"])
            writer.writerows(zip(times, energy, drift, result["momentum"][:, 0],
                                 result["momentum"][:, 1], momentum_drift,
                                 result["angular_momentum"], angular_drift))
        summary[method] = {
            "n": len(result["masses"]), "final_time": float(times[-1]),
            "dt": result["dt"], "theta": result["theta"],
            "softening": result["softening"], "gravitational_constant": result["gravitational_constant"],
            "elapsed_seconds": result["elapsed_seconds"], "force_seconds": result["force_seconds"],
            "max_abs_relative_energy_drift": float(np.max(np.abs(drift))),
            "max_momentum_drift_norm": float(np.max(momentum_drift)),
            "max_abs_angular_momentum_drift": float(np.max(np.abs(angular_drift))),
        }
    if len(results) == 2:
        first, second = results
        difference = second["positions"] - first["positions"]
        rms = np.sqrt(np.average(np.sum(difference**2, axis=2), axis=1, weights=first["masses"]))
        with (out_dir / "trajectory_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "mass_weighted_position_rms_difference"])
            writer.writerows(zip(first["times"], rms))
        summary["final_position_rms_difference"] = float(rms[-1])
    with (out_dir / "simulation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _plot_trajectories(results, out_dir)
    _plot_invariants(results, out_dir)
    if animate:
        _animate(results, out_dir)
    return summary


def _plot_trajectories(results, out_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5), squeeze=False)
    for ax, result in zip(axes[0], results):
        x = result["positions"]
        chosen = np.linspace(0, x.shape[1] - 1, min(24, x.shape[1]), dtype=int)
        for i in chosen:
            ax.plot(x[:, i, 0], x[:, i, 1], linewidth=0.65, alpha=0.6)
        ax.scatter(x[0, :, 0], x[0, :, 1], s=8, c="0.7", alpha=0.5, label="Начало")
        ax.scatter(x[-1, :, 0], x[-1, :, 1], s=10, c="#126aab", label="Конец")
        title = LABELS[result["method"]]
        if result["method"] == "barnes-hut":
            title += f", θ={result['theta']:g}"
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"N={results[0]['positions'].shape[1]}, t={results[0]['times'][-1]:g}; траектории до 24 частиц")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=180)
    plt.close(fig)


def _plot_invariants(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for result in results:
        label = LABELS[result["method"]]
        axes[0].plot(result["times"], (result["energy"] - result["energy"][0]) / result["energy_scale"], label=label)
        axes[1].plot(result["times"], np.linalg.norm(result["momentum"] - result["momentum"][0], axis=1), label=label)
        axes[2].plot(result["times"], result["angular_momentum"] - result["angular_momentum"][0], label=label)
    for ax, title, ylabel in zip(axes, ["Энергия", "Импульс", "Момент импульса"],
                                  ["(E(t) − E(0)) / |E(0)|", "‖P(t) − P(0)‖", "Lz(t) − Lz(0)"]):
        ax.set(title=title, xlabel="t", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "conservation.png", dpi=180)
    plt.close(fig)


def _animate(results, out_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), squeeze=False)
    all_x = np.concatenate([r["positions"].reshape(-1, 2) for r in results])
    low, high = np.min(all_x, axis=0), np.max(all_x, axis=0)
    margin = max(float(np.max(high - low)) * 0.07, 0.05)
    scatters, titles = [], []
    for ax, result in zip(axes[0], results):
        scatters.append(ax.scatter([], [], s=12, c="#126aab"))
        titles.append(ax.set_title(LABELS[result["method"]]))
        ax.set(xlim=(low[0] - margin, high[0] + margin),
               ylim=(low[1] - margin, high[1] + margin), aspect="equal", xlabel="x", ylabel="y")
    def update(frame):
        for scatter, title, result in zip(scatters, titles, results):
            scatter.set_offsets(result["positions"][frame])
            title.set_text(f"{LABELS[result['method']]}, t={result['times'][frame]:.3f}")
        return scatters + titles
    animation = FuncAnimation(fig, update, frames=len(results[0]["times"]), blit=False)
    fig.tight_layout()
    animation.save(out_dir / "animation.gif", writer=PillowWriter(fps=15))
    plt.close(fig)
