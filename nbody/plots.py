"""Сохранение расчёта, графиков и необязательной GIF-анимации."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from .rendering import _colors, render_animation, render_poster


LABELS = {"direct": "Прямой метод", "barnes-hut": "Барнс–Хат"}


def save_simulations(results, out_dir, *, animate=False, theme="cosmic", fps=24,
                     scenario="disk"):
    """Записать каждый расчёт в NPZ/CSV и совместно построить графики."""
    if theme not in {"cosmic", "paper"}:
        raise ValueError("theme должна быть cosmic или paper.")
    if not np.isfinite(fps) or not 1 <= fps <= 50:
        raise ValueError("Для GIF частота fps должна быть от 1 до 50.")
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
    _plot_trajectories(results, out_dir, theme=theme, scenario=scenario)
    _plot_invariants(results, out_dir, theme=theme)
    if theme == "cosmic":
        render_poster(results, out_dir / "poster.png", scenario=scenario)
    else:
        from PIL import Image
        with Image.open(out_dir / "trajectories.png") as poster:
            poster.save(out_dir / "poster.png")
    if animate:
        _animate(results, out_dir, theme=theme, fps=fps, scenario=scenario)
    return summary


def _plot_style(theme):
    dark = theme == "cosmic"
    return {
        "figure.facecolor": "#060a14" if dark else "white",
        "axes.facecolor": "#090f1d" if dark else "white",
        "savefig.facecolor": "#060a14" if dark else "white",
        "text.color": "#e6effc" if dark else "#172338",
        "axes.labelcolor": "#a3b6d1" if dark else "#34445b",
        "axes.edgecolor": "#293950" if dark else "#a4afbd",
        "xtick.color": "#91a7c7" if dark else "#526177",
        "ytick.color": "#91a7c7" if dark else "#526177",
        "grid.color": "#30405b" if dark else "#c5ccd5",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10,
        "legend.facecolor": "#101a2b" if dark else "white",
        "legend.edgecolor": "#293950" if dark else "#d9dfe8",
    }


def _plot_trajectories(results, out_dir, *, theme="cosmic", scenario="disk"):
    with plt.rc_context(_plot_style(theme)):
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5), squeeze=False)
        colors, _ = _colors(results[0]["positions"][0], scenario)
        for ax, result in zip(axes[0], results):
            x = result["positions"]
            chosen = np.linspace(0, x.shape[1] - 1, min(24, x.shape[1]), dtype=int)
            for i in chosen:
                ax.plot(x[:, i, 0], x[:, i, 1], linewidth=0.9, alpha=0.65,
                        color=colors[i] if theme == "cosmic" else None)
            ax.scatter(x[0, :, 0], x[0, :, 1], s=8, c="#74849a", alpha=0.45, label="Начало")
            ax.scatter(x[-1, :, 0], x[-1, :, 1], s=11,
                       c=colors if theme == "cosmic" else "#126aab", alpha=0.9, label="Конец")
            title = LABELS[result["method"]]
            if result["method"] == "barnes-hut":
                title += f", θ={result['theta']:g}"
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            ax.grid(alpha=0.28)
            ax.legend(framealpha=0.85)
        fig.suptitle(f"N={results[0]['positions'].shape[1]} · t={results[0]['times'][-1]:g} · траектории до 24 частиц")
        fig.tight_layout(pad=1.7)
        fig.savefig(out_dir / "trajectories.png", dpi=180)
        plt.close(fig)


def _plot_invariants(results, out_dir, *, theme="cosmic"):
    with plt.rc_context(_plot_style(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for result in results:
            label = LABELS[result["method"]]
            if theme == "cosmic":
                color = "#4aabff" if result["method"] == "direct" else "#ffb95f"
            else:
                color = "#176ca4" if result["method"] == "direct" else "#d9740d"
            axes[0].plot(result["times"], (result["energy"] - result["energy"][0]) / result["energy_scale"],
                         label=label, color=color, linewidth=1.7)
            axes[1].plot(result["times"], np.linalg.norm(result["momentum"] - result["momentum"][0], axis=1),
                         label=label, color=color, linewidth=1.7)
            axes[2].plot(result["times"], result["angular_momentum"] - result["angular_momentum"][0],
                         label=label, color=color, linewidth=1.7)
        for ax, title, ylabel in zip(axes, ["Энергия", "Импульс", "Момент импульса"],
                                      ["(E(t) − E(0)) / |E(0)|", "‖P(t) − P(0)‖", "Lz(t) − Lz(0)"]):
            ax.set(title=title, xlabel="t", ylabel=ylabel)
            ax.grid(alpha=0.28)
            ax.legend(framealpha=0.85)
        fig.tight_layout(pad=1.7)
        fig.savefig(out_dir / "conservation.png", dpi=180)
        plt.close(fig)


def _animate(results, out_dir, *, theme="cosmic", fps=24, scenario="disk"):
    if theme == "cosmic":
        return render_animation(results, out_dir / "animation.gif", scenario=scenario, fps=fps)
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
    animation.save(out_dir / "animation.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)
