"""Собрать GIF README из воспроизводимых расчётов, без сетевых ресурсов.

    python scripts/build_readme_media.py
    python scripts/build_readme_media.py --simulate-only
    python scripts/build_readme_media.py --render-only --only galaxies

Снимки хранятся в игнорируемом results/readme_media, готовые медиа и их
параметры — в docs/media. Изменение рендера не требует пересчёта динамики.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbody.forces import BACKEND, warm_up
from nbody.simulation import initial_conditions, simulate


CACHE = ROOT / "results" / "readme_media"
MEDIA = ROOT / "docs" / "media"
CASES = {
    "galaxies": {
        "scenario": "galaxies", "n": 4096, "seed": 42,
        "methods": ["barnes-hut"], "steps": 800, "dt": 0.0075,
        "sample_every": 5, "theta": 0.6, "softening": 0.03,
        "gravitational_constant": 1.0, "width": 1000, "height": 620,
        "fps": 24, "max_frames": 180,
    },
    "comparison": {
        "scenario": "clusters", "n": 512, "seed": 42,
        "methods": ["direct", "barnes-hut"], "steps": 400, "dt": 0.005,
        "sample_every": 4, "theta": 0.5, "softening": 0.03,
        "gravitational_constant": 1.0, "width": 1000, "height": 620,
        "fps": 24, "max_frames": 140,
    },
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_sources():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("nbody/forces.py", "nbody/simulation.py")
    }


def simulation_config(case):
    visual = {"width", "height", "fps", "max_frames"}
    return {key: value for key, value in case.items() if key not in visual}


def load_results(name, case, *, render_only=False):
    signature = {"parameters": simulation_config(case), "sources_sha256": model_sources()}
    config_path = CACHE / f"{name}_config.json"
    cache_valid = config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) == signature
    paths = [CACHE / f"{name}_{method}.npz" for method in case["methods"]]
    if cache_valid and all(path.exists() for path in paths):
        loaded = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                loaded.append({key: (value.item() if value.ndim == 0 else value.copy())
                               for key in data.files for value in [data[key]]})
        print(f"{name}: использованы сохранённые снимки.", flush=True)
        return loaded
    if render_only:
        raise ValueError(f"Нет актуального кэша {name}; запустите без --render-only.")
    if BACKEND != "numba":
        raise ValueError("Большие демонстрации требуют Numba. Установите requirements.txt в .venv.")
    CACHE.mkdir(parents=True, exist_ok=True)
    physical = {key: case[key] for key in ("softening", "gravitational_constant")}
    x, masses, velocities = initial_conditions(
        case["n"], seed=case["seed"], scenario=case["scenario"], **physical
    )
    output = []
    for method, path in zip(case["methods"], paths):
        print(f"{name}: {method}, N={case['n']}, {case['steps']} шагов...", flush=True)
        start = perf_counter()
        result = simulate(
            x, masses, velocities, method=method, steps=case["steps"], dt=case["dt"],
            sample_every=case["sample_every"], theta=case["theta"], **physical,
        )
        if not np.isfinite(result["positions"]).all():
            raise ValueError("Расчёт содержит нечисловые координаты.")
        np.savez_compressed(path, **result)
        output.append(result)
        print(f"  {len(result['times'])} снимков; {perf_counter() - start:.2f} с.", flush=True)
    write_json(config_path, signature)
    return output


def gif_info(path):
    with Image.open(path) as image:
        count = image.n_frames
        duration = 0
        for frame in range(count):
            image.seek(frame)
            duration += image.info.get("duration", 0)
        if count < 2:
            raise ValueError(f"{path.name}: получен неподвижный GIF.")
        return {"frames": count, "size": list(image.size),
                "duration_seconds": duration / 1000, "bytes": path.stat().st_size,
                "loop": image.info.get("loop")}


def render_case(name, case, results):
    from nbody.rendering import render_animation, render_poster

    MEDIA.mkdir(parents=True, exist_ok=True)
    gif_path = MEDIA / f"{name}.gif"
    poster_path = MEDIA / f"{name}-poster.png"
    width, height, frames = case["width"], case["height"], case["max_frames"]
    info = None
    # Ограничиваем размер загрузки README; все варианты включают начало и конец.
    for factor in (1.0, 0.88, 0.8):
        width = round(case["width"] * factor)
        height = round(case["height"] * factor)
        frames = max(96, round(case["max_frames"] * factor))
        print(f"{name}: GIF {width}×{height}, до {frames} кадров...", flush=True)
        render_animation(results, gif_path, scenario=case["scenario"], fps=case["fps"],
                         max_frames=frames, width=width, height=height)
        info = gif_info(gif_path)
        if info["bytes"] <= 8 * 1024**2:
            break
    if info["bytes"] > 8 * 1024**2:
        raise ValueError(f"{gif_path.name}: GIF превышает лимит 8 МиБ.")
    frame_index = round((len(results[0]["times"]) - 1) * 0.52)
    render_poster(results, poster_path, scenario=case["scenario"], frame_index=frame_index,
                  width=width, height=height)
    print(f"  {info['frames']} кадров; {info['bytes'] / 1024**2:.2f} МиБ.", flush=True)
    return {
        "file": f"{name}.gif", "poster": f"{name}-poster.png", **info,
        "sha256": hashlib.sha256(gif_path.read_bytes()).hexdigest(),
        "parameters": simulation_config(case),
        "model_time_final": float(results[0]["times"][-1]),
        "saved_simulation_frames": len(results[0]["times"]),
        "requested_fps": case["fps"], "theme": "cosmic",
        "sources_sha256": model_sources(),
        "rendering_source_sha256": hashlib.sha256((ROOT / "nbody/rendering.py").read_bytes()).hexdigest(),
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Воспроизвести анимации README из реальной модели N тел.")
    parser.add_argument("--only", choices=list(CASES), help="Собрать один из двух примеров")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate-only", action="store_true", help="Только подготовить NPZ-снимки")
    mode.add_argument("--render-only", action="store_true", help="Перерисовать актуальные NPZ без интегрирования")
    args = parser.parse_args()
    names = [args.only] if args.only else list(CASES)
    if not args.render_only:
        warm_up()
    entries = {}
    manifest_path = MEDIA / "manifest.json"
    if manifest_path.exists():
        entries = {entry["file"]: entry for entry in json.loads(manifest_path.read_text(encoding="utf-8"))["media"]}
    for name in names:
        case = CASES[name]
        results = load_results(name, case, render_only=args.render_only)
        if not args.simulate_only:
            entry = render_case(name, case, results)
            entries[entry["file"]] = entry
    if not args.simulate_only:
        write_json(manifest_path, {
            "description": "Реальные координаты N-body-модели; свечение и следы добавлены при отрисовке.",
            "model": "Ньютоновская гравитация в плоскости со смягчением Пламмера.",
            "colours": "Для galaxies и clusters цвет сохраняет принадлежность к исходной группе.",
            "software": {name: version(name) for name in ("numpy", "numba", "Pillow")},
            "media": [entries[key] for key in sorted(entries)],
        })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(2)
