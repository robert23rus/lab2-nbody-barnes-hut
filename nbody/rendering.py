"""Кинематографичная визуализация настоящих траекторий частиц.

Цвет закреплён за частицей при t=0. Свечение и короткие следы — только
способ отображения: координаты, силы и результаты моделирования не меняются.
Рендер не требует графического окна, FFmpeg, внешних шрифтов или видеокарты.
"""

from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


BLUE = np.array([0.12, 0.48, 1.0], dtype=np.float32)
GOLD = np.array([1.0, 0.58, 0.12], dtype=np.float32)
BACKGROUND = (3, 6, 13)
TEXT = (230, 239, 252)
MUTED = (128, 149, 177)
SCENARIOS = {
    "disk": "Вращающийся диск",
    "clusters": "Взаимодействие скоплений",
    "galaxies": "Сближение звёздных дисков",
    "binary": "Круговая орбита двух тел",
    "custom": "Пользовательские данные",
}
METHODS = {"direct": "ПРЯМОЙ МЕТОД", "barnes-hut": "БАРНС–ХАТ"}


@lru_cache(maxsize=16)
def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    filename = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    return ImageFont.truetype(str(filename), size=size)


def _validate(results, width, height):
    if not isinstance(width, int) or not isinstance(height, int) or width < 480 or height < 320:
        raise ValueError("Размер кадра должен быть не меньше 480×320 пикселей.")
    if not results or len(results) > 2:
        raise ValueError("Рендер принимает один или два расчёта.")
    shape = None
    for result in results:
        positions = np.asarray(result["positions"])
        times = np.asarray(result["times"])
        if (positions.ndim != 3 or positions.shape[2] != 2 or positions.shape[0] < 1
                or positions.shape[1] < 1 or times.shape != (len(positions),)
                or not np.isfinite(positions).all() or not np.isfinite(times).all()):
            raise ValueError("Нужны конечные координаты (K,N,2) и времена (K,).")
        if shape is not None and shape != positions.shape:
            raise ValueError("Сравниваемые расчёты должны иметь одинаковые размеры.")
        if shape is not None and not np.array_equal(times, results[0]["times"]):
            raise ValueError("Сравниваемые расчёты должны иметь одинаковые моменты времени.")
        shape = positions.shape


def _colors(initial, scenario):
    n = len(initial)
    if scenario in {"galaxies", "clusters", "binary"}:
        mix = np.where(np.arange(n) < n // 2, 0.0, 1.0)
    else:
        radius = np.linalg.norm(initial - np.median(initial, axis=0), axis=1)
        scale = max(float(np.quantile(radius, 0.95)), 1e-12)
        # Золотое ядро и холодная периферия; метка не меняется при движении.
        mix = (1 - np.clip(radius / scale, 0, 1)) ** 2
    colors = BLUE[None, :] * (1 - mix[:, None]) + GOLD[None, :] * mix[:, None]
    # Небольшой постоянный разброс яркости делает точки различимыми без мерцания.
    brightness = 0.75 + 1.4 * np.mod(np.arange(n) * 0.61803398875, 1.0) ** 3
    return colors.astype(np.float32), brightness.astype(np.float32)


def _camera(results, aspect, bounds):
    if bounds is None:
        all_positions = np.concatenate([np.asarray(r["positions"]).reshape(-1, 2) for r in results])
        if results[0]["positions"].shape[1] < 64:
            low, high = all_positions.min(axis=0), all_positions.max(axis=0)
        else:
            low, high = np.quantile(all_positions, [0.005, 0.995], axis=0)
        center = (low + high) / 2
        span = np.maximum(high - low, 0.3) * 1.16
    else:
        values = np.asarray(bounds, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("bounds задаётся как (xmin,xmax,ymin,ymax).")
        low, high = values[[0, 2]], values[[1, 3]]
        if np.any(high <= low):
            raise ValueError("Границы камеры должны задавать положительную область.")
        center, span = (low + high) / 2, high - low
    if span[0] / span[1] < aspect:
        span[0] = span[1] * aspect
    else:
        span[1] = span[0] / aspect
    return (float(center[0] - span[0] / 2), float(center[0] + span[0] / 2),
            float(center[1] - span[1] / 2), float(center[1] + span[1] / 2))


class _Scene:
    def __init__(self, results, scenario, width, height, bounds=None):
        _validate(results, width, height)
        self.results, self.scenario = results, scenario
        self.width, self.height = width, height
        self.count = len(results[0]["times"])
        self.n = results[0]["positions"].shape[1]
        self.margin, self.gap = 20, 16
        self.top, self.bottom = 96, height - 65
        self.panel_width = (width - 2 * self.margin - self.gap * (len(results) - 1)) // len(results)
        self.panel_height = self.bottom - self.top
        self.bounds = _camera(results, self.panel_width / self.panel_height, bounds)
        self.colors, self.brightness = _colors(results[0]["positions"][0], scenario)

    def _project(self, positions):
        xmin, xmax, ymin, ymax = self.bounds
        x = (positions[:, 0] - xmin) / (xmax - xmin) * (self.panel_width - 1)
        y = (ymax - positions[:, 1]) / (ymax - ymin) * (self.panel_height - 1)
        visible = (x >= 1) & (x < self.panel_width - 2) & (y >= 1) & (y < self.panel_height - 2)
        return x, y, visible

    def _particles(self, result, index):
        radiance = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.float32)
        history = np.asarray(result["positions"])
        # Истинные предыдущие положения, без нарисованных спиралей или орбит.
        first = max(0, index - 5)
        for sample in range(first, index + 1):
            age = index - sample
            strength = 0.32 ** age
            x, y, visible = self._project(history[sample])
            x, y = x[visible], y[visible]
            ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
            dx, dy = x - ix, y - iy
            color = self.colors[visible] * self.brightness[visible, None] * strength
            # Билинейное распределение даёт плавный субпиксельный ход точек.
            for ox, oy, weight in (
                (0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)),
                (0, 1, (1 - dx) * dy), (1, 1, dx * dy),
            ):
                np.add.at(radiance, (iy + oy, ix + ox), color * weight[:, None])
        luminous = (255 * (1 - np.exp(-2.8 * radiance)) ** 0.82).astype(np.uint8)
        stars = Image.fromarray(luminous)
        # Рассеяние света не меняет положения частиц: узкий core и два halo.
        core = stars.filter(ImageFilter.GaussianBlur(0.38))
        narrow = stars.filter(ImageFilter.GaussianBlur(1.8))
        wide = stars.filter(ImageFilter.GaussianBlur(6.0))
        halo = ImageChops.add(narrow, wide, scale=0.7)
        glow = ImageChops.add(core, halo, scale=0.88)
        background = Image.new("RGB", stars.size, BACKGROUND)
        glow = ImageChops.add(background, glow)
        _, _, visible = self._project(history[index])
        return glow, int(visible.sum())

    def render(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.count:
            raise ValueError("frame_index вне диапазона сохранённых снимков.")
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        title = SCENARIOS.get(self.scenario, "Гравитационная задача N тел")
        title_size = 25 if self.width >= 800 else 18
        draw.text((22, 14), "N–BODY  /  ГРАВИТАЦИОННАЯ ДИНАМИКА", font=_font(10, True), fill=(81, 149, 216))
        draw.text((20, 31), title, font=_font(title_size, True), fill=TEXT)
        current_time = float(self.results[0]["times"][index])
        time_text = f"t = {current_time:.3f}"
        time_font = _font(22 if self.width >= 800 else 16)
        time_width = draw.textlength(time_text, font=time_font)
        draw.text((self.width - time_width - 22, 34), time_text, font=time_font, fill=TEXT)
        first = self.results[0]
        if len(self.results) == 2 or first["method"] == "barnes-hut":
            parameters = f"N = {self.n:,}     θ = {float(first['theta']):g}     ε = {float(first['softening']):g}"
        else:
            parameters = f"N = {self.n:,}     ε = {float(first['softening']):g}     G = {float(first.get('gravitational_constant', 1)):g}"
        draw.text((22, 70), parameters.replace(",", " "), font=_font(12), fill=MUTED)
        snapshot = f"{index + 1} / {self.count}"
        draw.text((self.width - draw.textlength(snapshot, font=_font(11)) - 22, 71), snapshot,
                  font=_font(11), fill=MUTED)
        for panel, result in enumerate(self.results):
            left = self.margin + panel * (self.panel_width + self.gap)
            layer, visible = self._particles(result, index)
            image.paste(layer, (left, self.top))
            draw.rounded_rectangle((left, self.top, left + self.panel_width - 1, self.bottom),
                                   radius=10, outline=(25, 37, 57), width=1)
            method = METHODS.get(result["method"], result["method"])
            draw.text((left + 14, self.top + 12), method, font=_font(10, True), fill=(180, 200, 226))
            if visible < self.n:
                note = f"В кадре {visible} / {self.n}"
                draw.text((left + self.panel_width - draw.textlength(note, font=_font(9)) - 12,
                           self.top + 13), note, font=_font(9), fill=MUTED)
            self._scale_bar(draw, left)
        legend_y = self.height - 47
        if self.scenario in {"galaxies", "clusters", "binary"}:
            labels = ("Диск 1", "Диск 2") if self.scenario == "galaxies" else ("Группа 1", "Группа 2")
            pairs = ((BLUE, labels[0]), (GOLD, labels[1]))
        else:
            pairs = ((GOLD, "Ядро при t=0"), (BLUE, "Периферия при t=0"))
        cursor = 24
        for color, text in pairs:
            rgb = tuple((color * 255).astype(int))
            draw.ellipse((cursor, legend_y + 3, cursor + 7, legend_y + 10), fill=rgb)
            draw.text((cursor + 14, legend_y), text, font=_font(10), fill=MUTED)
            cursor += int(draw.textlength(text, font=_font(10))) + 40
        if self.width >= 800:
            note = "2D · цвет закреплён за частицей"
            draw.text((self.width - draw.textlength(note, font=_font(10)) - 23, legend_y), note,
                      font=_font(10), fill=(99, 119, 146))
        start, stop = float(first["times"][0]), float(first["times"][-1])
        progress = (current_time - start) / (stop - start) if stop > start else 1
        draw.line((24, self.height - 18, self.width - 24, self.height - 18), fill=(24, 35, 52), width=2)
        draw.line((24, self.height - 18, 24 + int((self.width - 48) * progress), self.height - 18),
                  fill=(56, 149, 240), width=2)
        return image

    def _scale_bar(self, draw, left):
        xmin, xmax, _, _ = self.bounds
        desired = (xmax - xmin) * 0.17
        exponent = 10 ** np.floor(np.log10(desired))
        length = max(value * exponent for value in (0.1, 0.2, 0.5, 1, 2, 5, 10) if value * exponent <= desired)
        pixels = int(length / (xmax - xmin) * self.panel_width)
        x, y = left + 16, self.bottom - 22
        draw.line((x, y, x + pixels, y), fill=(110, 131, 160), width=1)
        draw.line((x, y - 3, x, y + 3), fill=(110, 131, 160))
        draw.line((x + pixels, y - 3, x + pixels, y + 3), fill=(110, 131, 160))
        draw.text((x, y - 16), f"{length:g} ед.", font=_font(9), fill=MUTED)


def render_frame(results, frame_index, *, scenario="disk", width=1000, height=620, bounds=None):
    """RGB-кадр из сохранённого снимка; камера одинакова для всего расчёта.

    bounds: (xmin,xmax,ymin,ymax). Автовид берёт 0.5–99.5 процентили
    координат по всем кадрам: далёкие выбросы не уменьшают всё изображение.
    Фактическое число видимых частиц выводится, когда часть вышла из кадра.
    """
    return _Scene(results, scenario, width, height, bounds).render(frame_index)


def render_poster(results, out_path, *, scenario="disk", frame_index=None, width=1000, height=620):
    """Сохранить PNG выбранного снимка (по умолчанию — середины расчёта)."""
    scene = _Scene(results, scenario, width, height)
    index = scene.count // 2 if frame_index is None else frame_index
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render(index).save(out_path)
    return out_path


def render_animation(results, out_path, *, scenario="disk", fps=24, max_frames=180,
                     width=1000, height=620):
    """Сохранить зацикленную GIF с общей палитрой и без мерцания цветов.

    Не более max_frames фактических снимков, включая начальный и конечный.
    Интерполяция координат не используется. FPS задаёт скорость показа,
    а реальное модельное время постоянно показано в углу кадра.
    """
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 1 <= fps <= 50:
        raise ValueError("Для GIF частота fps должна быть от 1 до 50.")
    if not isinstance(max_frames, int) or max_frames < 2:
        raise ValueError("max_frames должно быть целым числом не меньше 2.")
    scene = _Scene(results, scenario, width, height)
    selected = np.unique(np.linspace(0, scene.count - 1, min(scene.count, max_frames)).round().astype(int))
    sample_indices = sorted({int(selected[0]), int(selected[len(selected) // 2]), int(selected[-1])})
    sample_image = Image.new("RGB", (width, height * len(sample_indices)))
    samples = {}
    for row, index in enumerate(sample_indices):
        samples[index] = scene.render(index)
        sample_image.paste(samples[index], (0, row * height))
    palette = sample_image.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                    dither=Image.Dither.NONE)
    frames = []
    for index in selected:
        rgb = samples[int(index)] if int(index) in samples else scene.render(int(index))
        frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
    # GIF хранит сотые доли секунды; чередование 40/50мс точно передаёт 24fps.
    boundaries = np.rint(np.arange(len(frames) + 1) * 100 / fps).astype(int) * 10
    durations = np.diff(boundaries).tolist()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, disposal=2, optimize=True,
                   palette=palette.getpalette())
    for frame in frames:
        frame.close()
    return out_path
