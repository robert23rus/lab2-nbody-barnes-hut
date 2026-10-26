"""Прямое суммирование и двумерный алгоритм Барнса–Хата.

Координаты лежат в плоскости, но закон тяготения остаётся ньютоновским:
a_i = G sum_j m_j (r_j-r_i) / (|r_j-r_i|² + eps²)^(3/2).
Numba, если установлен, компилирует оба метода с одинаковыми настройками.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

try:
    from numba import njit

    BACKEND = "numba"
except ImportError:
    BACKEND = "python"

    def njit(*args, **kwargs):
        """Оставить те же алгоритмы доступными без необязательного Numba."""
        def decorate(function):
            return function
        return decorate


def _validate(positions, masses, gravitational_constant, softening):
    """Проверить данные до входа в вычислительные ядра."""
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions должен быть непустым массивом формы (N, 2).")
    if masses.ndim != 1 or len(masses) != len(positions):
        raise ValueError("masses должен иметь форму (N,).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Координаты должны быть конечными числами.")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("Массы должны быть положительными конечными числами.")
    gravitational_constant = float(gravitational_constant)
    softening = float(softening)
    if not math.isfinite(gravitational_constant) or gravitational_constant < 0:
        raise ValueError("Гравитационная постоянная должна быть конечной и >= 0.")
    if not math.isfinite(softening) or softening < 0:
        raise ValueError("Параметр смягчения должен быть конечным и >= 0.")
    if softening == 0 and len(np.unique(positions, axis=0)) < len(positions):
        raise ValueError("При softening=0 совпавшие частицы дают сингулярную силу.")
    if not math.isfinite(float(np.sum(masses))):
        raise ValueError("Суммарная масса выходит за диапазон float64.")
    return (np.ascontiguousarray(positions), np.ascontiguousarray(masses),
            gravitational_constant, softening)


@njit(cache=True)
def _direct_kernel(positions, masses, gravitational_constant, softening):
    """Каждая неупорядоченная пара учитывается ровно один раз."""
    n = len(masses)
    result = np.zeros((n, 2), dtype=np.float64)
    eps2 = softening * softening
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            distance2 = dx * dx + dy * dy + eps2
            factor = gravitational_constant / (distance2 * math.sqrt(distance2))
            result[i, 0] += masses[j] * factor * dx
            result[i, 1] += masses[j] * factor * dy
            result[j, 0] -= masses[i] * factor * dx
            result[j, 1] -= masses[i] * factor * dy
    return result


@njit(cache=True)
def _build_tree(positions, masses, leaf_capacity, max_depth):
    """Построить дерево в плоских массивах без Python-объектов.

    Ветви с единственным непустым потомком сжимаются: уменьшается ячейка,
    но новый узел не создаётся. Поэтому узлов не больше 2N-1. Каждый узел
    хранит непрерывный диапазон перестановки order. Это позволяет точно
    определить, содержит ли узел целевую частицу, без геометрических допусков.
    """
    n = len(masses)
    capacity = 2 * n - 1
    order = np.arange(n)
    scratch = np.empty(n, dtype=np.int64)
    starts = np.zeros(capacity, dtype=np.int64)
    stops = np.zeros(capacity, dtype=np.int64)
    center_x = np.zeros(capacity, dtype=np.float64)
    center_y = np.zeros(capacity, dtype=np.float64)
    half_size = np.zeros(capacity, dtype=np.float64)
    depth = np.zeros(capacity, dtype=np.int64)
    total_mass = np.zeros(capacity, dtype=np.float64)
    com_x = np.zeros(capacity, dtype=np.float64)
    com_y = np.zeros(capacity, dtype=np.float64)
    children = np.full((capacity, 4), -1, dtype=np.int64)

    xmin = positions[0, 0]
    xmax = xmin
    ymin = positions[0, 1]
    ymax = ymin
    for i in range(1, n):
        xmin = min(xmin, positions[i, 0])
        xmax = max(xmax, positions[i, 0])
        ymin = min(ymin, positions[i, 1])
        ymax = max(ymax, positions[i, 1])
    center_x[0] = xmin + (xmax - xmin) * 0.5
    center_y[0] = ymin + (ymax - ymin) * 0.5
    half_size[0] = max(xmax - xmin, ymax - ymin) * 0.5 * (1.0 + 1e-12)
    stops[0] = n

    node_count = 1
    node = 0
    counts = np.empty(4, dtype=np.int64)
    offsets = np.empty(4, dtype=np.int64)
    while node < node_count:
        begin = starts[node]
        end = stops[node]
        first = order[begin]
        same_position = True
        for k in range(begin, end):
            particle = order[k]
            total_mass[node] += masses[particle]
            com_x[node] += masses[particle] * positions[particle, 0]
            com_y[node] += masses[particle] * positions[particle, 1]
            if (positions[particle, 0] != positions[first, 0]
                    or positions[particle, 1] != positions[first, 1]):
                same_position = False
        com_x[node] /= total_mass[node]
        com_y[node] /= total_mass[node]

        # Совпавшие координаты образуют один лист при любом числе частиц.
        while (end - begin > leaf_capacity and depth[node] < max_depth
               and half_size[node] > 0 and not same_position):
            counts[:] = 0
            for k in range(begin, end):
                particle = order[k]
                quadrant = 0
                if positions[particle, 0] >= center_x[node]:
                    quadrant += 1
                if positions[particle, 1] >= center_y[node]:
                    quadrant += 2
                counts[quadrant] += 1
            nonempty = 0
            only_quadrant = 0
            for quadrant in range(4):
                if counts[quadrant] > 0:
                    nonempty += 1
                    only_quadrant = quadrant

            child_half = half_size[node] * 0.5
            if nonempty == 1:
                center_x[node] += child_half * (1 if only_quadrant & 1 else -1)
                center_y[node] += child_half * (1 if only_quadrant & 2 else -1)
                half_size[node] = child_half
                depth[node] += 1
                continue

            # Упорядочить частицы по четырём дочерним ячейкам.
            offsets[0] = begin
            for quadrant in range(1, 4):
                offsets[quadrant] = offsets[quadrant - 1] + counts[quadrant - 1]
            for k in range(begin, end):
                particle = order[k]
                quadrant = 0
                if positions[particle, 0] >= center_x[node]:
                    quadrant += 1
                if positions[particle, 1] >= center_y[node]:
                    quadrant += 2
                scratch[offsets[quadrant]] = particle
                offsets[quadrant] += 1
            for k in range(begin, end):
                order[k] = scratch[k]

            next_begin = begin
            for quadrant in range(4):
                if counts[quadrant] > 0:
                    child = node_count
                    node_count += 1
                    children[node, quadrant] = child
                    starts[child] = next_begin
                    stops[child] = next_begin + counts[quadrant]
                    center_x[child] = center_x[node] + child_half * (1 if quadrant & 1 else -1)
                    center_y[child] = center_y[node] + child_half * (1 if quadrant & 2 else -1)
                    half_size[child] = child_half
                    depth[child] = depth[node] + 1
                    next_begin += counts[quadrant]
            break
        node += 1

    rank = np.empty(n, dtype=np.int64)
    for k in range(n):
        rank[order[k]] = k
    return (order, rank, starts, stops, half_size, total_mass,
            com_x, com_y, children, node_count)


@njit(cache=True)
def _barnes_hut_kernel(positions, masses, theta, gravitational_constant,
                       softening, leaf_capacity, max_depth):
    (order, rank, starts, stops, half_size, total_mass,
     com_x, com_y, children, node_count) = _build_tree(
        positions, masses, leaf_capacity, max_depth)
    n = len(masses)
    result = np.zeros((n, 2), dtype=np.float64)
    stack = np.empty(node_count, dtype=np.int64)
    eps2 = softening * softening
    theta2 = theta * theta
    for i in range(n):
        stack[0] = 0
        stack_size = 1
        while stack_size:
            stack_size -= 1
            node = stack[stack_size]
            is_leaf = True
            for quadrant in range(4):
                if children[node, quadrant] >= 0:
                    is_leaf = False
                    break
            if is_leaf:
                for k in range(starts[node], stops[node]):
                    j = order[k]
                    if j == i:
                        continue
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    distance2 = dx * dx + dy * dy + eps2
                    factor = (gravitational_constant * masses[j]
                              / (distance2 * math.sqrt(distance2)))
                    result[i, 0] += factor * dx
                    result[i, 1] += factor * dy
                continue

            dx = com_x[node] - positions[i, 0]
            dy = com_y[node] - positions[i, 1]
            geometric_distance2 = dx * dx + dy * dy
            contains_target = starts[node] <= rank[i] < stops[node]
            side = 2.0 * half_size[node]
            # d в критерии открытия не смягчается. Узел с i всегда раскрывается.
            if (not contains_target and geometric_distance2 > 0
                    and side * side < theta2 * geometric_distance2):
                distance2 = geometric_distance2 + eps2
                factor = (gravitational_constant * total_mass[node]
                          / (distance2 * math.sqrt(distance2)))
                result[i, 0] += factor * dx
                result[i, 1] += factor * dy
            else:
                for quadrant in range(4):
                    child = children[node, quadrant]
                    if child >= 0:
                        stack[stack_size] = child
                        stack_size += 1
    return result


def direct_accelerations(positions, masses, *, gravitational_constant=1.0,
                         softening=0.03):
    """Вычислить ускорения прямым методом O(N²), память O(N)."""
    positions, masses, gravitational_constant, softening = _validate(
        positions, masses, gravitational_constant, softening)
    return _direct_kernel(positions, masses, gravitational_constant, softening)


def barnes_hut_accelerations(positions, masses, *, theta=0.5,
                             gravitational_constant=1.0, softening=0.03,
                             leaf_capacity=1, max_depth=48):
    """Построить дерево и вычислить ускорения методом Барнса–Хата.

    Типичная сложность O(N log N), память O(N). theta=0 раскрывает все
    узлы и даёт прямую сумму с другим порядком сложения. Достижение
    max_depth оставляет несколько частиц в листе: они считаются напрямую.
    """
    positions, masses, gravitational_constant, softening = _validate(
        positions, masses, gravitational_constant, softening)
    theta = float(theta)
    if not math.isfinite(theta) or theta < 0:
        raise ValueError("theta должен быть конечным и >= 0.")
    if (not isinstance(leaf_capacity, Integral) or isinstance(leaf_capacity, bool)
            or leaf_capacity < 1):
        raise ValueError("leaf_capacity должен быть целым числом >= 1.")
    if (not isinstance(max_depth, Integral) or isinstance(max_depth, bool)
            or max_depth < 1):
        raise ValueError("max_depth должен быть целым числом >= 1.")
    return _barnes_hut_kernel(positions, masses, theta, gravitational_constant,
                              softening, int(leaf_capacity), int(max_depth))


def warm_up():
    """Скомпилировать оба ядра до измерения времени (при наличии Numba)."""
    positions = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    masses = np.ones(3)
    direct_accelerations(positions, masses)
    barnes_hut_accelerations(positions, masses)
