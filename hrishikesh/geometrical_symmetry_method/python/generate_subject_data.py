"""Generate deterministic synthetic incomplete point-cloud test cases.

This generator creates:
  1. an incomplete input scan with optional rubble/noise labels,
  2. an exact missing structural surface for the selected test case,
  3. a complete clean reference surface for audit/debug.

The core completion pipeline remains in python/completion_pipeline.py and is not
modified.  The exact missing surface is used only by the subject adapter to make
viewer results stable for synthetic test cases instead of relying on screen- or
case-specific patch deletion.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROMAN_CASES = {
    "baseline_demolition",
    "north_rim_blast",
    "diagonal_collapse",
    "absurd_half_chop",
}

PALMYRA_CASES = {
    "left_upper_breach",
    "right_pier_shear",
    "crown_column_blast",
    "absurd_half_chop",
}

LEANING_TOWER_CASES = {
    "upper_bell_collapse",
    "lower_arcade_breach",
    "diagonal_tower_crack",
    "absurd_half_chop",
}

STRUCTURAL_ROMAN = {"facade", "arch_edge", "seating", "floor"}
STRUCTURAL_PALMYRA = {"facade", "arch_edge", "column", "entablature", "base"}
STRUCTURAL_LEANING_TOWER = {"tower_wall", "arcade", "ring", "column", "base", "bell_chamber", "stair"}

PointRow = tuple[float, float, float, str]


def write_csv(points: list[PointRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "label"])
        for x, y, z, label in points:
            writer.writerow([round(float(x), 6), round(float(y), 6), round(float(z), 6), str(label)])


def in_arc(theta: float, start: float, end: float) -> bool:
    start %= 2.0 * math.pi
    end %= 2.0 * math.pi
    if start <= end:
        return start <= theta <= end
    return theta >= start or theta <= end


def theta_ellipse(x: float, y: float, a: float, b: float) -> float:
    return math.atan2(y / b, x / a) % (2.0 * math.pi)


def add_rubble(points: list[PointRow], rng: np.random.Generator,
               count: int, x_range: tuple[float, float], y_range: tuple[float, float],
               z_max: float) -> None:
    """Add labeled rubble.  The viewer excludes this from clean reconstruction."""
    for _ in range(count):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        z = min(float(rng.gamma(shape=1.35, scale=max(0.04, z_max / 5.0))), z_max)
        points.append((x, y, z, "rubble"))


def roman_arena_complete(seed: int = 11) -> list[PointRow]:
    """Clean complete Roman amphitheater-style reference surface."""
    points: list[PointRow] = []

    outer_a, outer_b = 6.2, 4.05
    seat_outer_a, seat_outer_b = 5.0, 3.22
    inner_a, inner_b = 2.30, 1.48
    max_height = 3.75
    angle_steps = 260
    number_of_arches = 30

    def ellipse(a: float, b: float, theta: float, z: float) -> tuple[float, float, float]:
        return (a * math.cos(theta), b * math.sin(theta), z)

    # One outer facade only.  Repeated arch openings are carved into the facade.
    for i in range(angle_steps):
        theta = (i / angle_steps) * 2.0 * math.pi
        arch_index = round(theta / (2.0 * math.pi) * number_of_arches)
        arch_center = (arch_index / number_of_arches) * 2.0 * math.pi
        angle_dist = abs(math.atan2(math.sin(theta - arch_center), math.cos(theta - arch_center)))
        for z in np.arange(0.0, max_height + 1e-9, 0.090):
            arch_width = 0.036
            arch_base = 0.36
            arch_top = 1.58
            rounded_top = arch_top + 0.46 * math.sqrt(max(0.0, 1.0 - angle_dist / arch_width))
            in_opening = angle_dist < arch_width and arch_base < z < rounded_top
            if in_opening:
                if abs(angle_dist - arch_width) < 0.004 or abs(z - rounded_top) < 0.055:
                    points.append((*ellipse(outer_a, outer_b, theta, z), "arch_edge"))
                continue
            points.append((*ellipse(outer_a, outer_b, theta, z), "facade"))

    # Base and roof rings to give a readable boundary without creating a second wall.
    for z in [0.0, 0.22, max_height]:
        for i in range(angle_steps):
            theta = (i / angle_steps) * 2.0 * math.pi
            points.append((*ellipse(outer_a, outer_b, theta, z), "facade"))

    # Horizontal stepped seating bowl.
    tiers = 15
    for t in range(tiers):
        r = t / (tiers - 1)
        a = inner_a + r * (seat_outer_a - inner_a)
        b = inner_b + r * (seat_outer_b - inner_b)
        z = 0.34 + r * 2.35
        for i in range(0, angle_steps, 2):
            theta = (i / angle_steps) * 2.0 * math.pi
            points.append((*ellipse(a, b, theta, z), "seating"))
            if t > 0 and i % 6 == 0:
                points.append((*ellipse(a, b, theta, z - 0.16), "seating"))

    # Flat arena floor.
    for x in np.arange(-inner_a, inner_a + 1e-9, 0.18):
        for y in np.arange(-inner_b, inner_b + 1e-9, 0.18):
            if (x * x) / (inner_a * inner_a) + (y * y) / (inner_b * inner_b) <= 1.0:
                points.append((float(x), float(y), 0.02, "floor"))

    return points


def roman_damage_mask(case: str, x: float, y: float, z: float, label: str) -> bool:
    """True means the structural point is removed from the incomplete input."""
    a, b = 6.2, 4.05
    theta = theta_ellipse(x, y, a, b)
    rho = math.sqrt((x / a) ** 2 + (y / b) ** 2)

    if label not in {"facade", "arch_edge", "seating"}:
        return False

    if case == "baseline_demolition":
        # One clear demolished sector, plus seating continuity.  No separate
        # random craters, so the red reconstruction is easy to evaluate.
        if label in {"facade", "arch_edge"} and in_arc(theta, 1.03 * math.pi, 1.32 * math.pi) and 0.25 < z < 3.55:
            return True
        if label == "seating" and in_arc(theta, 1.02 * math.pi, 1.31 * math.pi) and rho > 0.57 and z > 0.55:
            return True

    elif case == "north_rim_blast":
        # Large but coherent upper-rim loss.  Lower structure remains so the
        # red patch is a clean upper continuation rather than an entire cylinder.
        if label in {"facade", "arch_edge"} and in_arc(theta, 0.29 * math.pi, 0.65 * math.pi) and z > 1.10:
            return True
        if label == "seating" and in_arc(theta, 0.34 * math.pi, 0.60 * math.pi) and rho > 0.58 and z > 0.85:
            return True

    elif case == "diagonal_collapse":
        # Diagonal scar on the arena surface.  The mask is wider than before so
        # reconstruction fills a visible structural slice, not sparse speckles.
        diagonal = (theta - 1.30 * math.pi) + 0.20 * (z - 1.65)
        if abs(diagonal) < 0.27 and 0.35 < z < 3.45:
            return True
        if label == "seating" and abs(diagonal) < 0.24 and rho > 0.55 and z > 0.60:
            return True

    elif case == "absurd_half_chop":
        # Non-symmetric half-demolition.  It removes roughly one side of the
        # amphitheater, but the boundary changes with height and radius so the
        # cut is intentionally absurd instead of a clean mirror split.
        boundary_shift = 0.13 * math.sin(2.2 * z) + 0.08 * math.sin(6.0 * rho)
        start = (1.55 + boundary_shift) * math.pi
        end = (0.35 + 0.07 * math.sin(1.7 * z)) * math.pi
        jagged = 0.10 * math.sin(4.5 * theta + 1.2 * z)
        if in_arc(theta, start, end) and z > 0.18 and rho > 0.48 + jagged:
            return True
        if label == "seating" and in_arc(theta, start - 0.08, end + 0.06) and z > 0.55 and rho > 0.50:
            return True

    return False


def generate_roman_arena(case: str, seed: int = 11) -> tuple[list[PointRow], list[PointRow], list[PointRow]]:
    if case not in ROMAN_CASES:
        raise ValueError(f"Unknown Roman Arena case: {case}")

    rng = np.random.default_rng(seed)
    reference = roman_arena_complete(seed=seed)
    observed: list[PointRow] = []
    missing: list[PointRow] = []

    for row in reference:
        x, y, z, label = row
        if roman_damage_mask(case, x, y, z, label):
            missing.append(row)
        else:
            observed.append(row)

    if case == "baseline_demolition":
        add_rubble(observed, rng, 420, (-5.8, -2.1), (-3.9, -0.5), 0.75)
    elif case == "north_rim_blast":
        add_rubble(observed, rng, 520, (-0.6, 3.7), (1.0, 4.2), 0.95)
    elif case == "diagonal_collapse":
        add_rubble(observed, rng, 500, (-5.6, -1.2), (-4.1, -0.5), 0.75)
    elif case == "absurd_half_chop":
        add_rubble(observed, rng, 760, (-6.0, 1.3), (-4.2, 4.2), 1.05)

    return observed, missing, reference


def palmyra_arch_complete(seed: int = 21) -> list[PointRow]:
    """Complete Palmyra-style arch reference surface."""
    points: list[PointRow] = []

    width = 6.4
    half_w = width / 2.0
    depth_faces = [-0.22, 0.22]
    height = 5.2
    arch_center_z = 1.55
    arch_radius = 1.55
    arch_clear_half = 1.05

    def opening(x: float, z: float) -> bool:
        if abs(x) > arch_clear_half:
            return False
        cap = arch_center_z + math.sqrt(max(0.0, arch_radius * arch_radius - x * x))
        return 0.20 < z < cap

    # Front/rear facade sheets.
    for y in depth_faces:
        for x in np.arange(-half_w, half_w + 1e-9, 0.070):
            for z in np.arange(0.0, height + 1e-9, 0.070):
                xf = float(x)
                zf = float(z)
                if opening(xf, zf):
                    continue
                if zf < 0.42:
                    label = "base"
                elif zf > 3.72:
                    label = "entablature"
                elif abs(xf) > 1.08 and zf < 3.70:
                    label = "column"
                else:
                    label = "facade"
                points.append((xf, y, zf, label))

    # Side thickness surfaces.
    for x in [-half_w, half_w]:
        for y in np.arange(-0.22, 0.22 + 1e-9, 0.070):
            for z in np.arange(0.0, height + 1e-9, 0.090):
                points.append((float(x), float(y), float(z), "facade"))

    # Base and entablature shelf surfaces.
    for z, label in [(0.0, "base"), (0.40, "base"), (3.72, "entablature"), (5.20, "entablature")]:
        for x in np.arange(-half_w, half_w + 1e-9, 0.070):
            for y in np.arange(-0.22, 0.22 + 1e-9, 0.070):
                xf = float(x)
                if opening(xf, float(z)):
                    continue
                points.append((xf, float(y), z, label))

    # Arch rings.
    for radius in [1.18, 1.48]:
        for theta in np.linspace(0.0, math.pi, 120):
            x = radius * math.cos(float(theta))
            z = arch_center_z + radius * math.sin(float(theta))
            for y in np.linspace(-0.24, 0.24, 6):
                points.append((float(x), float(y), float(z), "arch_edge"))

    # Vertical jamb edges.
    for x in [-1.18, -1.48, 1.18, 1.48]:
        for z in np.arange(0.28, arch_center_z + 0.04, 0.060):
            for y in np.linspace(-0.24, 0.24, 5):
                points.append((float(x), float(y), float(z), "arch_edge"))

    return points


def palmyra_damage_mask(case: str, x: float, y: float, z: float, label: str) -> bool:
    if label not in STRUCTURAL_PALMYRA:
        return False

    if case == "left_upper_breach":
        if x < -1.15 and z > 3.18:
            return True
        if -1.78 < x < -0.66 and 2.25 < z < 3.55:
            return True
        if label == "arch_edge" and x < -0.20 and z > 2.55:
            return True

    elif case == "right_pier_shear":
        if 1.55 < x < 2.52 and 0.58 < z < 3.52:
            return True
        if 2.38 < x < 3.18 and z < 0.92:
            return True
        if label == "arch_edge" and x > 0.72 and 1.62 < z < 2.75:
            return True

    elif case == "crown_column_blast":
        if -0.95 < x < 0.05 and z > 3.66:
            return True
        if -3.18 < x < -1.85 and 1.06 < z < 2.90:
            return True
        if label == "arch_edge" and x < -0.50 and 1.70 < z < 2.82:
            return True

    elif case == "absurd_half_chop":
        # Absurd non-symmetric half chop of the arch facade.  The cut is not a
        # vertical x=0 split; it leans with height and leaves/erases different
        # architectural bands so reconstruction is more challenging.
        cut = x + 0.20 * (z - 2.65) - 0.10 * math.sin(3.0 * z) + 0.22 * y
        if cut < -0.12 and 0.18 < z < 5.08:
            return True
        if label == "arch_edge" and x < 0.15 and z > 1.45 and (x + 0.16 * z) < 0.55:
            return True
        if label in {"base", "column"} and x < -0.55 and z < 1.05:
            return True

    return False


def generate_palmyra_arch(case: str, seed: int = 21) -> tuple[list[PointRow], list[PointRow], list[PointRow]]:
    if case not in PALMYRA_CASES:
        raise ValueError(f"Unknown Palmyra Arch case: {case}")

    rng = np.random.default_rng(seed)
    reference = palmyra_arch_complete(seed=seed)
    observed: list[PointRow] = []
    missing: list[PointRow] = []

    for row in reference:
        x, y, z, label = row
        if palmyra_damage_mask(case, x, y, z, label):
            missing.append(row)
        else:
            observed.append(row)

    if case == "left_upper_breach":
        add_rubble(observed, rng, 430, (-3.20, -0.65), (-0.62, 0.55), 0.85)
    elif case == "right_pier_shear":
        add_rubble(observed, rng, 440, (1.15, 3.25), (-0.62, 0.62), 0.85)
    elif case == "crown_column_blast":
        add_rubble(observed, rng, 460, (-3.25, -1.30), (-0.70, 0.60), 0.90)
    elif case == "absurd_half_chop":
        add_rubble(observed, rng, 650, (-3.25, 0.15), (-0.70, 0.70), 1.05)

    return observed, missing, reference




def petra_treasury_complete(seed: int = 41) -> list[PointRow]:
    """Complete 3D Petra Treasury-style monument reference surface.

    This version is intentionally more 3D than the previous flat facade.  It
    keeps the project deterministic, but models the monument as a carved cliff
    volume with depth: a rough sandstone rock mass, side canyon returns, a
    recessed central doorway, projecting columns, thick cornices, steps,
    pediment surfaces, an upper tholos, and side relief/niches.  The synthetic
    missing files are still exact structural masks, so the viewer remains stable
    without changing the core completion pipeline.
    """
    points: list[PointRow] = []

    rng = np.random.default_rng(seed)
    cliff_w = 6.2
    cliff_h = 6.75
    cliff_back_y = 1.45
    rock_front_y = 0.08
    facade_w = 4.15
    facade_h = 5.78
    facade_y = -0.24

    def rough(x: float, z: float) -> float:
        return 0.040 * math.sin(1.15 * x + 0.48 * z) + 0.025 * math.sin(2.85 * x - 1.20 * z)

    def doorway_opening(x: float, z: float) -> bool:
        # Large central portal.  The actual recess surfaces are added later, so
        # the front sheet is skipped here.
        return abs(x) < 0.72 and 0.22 < z < 2.18

    def side_niche_opening(x: float, z: float) -> bool:
        specs = [(-2.45, 2.04, 0.33, 0.44), (2.45, 2.04, 0.33, 0.44),
                 (-2.35, 4.18, 0.28, 0.36), (2.35, 4.18, 0.28, 0.36)]
        for cx, cz, rx, rz in specs:
            if ((x - cx) / rx) ** 2 + ((z - cz) / rz) ** 2 <= 1.0:
                return True
        return False

    # ------------------------------------------------------------
    # 1. Rough sandstone cliff mass with real depth.
    # ------------------------------------------------------------
    # Front cliff surface around the carved facade.  The carved facade replaces
    # the middle region; margins remain rough rock.
    for x in np.arange(-cliff_w, cliff_w + 1e-9, 0.090):
        for z in np.arange(0.0, cliff_h + 1e-9, 0.090):
            xf = float(x)
            zf = float(z)
            in_carved_zone = abs(xf) < facade_w and 0.05 < zf < facade_h
            if in_carved_zone and abs(xf) < facade_w - 0.25 and zf < facade_h - 0.18:
                continue
            y = rock_front_y + rough(xf, zf)
            points.append((xf, y, zf, "rock"))

    # Side canyon returns make the monument read as a 3D rock block instead of
    # a poster-like vertical wall.
    for x in [-cliff_w, cliff_w]:
        for y in np.arange(rock_front_y, cliff_back_y + 1e-9, 0.110):
            for z in np.arange(0.0, cliff_h + 1e-9, 0.105):
                yy = float(y)
                zz = float(z)
                xx = float(x + 0.055 * math.sin(2.1 * yy + 0.6 * zz) * (1 if x > 0 else -1))
                points.append((xx, yy, zz, "rock"))

    # Top ledge and shallow back shelf, useful when the user orbits the camera.
    for z in [cliff_h - 0.02, cliff_h - 0.16]:
        for x in np.arange(-cliff_w, cliff_w + 1e-9, 0.120):
            for y in np.arange(rock_front_y, cliff_back_y + 1e-9, 0.120):
                points.append((float(x), float(y), float(z + 0.025 * math.sin(float(x))), "rock"))

    # A few shallow cave/ravine rings carved into the cliff margins.
    for cx, cz, rx, rz in [(-4.75, 1.90, 0.44, 0.34), (-4.55, 3.25, 0.34, 0.31),
                           (4.65, 2.22, 0.40, 0.34), (4.40, 4.18, 0.38, 0.32)]:
        for theta in np.linspace(0, 2 * math.pi, 75, endpoint=False):
            x = cx + rx * math.cos(float(theta))
            z = cz + rz * math.sin(float(theta))
            for y in np.linspace(rock_front_y - 0.22, rock_front_y + 0.18, 5):
                points.append((float(x), float(y), float(z), "niche_edge"))

    # ------------------------------------------------------------
    # 2. Carved facade sheets and returns.
    # ------------------------------------------------------------
    for x in np.arange(-facade_w, facade_w + 1e-9, 0.065):
        for z in np.arange(0.0, facade_h + 1e-9, 0.065):
            xf = float(x)
            zf = float(z)
            if doorway_opening(xf, zf) or side_niche_opening(xf, zf):
                continue
            # Multi-depth relief: lower body is more deeply carved than the high
            # cliff, so orbiting shows real parallax.
            relief = 0.016 * math.sin(2.7 * xf) * math.sin(1.45 * zf)
            y = facade_y + relief
            if zf < 0.38:
                label = "base"
            elif 3.05 < zf < 3.55 or zf > 5.05:
                label = "cornice"
            else:
                label = "facade"
            points.append((xf, y, zf, label))

    # Left/right and top facade returns connect the carved face into the cliff.
    for x in [-facade_w, facade_w]:
        for y in np.arange(facade_y, rock_front_y + 1e-9, 0.065):
            for z in np.arange(0.0, facade_h + 1e-9, 0.080):
                points.append((float(x), float(y), float(z), "facade"))
    for z in [facade_h - 0.02]:
        for x in np.arange(-facade_w, facade_w + 1e-9, 0.080):
            for y in np.arange(facade_y, rock_front_y + 1e-9, 0.070):
                points.append((float(x), float(y), float(z), "cornice"))

    # Deep doorway recess with side jambs, top lintel, and back wall.
    door_x = 0.72
    door_y0 = -0.82
    for x in [-door_x, door_x]:
        for y in np.arange(door_y0, facade_y + 1e-9, 0.065):
            for z in np.arange(0.20, 2.18 + 1e-9, 0.060):
                points.append((float(x), float(y), float(z), "doorway_edge"))
    for x in np.arange(-door_x, door_x + 1e-9, 0.060):
        for y in np.arange(door_y0, facade_y + 1e-9, 0.065):
            points.append((float(x), float(y), 2.18, "doorway_edge"))
    for x in np.arange(-door_x, door_x + 1e-9, 0.075):
        for z in np.arange(0.25, 2.08 + 1e-9, 0.075):
            points.append((float(x), float(door_y0), float(z), "doorway_edge"))

    # Side niche outlines and shallow recesses.
    for cx, cz, rx, rz in [(-2.45, 2.04, 0.33, 0.44), (2.45, 2.04, 0.33, 0.44),
                           (-2.35, 4.18, 0.28, 0.36), (2.35, 4.18, 0.28, 0.36)]:
        for theta in np.linspace(0, 2 * math.pi, 70, endpoint=False):
            x = cx + rx * math.cos(float(theta))
            z = cz + rz * math.sin(float(theta))
            for y in np.linspace(facade_y - 0.32, facade_y + 0.02, 5):
                points.append((float(x), float(y), float(z), "niche_edge"))
        for x in np.arange(cx - rx * 0.75, cx + rx * 0.75 + 1e-9, 0.060):
            for z in np.arange(cz - rz * 0.60, cz + rz * 0.60 + 1e-9, 0.060):
                if ((x - cx) / rx) ** 2 + ((z - cz) / rz) ** 2 < 0.70:
                    points.append((float(x), facade_y - 0.28, float(z), "niche_edge"))

    # ------------------------------------------------------------
    # 3. Protruding columns, cornices, pediment, tholos, stairs.
    # ------------------------------------------------------------
    def add_column(cx: float, z0: float, z1: float, radius: float, label: str = "column") -> None:
        # Mostly front half of a real cylinder, embedded into the facade at the
        # rear.  This gives visible 3D depth without creating isolated tubes.
        for z in np.arange(z0, z1 + 1e-9, 0.060):
            for t in np.linspace(0, 2 * math.pi, 36, endpoint=False):
                st = math.sin(float(t))
                if st > 0.38:
                    continue
                x = cx + radius * math.cos(float(t))
                y = facade_y - 0.23 + radius * st
                points.append((float(x), float(y), float(z), label))
        # capital/base block rings with thickness in y.
        for z in [z0, z0 + 0.18, z1 - 0.18, z1]:
            for x in np.arange(cx - radius * 1.9, cx + radius * 1.9 + 1e-9, 0.035):
                for y in np.linspace(facade_y - 0.42, facade_y + 0.02, 7):
                    points.append((float(x), float(y), float(z), label))

    for cx in [-3.10, -1.78, 1.78, 3.10]:
        add_column(cx, 0.46, 3.10, 0.18)
    for cx in [-0.95, 0.95]:
        add_column(cx, 0.46, 2.30, 0.12)

    # Thick cornices and base slabs with genuine depth.
    for z, depth, label in [(0.34, 0.48, "base"), (3.12, 0.50, "cornice"),
                            (3.44, 0.44, "cornice"), (5.12, 0.42, "cornice")]:
        for x in np.arange(-4.15, 4.15 + 1e-9, 0.055):
            for y in np.linspace(facade_y - depth, facade_y + 0.05, 9):
                points.append((float(x), float(y), float(z), label))

    # Lower triangular pediment as a thick sloped relief surface.
    def pediment_top(x: float) -> float:
        return 3.42 + max(0.0, 1.05 * (1.0 - abs(x) / 3.35))

    for x in np.arange(-3.35, 3.35 + 1e-9, 0.060):
        top = pediment_top(float(x))
        for z in np.arange(3.42, top + 1e-9, 0.060):
            for y in np.linspace(facade_y - 0.24, facade_y - 0.05, 3):
                points.append((float(x), float(y), float(z), "pediment"))
    for side in [-1, 1]:
        for u in np.linspace(0, 1, 120):
            x = side * 3.35 * u
            z = 3.42 + 1.05 * (1 - u)
            for y in np.linspace(facade_y - 0.42, facade_y + 0.02, 7):
                points.append((float(x), float(y), float(z), "pediment"))

    # Upper tholos / circular shrine with depth.
    for cx in [-0.82, -0.42, 0.42, 0.82]:
        add_column(cx, 4.12, 5.12, 0.085, label="tholos")
    for theta in np.linspace(0, math.pi, 90):
        for r in [0.56, 0.74]:
            x = r * math.cos(float(theta))
            z = 4.48 + 0.74 * math.sin(float(theta))
            for y in np.linspace(facade_y - 0.38, facade_y + 0.03, 6):
                points.append((float(x), float(y), float(z), "tholos"))
    for theta in np.linspace(0, 2 * math.pi, 42, endpoint=False):
        for z in np.arange(5.22, 5.66, 0.040):
            radius = 0.13 + 0.06 * math.sin((z - 5.22) * math.pi / 0.44)
            x = radius * math.cos(float(theta))
            y = facade_y - 0.18 + 0.75 * radius * math.sin(float(theta))
            points.append((float(x), float(y), float(z), "tholos"))

    # Wide front stairs / plaza blocks.
    for step, z in enumerate([0.00, 0.10, 0.20, 0.30, 0.40]):
        y0 = -1.72 + step * 0.27
        for x in np.arange(-4.45 + 0.15 * step, 4.45 - 0.15 * step + 1e-9, 0.085):
            for y in np.arange(y0, y0 + 0.24 + 1e-9, 0.060):
                points.append((float(x), float(y), float(z), "stair"))
        # vertical riser face
        for x in np.arange(-4.45 + 0.15 * step, 4.45 - 0.15 * step + 1e-9, 0.090):
            for zz in np.arange(max(0.0, z - 0.10), z + 1e-9, 0.040):
                points.append((float(x), float(y0), float(zz), "stair"))

    return points


def petra_damage_mask(case: str, x: float, y: float, z: float, label: str) -> bool:
    if label not in STRUCTURAL_PETRA:
        return False

    if case == "left_colonnade_collapse":
        # A realistic collapse on the left lower facade: columns, base, and part
        # of the lower pediment/entablature are gone while the rock remains.
        if label in {"column", "base", "cornice", "facade"} and x < -1.25 and 0.25 < z < 3.45:
            return True
        if label in {"pediment", "niche_edge"} and x < -0.85 and 2.70 < z < 4.15:
            return True
        if label == "stair" and x < -1.65 and z < 0.26:
            return True

    elif case == "pediment_blast":
        # Upper ceremonial damage: central pediment, tholos/urn, and upper cornice.
        if label in {"pediment", "tholos", "cornice", "facade"} and -1.55 < x < 1.55 and z > 3.38:
            return True
        if label == "niche_edge" and abs(x) < 2.6 and z > 3.75:
            return True

    elif case == "diagonal_facade_shear":
        # A diagonal fracture through the facade and columns.  This is harder but
        # still coherent so the red particles form a continuous missing surface.
        band = x + 0.72 * (z - 2.15)
        if label in {"facade", "column", "cornice", "pediment", "doorway_edge", "niche_edge"} and -0.28 < band < 0.48 and 0.55 < z < 5.25:
            return True
        if label == "stair" and -0.45 < x < 0.85 and z < 0.24:
            return True

    return False


def generate_petra_treasury(case: str, seed: int = 41) -> tuple[list[PointRow], list[PointRow], list[PointRow]]:
    if case not in PETRA_CASES:
        raise ValueError(f"Unknown Petra Treasury case: {case}")

    rng = np.random.default_rng(seed)
    reference = petra_treasury_complete(seed=seed)
    observed: list[PointRow] = []
    missing: list[PointRow] = []

    for row in reference:
        x, y, z, label = row
        if petra_damage_mask(case, x, y, z, label):
            missing.append(row)
        else:
            observed.append(row)

    if case == "left_colonnade_collapse":
        add_rubble(observed, rng, 620, (-3.8, -1.05), (-1.30, -0.05), 0.90)
    elif case == "pediment_blast":
        add_rubble(observed, rng, 540, (-1.55, 1.55), (-0.95, 0.05), 0.75)
    elif case == "diagonal_facade_shear":
        add_rubble(observed, rng, 650, (-0.8, 1.5), (-1.20, -0.05), 0.95)

    return observed, missing, reference


def leaning_tower_complete(seed: int = 51) -> list[PointRow]:
    """Complete synthetic Leaning Tower of Pisa style point cloud.

    The tower is intentionally modeled as a 3D leaning cylindrical monument with
    ring galleries, repeated arches, columns, top bell chamber, and base steps.
    Coordinates are synthetic and suitable for testing reconstruction only.
    """
    points: list[PointRow] = []
    rng = np.random.default_rng(seed)

    height = 6.6
    lean = 0.105  # x-shift per unit height; visually clear but not extreme
    radius = 1.28
    floors = 8
    floor_h = height / floors
    theta_steps = 216

    def transform(xl: float, yl: float, z: float) -> tuple[float, float, float]:
        # The tower leans in +X as z increases.  Small surface roughness makes
        # the synthetic scan less sterile without changing the underlying shape.
        return (xl + lean * z, yl, z)

    def cyl_point(r: float, theta: float, z: float) -> tuple[float, float, float]:
        return transform(r * math.cos(theta), r * math.sin(theta), z)

    def angle_dist(a: float, b: float) -> float:
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    # Main cylindrical masonry wall with repeated arcade openings.
    openings_per_floor = 16
    for i in range(theta_steps):
        theta = (i / theta_steps) * 2.0 * math.pi
        for z in np.arange(0.20, height + 1e-9, 0.060):
            level = min(int(z / floor_h), floors - 1)
            local = z - level * floor_h
            center = round(theta / (2.0 * math.pi) * openings_per_floor) / openings_per_floor * 2.0 * math.pi
            d = angle_dist(theta, center)

            # Open arches on each gallery.  Edges remain as arcade points.
            arch_w = 0.050
            arch_base = 0.20 * floor_h
            arch_top = 0.70 * floor_h
            rounded_top = arch_top + 0.18 * floor_h * math.sqrt(max(0.0, 1.0 - d / arch_w))
            in_opening = d < arch_w and arch_base < local < rounded_top and z > 0.45 and z < 6.05

            if in_opening:
                if abs(d - arch_w) < 0.006 or abs(local - rounded_top) < 0.040:
                    points.append((*cyl_point(radius + 0.010, theta, float(z)), "arcade"))
                continue

            # Slight modulation imitates old stone courses and makes the object
            # less perfectly cylindrical while preserving the leaning tower shape.
            r = radius + 0.018 * math.sin(5 * z + 3 * math.sin(theta))
            points.append((*cyl_point(r, theta, float(z)), "tower_wall"))

    # Gallery rings / floor slabs.
    ring_zs = [0.18] + [k * floor_h for k in range(1, floors + 1)]
    for z in ring_zs:
        for band in [-0.045, 0.0, 0.045]:
            zz = min(height, max(0.05, z + band))
            for i in range(theta_steps):
                theta = (i / theta_steps) * 2.0 * math.pi
                points.append((*cyl_point(radius + 0.15, theta, float(zz)), "ring"))

    # External columns around galleries.
    column_count = 16
    for level in range(1, floors):
        z0 = level * floor_h + 0.08
        z1 = min((level + 1) * floor_h - 0.12, height - 0.15)
        for c in range(column_count):
            theta = (c / column_count) * 2.0 * math.pi
            for z in np.arange(z0, z1 + 1e-9, 0.065):
                # tiny circular column surface around its centerline
                for off in [-0.010, 0.010]:
                    points.append((*cyl_point(radius + 0.25 + off, theta, float(z)), "column"))

    # Top bell chamber: slightly smaller upper crown with larger openings.
    for i in range(theta_steps):
        theta = (i / theta_steps) * 2.0 * math.pi
        for z in np.arange(5.75, height + 0.25, 0.060):
            local = z - 5.75
            center = round(theta / (2.0 * math.pi) * 10) / 10 * 2.0 * math.pi
            d = angle_dist(theta, center)
            in_open = d < 0.070 and 0.16 < local < 0.62
            if in_open:
                if abs(d - 0.070) < 0.008:
                    points.append((*cyl_point(radius + 0.03, theta, float(z)), "arcade"))
                continue
            points.append((*cyl_point(radius + 0.05, theta, float(z)), "bell_chamber"))

    # Circular base plinth and step rings on ground.
    for z, r in [(0.00, 1.72), (0.08, 1.62), (0.16, 1.48)]:
        for rr in np.arange(0.0, r + 1e-9, 0.12):
            for i in range(0, theta_steps, 4):
                theta = (i / theta_steps) * 2.0 * math.pi
                x, y, zz = transform(rr * math.cos(theta), rr * math.sin(theta), z)
                points.append((x, y, zz, "base"))
        for i in range(theta_steps):
            theta = (i / theta_steps) * 2.0 * math.pi
            points.append((*cyl_point(r, theta, z), "base"))

    # Narrow spiral-like exterior stair/stone trace to provide a nontrivial 3D detail.
    for k in range(160):
        t = k / 159.0
        theta = 1.15 * math.pi + 1.4 * t
        z = 0.25 + 1.25 * t
        points.append((*cyl_point(radius + 0.34, theta, z), "stair"))

    return points


def leaning_tower_damage_mask(case: str, x: float, y: float, z: float, label: str) -> bool:
    """True means remove this structural point from the incomplete input."""
    height = 6.6
    lean = 0.105
    radius = 1.28
    # undo lean before angular tests
    xl = x - lean * z
    theta = math.atan2(y, xl) % (2.0 * math.pi)
    rho = math.sqrt(xl * xl + y * y) / max(radius, 1e-9)

    if label not in STRUCTURAL_LEANING_TOWER:
        return False

    if case == "upper_bell_collapse":
        # Top/bell chamber collapsed on the uphill side; includes ring and columns.
        if in_arc(theta, 0.10 * math.pi, 0.55 * math.pi) and z > 4.55:
            return True
        if label in {"ring", "column", "arcade"} and in_arc(theta, 0.05 * math.pi, 0.62 * math.pi) and z > 4.15:
            return True

    elif case == "lower_arcade_breach":
        # Large missing breach through lower galleries and nearby base.
        if in_arc(theta, 1.10 * math.pi, 1.55 * math.pi) and 0.35 < z < 3.10 and rho > 0.78:
            return True
        if label in {"base", "stair"} and in_arc(theta, 1.15 * math.pi, 1.52 * math.pi) and z < 0.45:
            return True

    elif case == "diagonal_tower_crack":
        # Diagonal demolition scar across the leaning shaft.
        diagonal = (theta - 1.55 * math.pi) + 0.42 * ((z / height) - 0.5)
        if abs(diagonal) < 0.17 and 0.55 < z < 5.90 and rho > 0.76:
            return True
        if abs(diagonal) < 0.22 and label in {"ring", "column"} and 0.60 < z < 5.90:
            return True

    elif case == "absurd_half_chop":
        # Irregular half-tower chop.  The missing sector twists as height
        # increases, so it does not form a symmetric left/right removal.
        twist = 0.26 * (z / height) + 0.08 * math.sin(2.8 * z)
        start = (0.82 + twist) * math.pi
        end = (1.82 + 0.10 * math.sin(1.5 * z)) * math.pi
        jagged = 0.06 * math.sin(5.0 * theta + 1.6 * z)
        if in_arc(theta, start, end) and z > 0.18 and rho > 0.58 + jagged:
            return True
        if label in {"base", "stair"} and in_arc(theta, start - 0.10, end + 0.06) and z < 0.55:
            return True
        if label == "bell_chamber" and in_arc(theta, start - 0.12, end + 0.10) and z > 5.65:
            return True

    return False


def generate_leaning_tower(case: str, seed: int = 51) -> tuple[list[PointRow], list[PointRow], list[PointRow]]:
    if case not in LEANING_TOWER_CASES:
        raise ValueError(f"Unknown Leaning Tower case: {case}")

    rng = np.random.default_rng(seed)
    reference = leaning_tower_complete(seed=seed)
    observed: list[PointRow] = []
    missing: list[PointRow] = []

    for row in reference:
        x, y, z, label = row
        if leaning_tower_damage_mask(case, x, y, z, label):
            missing.append(row)
        else:
            observed.append(row)

    # Labeled rubble/noise remains in input view, but is excluded from clean reconstruction.
    if case == "upper_bell_collapse":
        add_rubble(observed, rng, 420, (0.0, 2.7), (0.0, 2.2), 1.0)
    elif case == "lower_arcade_breach":
        add_rubble(observed, rng, 520, (-2.5, 0.1), (-2.2, -0.3), 0.8)
    elif case == "diagonal_tower_crack":
        add_rubble(observed, rng, 520, (-2.4, 1.6), (-1.8, 1.6), 0.9)
    elif case == "absurd_half_chop":
        add_rubble(observed, rng, 780, (-2.6, 2.8), (-2.5, 1.0), 1.05)

    return observed, missing, reference

def default_missing_output(output: Path) -> Path:
    return output.with_name(output.stem + "_missing.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test-case point cloud data.")
    parser.add_argument("--subject", required=True, choices=["roman_arena", "palmyra_arch", "leaning_tower"])
    parser.add_argument("--case", required=True, help="Test case id for the subject.")
    parser.add_argument("--output", required=True, help="Output incomplete Pinc CSV path.")
    parser.add_argument("--missing-output", default=None, help="Optional exact missing structural CSV path.")
    args = parser.parse_args()

    if args.subject == "roman_arena":
        observed, missing, reference = generate_roman_arena(args.case)
        ref_name = "roman_arena_complete.csv"
    elif args.subject == "palmyra_arch":
        observed, missing, reference = generate_palmyra_arch(args.case)
        ref_name = "palmyra_arch_complete.csv"
    elif args.subject == "leaning_tower":
        observed, missing, reference = generate_leaning_tower(args.case)
        ref_name = "leaning_tower_complete.csv"
    else:
        raise ValueError(args.subject)

    output = Path(args.output)
    missing_output = Path(args.missing_output) if args.missing_output else default_missing_output(output)

    write_csv(observed, output)
    write_csv(missing, missing_output)

    ref_dir = ROOT / "data" / "source"
    write_csv(reference, ref_dir / ref_name)

    print(f"Generated {len(observed)} observed points for {args.subject}/{args.case}: {output}")
    print(f"Generated {len(missing)} exact missing structural points: {missing_output}")


if __name__ == "__main__":
    main()
