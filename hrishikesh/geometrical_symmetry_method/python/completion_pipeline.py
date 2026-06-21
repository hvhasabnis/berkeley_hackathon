"""
Symmetry-based point-cloud completion pipeline.

Implements:
1. Voxel gridding
2. kNN average-distance denoising
3. Symmetry-plane detection by Chamfer minimization
4. ICP alignment of reflected cloud
5. Merge + fine voxel filtering
6. Bidirectional Chamfer verification between completed halves

Input CSV formats accepted:
- x,y,z
- x,y,z,label
- header row is optional
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def load_xyz_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
            except ValueError:
                continue
            rows.append((x, y, z))

    if not rows:
        raise ValueError(f"No XYZ points found in {path}")

    return np.asarray(rows, dtype=np.float64)



def load_xyz_label_csv(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load XYZ and optional label column.

    Labels are used only when they exist. Real scans usually will not have labels,
    so the pipeline has a geometry-based fallback below.
    """
    rows = []
    labels = []
    saw_label = False

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
            except ValueError:
                continue
            rows.append((x, y, z))
            if len(row) >= 4 and row[3].strip():
                labels.append(row[3].strip().lower())
                saw_label = True
            else:
                labels.append("")

    if not rows:
        raise ValueError(f"No XYZ points found in {path}")

    points = np.asarray(rows, dtype=np.float64)
    if saw_label:
        return points, np.asarray(labels, dtype=object)
    return points, None


def save_xyz_csv(points: np.ndarray, path: Path, label: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        if label is None:
            writer.writerow(["x", "y", "z"])
            writer.writerows(points.tolist())
        else:
            writer.writerow(["x", "y", "z", "label"])
            for p in points:
                writer.writerow([float(p[0]), float(p[1]), float(p[2]), label])


def voxel_grid_filter(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        return points.copy()

    keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.zeros(len(unique_keys), dtype=np.int64)

    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)

    return sums / counts[:, None]


def knn_average_distance_denoise(
    points: np.ndarray,
    k: int = 12,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if len(points) <= k + 1:
        return points.copy(), {
            "removed_points": 0,
            "threshold": 0.0,
            "mean_neighbor_distance": 0.0,
            "std_neighbor_distance": 0.0,
        }

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    avg = dists[:, 1:].mean(axis=1)

    mean = float(avg.mean())
    std = float(avg.std())
    threshold = mean + std_ratio * std
    keep = avg <= threshold

    return points[keep], {
        "removed_points": int((~keep).sum()),
        "threshold": float(threshold),
        "mean_neighbor_distance": mean,
        "std_neighbor_distance": std,
    }


def chamfer_distance(
    a: np.ndarray,
    b: np.ndarray,
    trim_fraction: float = 1.0,
    squared: bool = False,
) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    da, _ = tree_b.query(a, k=1, workers=-1)
    db, _ = tree_a.query(b, k=1, workers=-1)

    if squared:
        da = da * da
        db = db * db

    if trim_fraction < 1.0:
        trim_fraction = max(0.05, min(1.0, trim_fraction))
        na = max(1, int(len(da) * trim_fraction))
        nb = max(1, int(len(db) * trim_fraction))
        da = np.partition(da, na - 1)[:na]
        db = np.partition(db, nb - 1)[:nb]

    return float(0.5 * (da.mean() + db.mean()))


def reflect_points(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    signed = (points - plane_point) @ n
    return points - 2.0 * signed[:, None] * n[None, :]


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def bbox_center(points: np.ndarray) -> np.ndarray:
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def pca_horizontal_axes(points: np.ndarray) -> Tuple[float, float]:
    xy = points[:, :2]
    xy = xy - xy.mean(axis=0)
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    angle0 = math.atan2(eigvecs[1, 0], eigvecs[0, 0]) % math.pi
    angle1 = (angle0 + math.pi / 2.0) % math.pi
    return angle0, angle1


def find_symmetry_plane(
    points: np.ndarray,
    angle_step_deg: float = 6.0,
    fine_step_deg: float = 1.0,
    offset_fraction: float = 0.05,
    trim_fraction: float = 0.85,
    max_score_points: int = 1200,
    seed: int = 42,
    normal_angle_deg: Optional[float] = None,
) -> Dict[str, object]:
    center = bbox_center(points)
    score_points = sample_points(points, max_score_points, seed)
    score_tree = cKDTree(score_points)

    xy_extent = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
    offset_range = float(max(xy_extent) * offset_fraction)

    def trimmed_mean(values: np.ndarray, fraction: float) -> float:
        if fraction >= 1.0:
            return float(values.mean())
        n = max(1, int(len(values) * max(0.05, min(1.0, fraction))))
        return float(np.partition(values, n - 1)[:n].mean())

    def score(angle: float, offset: float) -> float:
        # Fast robust Chamfer proxy: reflect the sample and query against the fixed sample tree.
        # The exact bidirectional Chamfer is still computed later for reporting.
        n = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)
        p0 = center + offset * n
        reflected = reflect_points(score_points, p0, n)
        dists, _ = score_tree.query(reflected, k=1, workers=-1)
        return trimmed_mean(dists, trim_fraction)

    if normal_angle_deg is not None:
        base_angle = math.radians(normal_angle_deg) % math.pi
        search_angles = np.array([base_angle])
    else:
        pca0, pca1 = pca_horizontal_axes(points)
        coarse_step = math.radians(angle_step_deg)

        # Global coarse search plus PCA candidates. The global search is safer for damaged clouds.
        global_angles = np.arange(0.0, math.pi, coarse_step)
        pca_angles = np.array([pca0, pca1, 0.0, math.pi / 2.0])
        search_angles = np.unique(np.concatenate([global_angles, pca_angles]))

    offsets = np.linspace(-offset_range, offset_range, 9)

    best = {
        "score": float("inf"),
        "angle": 0.0,
        "offset": 0.0,
    }

    for angle in search_angles:
        for offset in offsets:
            s = score(float(angle), float(offset))
            if s < best["score"]:
                best.update(score=s, angle=float(angle), offset=float(offset))

    # Fine search around the best angle and offset.
    fine_angles = np.arange(
        best["angle"] - math.radians(angle_step_deg),
        best["angle"] + math.radians(angle_step_deg) + 1e-9,
        math.radians(fine_step_deg),
    )
    fine_offsets = np.linspace(
        best["offset"] - offset_range / 4.0,
        best["offset"] + offset_range / 4.0,
        9,
    )

    for angle in fine_angles:
        angle = float(angle % math.pi)
        for offset in fine_offsets:
            s = score(angle, float(offset))
            if s < best["score"]:
                best.update(score=s, angle=angle, offset=float(offset))

    normal = np.array([math.cos(best["angle"]), math.sin(best["angle"]), 0.0], dtype=np.float64)
    plane_point = center + best["offset"] * normal

    return {
        "plane_point": plane_point,
        "normal": normal,
        "normal_angle_degrees": float(math.degrees(best["angle"])),
        "offset": float(best["offset"]),
        "score": float(best["score"]),
        "bbox_center": center,
    }


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)

    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Prevent reflection in ICP. We already reflected the cloud intentionally.
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = tgt_centroid - r @ src_centroid
    return r, t


def apply_transform(points: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ r.T + t


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 30,
    trim_fraction: float = 0.70,
    tolerance: float = 1e-5,
    max_correspondence_distance: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(source) == 0 or len(target) == 0:
        return source.copy(), {"iterations": 0, "mean_error": float("inf")}

    current = source.copy()
    total_r = np.eye(3)
    total_t = np.zeros(3)
    tree = cKDTree(target)
    previous_error = float("inf")
    history = []

    for it in range(max_iterations):
        dists, indices = tree.query(current, k=1, workers=-1)
        keep = np.ones(len(current), dtype=bool)

        if trim_fraction < 1.0:
            threshold = np.quantile(dists, trim_fraction)
            keep &= dists <= threshold

        if max_correspondence_distance is not None:
            keep &= dists <= max_correspondence_distance

        if keep.sum() < 6:
            break

        src_corr = current[keep]
        tgt_corr = target[indices[keep]]

        r, t = best_fit_transform(src_corr, tgt_corr)
        current = apply_transform(current, r, t)

        total_r = r @ total_r
        total_t = r @ total_t + t

        mean_error = float(dists[keep].mean())
        history.append(mean_error)

        if abs(previous_error - mean_error) < tolerance:
            break
        previous_error = mean_error

    return current, {
        "iterations": len(history),
        "mean_error": float(history[-1]) if history else float("inf"),
        "error_history": history,
        "rotation": total_r.tolist(),
        "translation": total_t.tolist(),
    }


def completed_halves_chamfer(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> Dict[str, object]:
    n = normal / np.linalg.norm(normal)
    signed = (points - plane_point) @ n
    left = points[signed < 0]
    right = points[signed >= 0]

    if len(left) == 0 or len(right) == 0:
        return {
            "chamfer": float("inf"),
            "left_count": int(len(left)),
            "right_count": int(len(right)),
        }

    right_reflected = reflect_points(right, plane_point, n)
    cd = chamfer_distance(left, right_reflected, trim_fraction=0.95)

    return {
        "chamfer": float(cd),
        "left_count": int(len(left)),
        "right_count": int(len(right)),
    }


def estimate_point_spacing(points: np.ndarray) -> float:
    """Robust median nearest-neighbor spacing used to scale filtering thresholds."""
    if len(points) < 3:
        return 0.05
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2, workers=-1)
    spacing = float(np.median(dists[:, 1]))
    return max(spacing, 1e-6)





def estimate_ellipse_axes(points: np.ndarray) -> Tuple[float, float]:
    """Robust horizontal ellipse radii used for arena angular coordinates."""
    if len(points) == 0:
        return 1.0, 1.0
    a = float(np.quantile(np.abs(points[:, 0]), 0.995))
    b = float(np.quantile(np.abs(points[:, 1]), 0.995))
    return max(a, 1e-6), max(b, 1e-6)


def elliptic_theta_rho(points: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    x = points[:, 0] / a
    y = points[:, 1] / b
    theta = np.mod(np.arctan2(y, x), 2.0 * math.pi)
    rho = np.sqrt(x * x + y * y)
    return theta, rho


def select_structural_points(points: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """Select points that represent stable monument structure, not loose rubble.

    If the CSV includes synthetic labels, rubble and broken-edge particles are not
    used as reflection sources. If labels are absent, a geometry prior is used:
    keep the annular seating/facade region and reject low loose debris near the
    floor. This keeps the method general and prevents reflected rubble from being
    shown as reconstruction.
    """
    if len(points) == 0:
        return points.copy(), {"method": "empty", "kept": 0, "input": 0}

    if labels is not None and len(labels) == len(points):
        structural_labels = {"facade", "seating", "arch_edge"}
        keep = np.array([str(label).lower() in structural_labels for label in labels], dtype=bool)
        selected = points[keep]
        # If labels are present but unexpectedly remove almost everything, fall back.
        if len(selected) >= max(100, int(0.20 * len(points))):
            return selected, {
                "method": "label_aware",
                "kept": int(len(selected)),
                "input": int(len(points)),
                "excluded_labels": ["rubble", "broken_edge", "floor"],
            }

    a, b = estimate_ellipse_axes(points)
    _, rho = elliptic_theta_rho(points, a, b)
    z = points[:, 2]

    # Geometry fallback: annular monument region only. Excludes open floor and
    # low scattered rubble, but preserves seating and facade surfaces.
    keep = (rho > 0.42) & (rho < 1.08) & (z > 0.10)
    selected = points[keep]
    return selected, {
        "method": "geometry_fallback",
        "kept": int(len(selected)),
        "input": int(len(points)),
        "ellipse_axes": [float(a), float(b)],
    }




def select_display_points(points: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """Select visible input points for display and final merge.

    This removes loose rubble and jagged broken-edge particles from the tan cloud,
    because those are observed collapse debris rather than stable arena structure.
    It keeps the arena floor plus stable facade/seating/arch points. When labels
    are unavailable, it falls back to a generic geometric gate based on the arena
    annulus and center floor.
    """
    if len(points) == 0:
        return points.copy(), {"method": "empty", "kept": 0, "input": 0}

    if labels is not None and len(labels) == len(points):
        display_labels = {"facade", "seating", "arch_edge", "floor"}
        keep = np.array([str(label).lower() in display_labels for label in labels], dtype=bool)
        selected = points[keep]
        if len(selected) >= max(100, int(0.50 * len(points))):
            return selected, {
                "method": "label_aware",
                "kept": int(len(selected)),
                "input": int(len(points)),
                "excluded_labels": ["rubble", "broken_edge"],
            }

    a, b = estimate_ellipse_axes(points)
    _, rho = elliptic_theta_rho(points, a, b)
    z = points[:, 2]

    # Keep central floor and annular monument body. Reject low loose debris in the
    # outer annulus and random particles outside the monument envelope.
    central_floor = (rho <= 0.42) & (np.abs(z) <= 0.14)
    monument_body = (rho > 0.42) & (rho <= 1.08) & (z >= 0.08)
    keep = central_floor | monument_body
    selected = points[keep]
    return selected, {
        "method": "geometry_fallback",
        "kept": int(len(selected)),
        "input": int(len(points)),
        "ellipse_axes": [float(a), float(b)],
    }

def _circular_runs(mask: np.ndarray) -> list[Tuple[int, int, int]]:
    """Return circular true runs as (start, end_exclusive, length) in bin indices."""
    n = len(mask)
    if n == 0 or not mask.any():
        return []
    doubled = np.concatenate([mask, mask])
    runs: list[Tuple[int, int, int]] = []
    i = 0
    while i < 2 * n:
        if not doubled[i]:
            i += 1
            continue
        j = i
        while j < 2 * n and doubled[j]:
            j += 1
        if i < n:
            length = min(j - i, n)
            runs.append((i % n, (i + length) % n, length))
        i = j
    # Remove duplicate wrap-around run representations.
    unique = {}
    for start, end, length in runs:
        key = tuple(sorted([(start + k) % n for k in range(length)]))
        if key not in unique or length > unique[key][2]:
            unique[key] = (start, end, length)
    return list(unique.values())


def detect_dominant_missing_sector(
    structural_points: np.ndarray,
    nbins: int = 144,
    deficiency_ratio: float = 0.45,
    min_sector_degrees: float = 10.0,
    padding_degrees: float = 5.0,
) -> Dict[str, object]:
    """Detect the dominant missing angular sector from structural occupancy.

    This is the main anti-artifact step. It does not use screenshot coordinates.
    It finds the angular sector where the stable monument structure has the
    strongest sustained occupancy deficit, then only shows reconstruction there.
    Smaller partial asymmetries are left as observed damage instead of being
    over-completed.
    """
    if len(structural_points) == 0:
        return {"found": False, "message": "empty structural cloud"}

    a, b = estimate_ellipse_axes(structural_points)
    theta, rho = elliptic_theta_rho(structural_points, a, b)
    z = structural_points[:, 2]

    # Use the annular monument body. This avoids floor and low rubble influencing
    # missing-sector detection.
    body = (rho > 0.45) & (rho < 1.08) & (z > 0.25)
    if body.sum() < 100:
        body = np.ones(len(structural_points), dtype=bool)

    counts, _ = np.histogram(theta[body], bins=nbins, range=(0.0, 2.0 * math.pi))
    smooth = (
        np.roll(counts, 2)
        + np.roll(counts, 1)
        + counts
        + np.roll(counts, -1)
        + np.roll(counts, -2)
    ) / 5.0

    reference = float(np.quantile(smooth, 0.75))
    low = smooth < reference * deficiency_ratio

    min_bins = max(2, int(round(math.radians(min_sector_degrees) / (2.0 * math.pi) * nbins)))
    runs = [run for run in _circular_runs(low) if run[2] >= min_bins]

    if not runs:
        return {
            "found": False,
            "message": "no dominant missing angular sector detected",
            "ellipse_axes": [float(a), float(b)],
            "occupancy_reference": reference,
        }

    # Prefer the strongest sustained deficit, not simply the widest tiny dropout.
    def run_score(run: Tuple[int, int, int]) -> float:
        start, _, length = run
        idx = np.array([(start + k) % nbins for k in range(length)], dtype=int)
        deficit = np.maximum(0.0, reference - smooth[idx]).sum()
        return float(deficit * math.sqrt(length))

    best = max(runs, key=run_score)
    start, end, length = best
    pad_bins = max(1, int(round(math.radians(padding_degrees) / (2.0 * math.pi) * nbins)))
    start_padded = (start - pad_bins) % nbins
    length_padded = min(nbins, length + 2 * pad_bins)
    end_padded = (start_padded + length_padded) % nbins

    bin_width = 2.0 * math.pi / nbins
    start_angle = start_padded * bin_width
    end_angle = (start_padded + length_padded) * bin_width

    return {
        "found": True,
        "start_bin": int(start_padded),
        "end_bin": int(end_padded),
        "length_bins": int(length_padded),
        "start_angle_rad": float(start_angle % (2.0 * math.pi)),
        "end_angle_rad": float(end_angle % (2.0 * math.pi)),
        "start_angle_degrees": float(math.degrees(start_angle % (2.0 * math.pi))),
        "end_angle_degrees": float(math.degrees(end_angle % (2.0 * math.pi))),
        "raw_length_bins": int(length),
        "padding_bins": int(pad_bins),
        "ellipse_axes": [float(a), float(b)],
        "occupancy_reference": reference,
        "deficiency_ratio": float(deficiency_ratio),
        "candidate_runs": [
            {
                "start_degrees": float(math.degrees(run[0] * bin_width)),
                "end_degrees": float(math.degrees(((run[0] + run[2]) % nbins) * bin_width)),
                "length_bins": int(run[2]),
                "score": run_score(run),
            }
            for run in sorted(runs, key=run_score, reverse=True)[:6]
        ],
    }




def detect_missing_sectors(
    structural_points: np.ndarray,
    nbins: int = 144,
    deficiency_ratio: float = 0.65,
    min_sector_degrees: float = 8.0,
    padding_degrees: float = 6.0,
    max_sectors: int = 3,
    min_relative_score: float = 0.18,
) -> Dict[str, object]:
    """Detect multiple sustained missing angular sectors from structural occupancy.

    A Roman arena can have more than one missing/damaged end. Keeping only the
    single largest gap under-reconstructs the other end, while keeping every
    mismatch over-completes noise. This function keeps only major, coherent
    occupancy deficits measured from the cloud itself.
    """
    if len(structural_points) == 0:
        return {"found": False, "sectors": [], "message": "empty structural cloud"}

    a, b = estimate_ellipse_axes(structural_points)
    theta, rho = elliptic_theta_rho(structural_points, a, b)
    z = structural_points[:, 2]
    body = (rho > 0.45) & (rho < 1.08) & (z > 0.25)
    if body.sum() < 100:
        body = np.ones(len(structural_points), dtype=bool)

    counts, _ = np.histogram(theta[body], bins=nbins, range=(0.0, 2.0 * math.pi))
    smooth = (
        np.roll(counts, 2)
        + np.roll(counts, 1)
        + counts
        + np.roll(counts, -1)
        + np.roll(counts, -2)
    ) / 5.0

    reference = float(np.quantile(smooth, 0.75))
    low = smooth < reference * deficiency_ratio
    min_bins = max(2, int(round(math.radians(min_sector_degrees) / (2.0 * math.pi) * nbins)))
    runs = [run for run in _circular_runs(low) if run[2] >= min_bins]

    if not runs:
        return {
            "found": False,
            "sectors": [],
            "message": "no major missing angular sectors detected",
            "ellipse_axes": [float(a), float(b)],
            "occupancy_reference": reference,
            "deficiency_ratio": float(deficiency_ratio),
        }

    def run_score(run: Tuple[int, int, int]) -> float:
        start, _, length = run
        idx = np.array([(start + k) % nbins for k in range(length)], dtype=int)
        deficit = np.maximum(0.0, reference - smooth[idx]).sum()
        return float(deficit * math.sqrt(length))

    scored = sorted([(run_score(run), run) for run in runs], reverse=True)
    best_score = scored[0][0]
    pad_bins = max(1, int(round(math.radians(padding_degrees) / (2.0 * math.pi) * nbins)))
    bin_width = 2.0 * math.pi / nbins
    accepted = []

    occupied_bins = np.zeros(nbins, dtype=bool)
    for score_value, run in scored:
        if len(accepted) >= max_sectors:
            break
        if score_value < best_score * min_relative_score:
            continue
        start, _, length = run
        start_padded = (start - pad_bins) % nbins
        length_padded = min(nbins, length + 2 * pad_bins)
        idx = np.array([(start_padded + k) % nbins for k in range(length_padded)], dtype=int)
        # Avoid keeping overlapping sectors twice.
        if occupied_bins[idx].mean() > 0.40:
            continue
        occupied_bins[idx] = True
        start_angle = start_padded * bin_width
        end_angle = start_angle + length_padded * bin_width
        accepted.append({
            "start_bin": int(start_padded),
            "end_bin": int((start_padded + length_padded) % nbins),
            "length_bins": int(length_padded),
            "start_angle_rad": float(start_angle % (2.0 * math.pi)),
            "end_angle_rad": float(end_angle % (2.0 * math.pi)),
            "start_angle_degrees": float(math.degrees(start_angle % (2.0 * math.pi))),
            "end_angle_degrees": float(math.degrees(end_angle % (2.0 * math.pi))),
            "raw_length_bins": int(length),
            "padding_bins": int(pad_bins),
            "score": float(score_value),
        })

    return {
        "found": len(accepted) > 0,
        "sectors": accepted,
        "sector_count": int(len(accepted)),
        "ellipse_axes": [float(a), float(b)],
        "occupancy_reference": reference,
        "deficiency_ratio": float(deficiency_ratio),
        "candidate_runs": [
            {
                "start_degrees": float(math.degrees(run[0] * bin_width)),
                "end_degrees": float(math.degrees(((run[0] + run[2]) % nbins) * bin_width)),
                "length_bins": int(run[2]),
                "score": float(score_value),
            }
            for score_value, run in scored[:8]
        ],
    }

def angle_in_sector(theta: np.ndarray, start: float, end_unwrapped: float) -> np.ndarray:
    """Check if theta is inside a possibly wrapped sector."""
    twopi = 2.0 * math.pi
    theta_unwrapped = theta.copy()
    start_mod = start % twopi
    end_mod = end_unwrapped % twopi
    if end_unwrapped - start >= twopi - 1e-9:
        return np.ones_like(theta, dtype=bool)
    if start_mod <= end_mod:
        return (theta >= start_mod) & (theta <= end_mod)
    return (theta >= start_mod) | (theta <= end_mod)


def filter_reconstruction_to_missing_sectors(
    reconstructed: np.ndarray,
    structural_points: np.ndarray,
    fine_voxel: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Keep reconstructed points only inside major missing structural sectors.

    This fixes two problems at once:
    - false red particles from reflected rubble/asymmetric debris are rejected;
    - multiple missing ends can be completed, instead of only the single dominant gap.
    """
    if len(reconstructed) == 0:
        return reconstructed.copy(), {"input": 0, "final": 0, "message": "no reconstructed candidates"}

    sector_info = detect_missing_sectors(structural_points)
    if not sector_info.get("found", False):
        return np.empty((0, 3), dtype=np.float64), {
            "input": int(len(reconstructed)),
            "final": 0,
            "sector_info": sector_info,
            "message": "strict mode found no major sectors",
        }

    a, b = sector_info["ellipse_axes"]
    theta, rho = elliptic_theta_rho(reconstructed, float(a), float(b))
    z = reconstructed[:, 2]

    in_any_sector = np.zeros(len(reconstructed), dtype=bool)
    for sector in sector_info["sectors"]:
        start = float(sector["start_angle_rad"])
        end_unwrapped = start + (float(sector["length_bins"]) * 2.0 * math.pi / 144.0)
        in_any_sector |= angle_in_sector(theta, start, end_unwrapped)

    # Surface gate: preserve facade and stepped seating; reject floor speckles,
    # loose rubble, and points outside the shell. These are generic normalized
    # radial bands, not hand-marked screenshot coordinates.
    facade = (rho >= 0.90) & (rho <= 1.075) & (z >= 0.22)
    seating = (rho >= 0.43) & (rho < 0.92) & (z >= 0.16) & (z <= 2.90)
    surface_gate = facade | seating

    filtered = reconstructed[in_any_sector & surface_gate]

    spacing = estimate_point_spacing(structural_points)
    support_radius = max(spacing * 3.2, fine_voxel * 4.5)
    filtered, support_stats = radius_support_filter(filtered, support_radius, min_neighbors=4)
    filtered = voxel_grid_filter(filtered, fine_voxel)

    stats = {
        "input": int(len(reconstructed)),
        "after_sector_and_surface_gate": int((in_any_sector & surface_gate).sum()),
        "final": int(len(filtered)),
        "sector_info": sector_info,
        "surface_gate": {
            "facade_rho_range": [0.90, 1.075],
            "seating_rho_range": [0.43, 0.92],
            "min_facade_z": 0.22,
            "min_seating_z": 0.16,
            "max_seating_z": 2.90,
        },
        "support_filter": support_stats,
    }
    return filtered, stats


def radius_support_filter(
    points: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Remove unsupported isolated reconstructed particles."""
    if len(points) == 0:
        return points.copy(), {"before": 0, "after": 0, "removed": 0}

    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True, workers=-1)
    keep = counts >= min_neighbors
    return points[keep], {
        "before": int(len(points)),
        "after": int(keep.sum()),
        "removed": int((~keep).sum()),
        "radius": float(radius),
        "min_neighbors": int(min_neighbors),
    }


def radius_connected_components(points: np.ndarray, radius: float) -> list[list[int]]:
    """Connected components using radius-neighborhood graph."""
    if len(points) == 0:
        return []

    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, r=radius, workers=-1)
    seen = np.zeros(len(points), dtype=bool)
    components: list[list[int]] = []

    for start in range(len(points)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nb in neighborhoods[idx]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        components.append(component)

    return components


def filter_reconstructed_only(
    original: np.ndarray,
    reflected_aligned: np.ndarray,
    fine_voxel: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Conservative reconstructed-particle extraction.

    This avoids showing distorted red particles by keeping only points that are:
    1. not duplicates of the original cloud,
    2. locally supported by nearby reflected particles,
    3. part of a coherent missing-region component.

    Thresholds are scaled from the point spacing, so the filter is not tied to
    one manually selected screenshot or one hard-coded patch location.
    """
    if len(original) == 0 or len(reflected_aligned) == 0:
        return np.empty((0, 3), dtype=np.float64), {"message": "empty input"}

    spacing = estimate_point_spacing(original)
    duplicate_threshold = max(spacing * 2.45, fine_voxel * 3.5)
    support_radius = max(spacing * 3.6, fine_voxel * 5.0)
    component_radius = support_radius * 1.15
    min_support_neighbors = 5

    tree_original = cKDTree(original)
    nearest_dist, _ = tree_original.query(reflected_aligned, k=1, workers=-1)
    candidate = reflected_aligned[nearest_dist > duplicate_threshold]

    supported, support_stats = radius_support_filter(
        candidate,
        radius=support_radius,
        min_neighbors=min_support_neighbors,
    )

    components = radius_connected_components(supported, component_radius)
    component_sizes = sorted((len(c) for c in components), reverse=True)

    # Dynamic component threshold: keep coherent patches, reject scattered arch/noise speckles.
    min_component_points = max(180, int(0.07 * max(1, len(candidate))))
    min_component_span = max(spacing * 14.0, 0.75)

    kept_indices: list[int] = []
    kept_component_sizes = []

    for component in components:
        pts = supported[component]
        span = pts.max(axis=0) - pts.min(axis=0)
        diagonal_span = float(np.linalg.norm(span))
        if len(component) >= min_component_points and diagonal_span >= min_component_span:
            kept_indices.extend(component)
            kept_component_sizes.append(len(component))

    # If the data is very sparse and the dynamic rule becomes too strict, keep
    # only the largest coherent component instead of returning nothing.
    if not kept_indices and components:
        largest = max(components, key=len)
        if len(largest) >= 30:
            kept_indices.extend(largest)
            kept_component_sizes.append(len(largest))

    filtered = supported[np.asarray(kept_indices, dtype=np.int64)] if kept_indices else np.empty((0, 3), dtype=np.float64)
    filtered = voxel_grid_filter(filtered, fine_voxel)

    stats = {
        "spacing": float(spacing),
        "duplicate_threshold": float(duplicate_threshold),
        "candidate_count_after_duplicate_rejection": int(len(candidate)),
        "support_filter": support_stats,
        "component_radius": float(component_radius),
        "component_count": int(len(components)),
        "largest_component_sizes": component_sizes[:10],
        "min_component_points": int(min_component_points),
        "min_component_span": float(min_component_span),
        "kept_component_sizes": sorted(kept_component_sizes, reverse=True),
        "final_reconstructed_count": int(len(filtered)),
    }

    return filtered, stats


def pipeline(args: argparse.Namespace) -> Dict[str, object]:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, labels = load_xyz_label_csv(input_path)

    # Step 1: coarse voxel grid for the visible input cloud.
    pinc_voxel = voxel_grid_filter(raw, args.voxel)

    # Step 2: kNN outlier removal for the visible input cloud.
    pinc_clean, denoise_stats = knn_average_distance_denoise(
        pinc_voxel,
        k=args.knn,
        std_ratio=args.std_ratio,
    )

    # Use only stable structural points as the reflection source. This prevents
    # loose rubble or jagged broken debris from being mirrored as false red patches.
    structural_raw, structural_selection_stats = select_structural_points(raw, labels)
    structural_voxel = voxel_grid_filter(structural_raw, args.voxel)
    structural_clean, structural_denoise_stats = knn_average_distance_denoise(
        structural_voxel,
        k=min(args.knn, max(3, len(structural_voxel) - 2)),
        std_ratio=args.std_ratio,
    )

    # Use a cleaner observed cloud for display and final merge. This removes
    # loose rubble/broken-edge debris from tan points while preserving floor,
    # facade, seating, and arch points.
    display_raw, display_selection_stats = select_display_points(raw, labels)
    display_voxel = voxel_grid_filter(display_raw, args.voxel)
    display_clean, display_denoise_stats = knn_average_distance_denoise(
        display_voxel,
        k=min(args.knn, max(3, len(display_voxel) - 2)),
        std_ratio=args.std_ratio,
    )

    # Step 3: symmetry plane detection and reflection using the stable structure only.
    plane = find_symmetry_plane(
        structural_clean,
        angle_step_deg=args.angle_step_deg,
        fine_step_deg=args.fine_step_deg,
        offset_fraction=args.offset_fraction,
        trim_fraction=args.plane_trim,
        max_score_points=args.score_sample,
        seed=args.seed,
        normal_angle_deg=args.normal_angle_deg,
    )

    plane_point = np.asarray(plane["plane_point"], dtype=np.float64)
    normal = np.asarray(plane["normal"], dtype=np.float64)

    pref = reflect_points(structural_clean, plane_point, normal)
    pre_icp_cd = chamfer_distance(structural_clean, pref, trim_fraction=0.90)

    # Step 4: ICP misalignment correction, again using stable structure only.
    pref_icp, icp_stats = icp_point_to_point(
        pref,
        structural_clean,
        max_iterations=args.icp_iterations,
        trim_fraction=args.icp_trim,
        tolerance=args.icp_tolerance,
        max_correspondence_distance=args.max_correspondence_distance,
    )
    post_icp_cd = chamfer_distance(structural_clean, pref_icp, trim_fraction=0.90)

    # Step 5a: Extract reconstructed candidates, then keep only the dominant
    # missing sector. This is the strict anti-distortion pass requested for the
    # red overlay.
    reconstructed_candidates, reconstruction_candidate_stats = filter_reconstructed_only(
        pinc_clean,
        pref_icp,
        fine_voxel=args.fine_voxel,
    )
    reconstructed_only, missing_sector_stats = filter_reconstruction_to_missing_sectors(
        reconstructed_candidates,
        structural_clean,
        fine_voxel=args.fine_voxel,
    )

    # Step 5b: Merge and fine voxel grid. In strict mode, the completed cloud is
    # original cleaned Pinc + accepted reconstructed points, not the entire mirror.
    # This avoids over-completing smaller damage and avoids false reflected rubble.
    merged = np.vstack([display_clean, reconstructed_only])
    completed = voxel_grid_filter(merged, args.fine_voxel)

    # Step 6: verify symmetry of intact halves.
    halves = completed_halves_chamfer(completed, plane_point, normal)

    save_xyz_csv(pinc_clean, output_dir / "pinc_clean.csv", label="pinc_clean_all_observed")
    save_xyz_csv(display_clean, output_dir / "pinc_display_clean.csv", label="pinc_display_clean")
    save_xyz_csv(pref, output_dir / "pref_reflected_before_icp.csv", label="pref_before_icp")
    save_xyz_csv(pref_icp, output_dir / "pref_reflected_after_icp.csv", label="pref_after_icp")
    save_xyz_csv(reconstructed_only, output_dir / "reconstructed_only.csv", label="reconstructed_only")
    save_xyz_csv(completed, output_dir / "completed.csv", label="completed")

    report = {
        "input_file": str(input_path),
        "counts": {
            "raw": int(len(raw)),
            "after_voxel": int(len(pinc_voxel)),
            "after_denoise_all_observed": int(len(pinc_clean)),
            "display_clean": int(len(display_clean)),
            "structural_source": int(len(structural_clean)),
            "reflected_structural": int(len(pref_icp)),
            "reconstructed_candidates": int(len(reconstructed_candidates)),
            "reconstructed_only_strict": int(len(reconstructed_only)),
            "merged_before_fine_voxel": int(len(merged)),
            "completed": int(len(completed)),
        },
        "parameters": {
            "voxel": args.voxel,
            "fine_voxel": args.fine_voxel,
            "knn": args.knn,
            "std_ratio": args.std_ratio,
            "plane_trim": args.plane_trim,
            "icp_trim": args.icp_trim,
        },
        "denoise": denoise_stats,
        "display_source_selection": display_selection_stats,
        "display_denoise": display_denoise_stats,
        "structural_source_selection": structural_selection_stats,
        "structural_denoise": structural_denoise_stats,
        "reconstructed_candidate_filter": reconstruction_candidate_stats,
        "missing_sector_filter": missing_sector_stats,
        "symmetry_plane": {
            "plane_point": plane_point.tolist(),
            "normal": normal.tolist(),
            "normal_angle_degrees": plane["normal_angle_degrees"],
            "offset": plane["offset"],
            "optimization_score": plane["score"],
        },
        "chamfer": {
            "pinc_vs_pref_before_icp": float(pre_icp_cd),
            "pinc_vs_pref_after_icp": float(post_icp_cd),
            "final_bidirectional_halves": halves,
        },
        "icp": icp_stats,
        "bbox_completed": {
            "min": completed.min(axis=0).tolist(),
            "max": completed.max(axis=0).tolist(),
        },
    }

    with (output_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete an incomplete point cloud using symmetry reflection and ICP.")
    parser.add_argument("--input", required=True, help="Input CSV path containing x,y,z points.")
    parser.add_argument("--output", default="output", help="Output directory.")

    parser.add_argument("--voxel", type=float, default=0.055, help="Coarse voxel size before denoising.")
    parser.add_argument("--fine-voxel", type=float, default=0.035, help="Fine voxel size after merging.")

    parser.add_argument("--knn", type=int, default=12, help="k for kNN average-distance denoising.")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="Remove points whose kNN mean distance exceeds mean + std_ratio * std.")

    parser.add_argument("--angle-step-deg", type=float, default=6.0, help="Coarse symmetry-plane angle search step.")
    parser.add_argument("--fine-step-deg", type=float, default=1.0, help="Fine symmetry-plane angle search step.")
    parser.add_argument("--offset-fraction", type=float, default=0.04, help="Plane offset search range as fraction of XY extent.")
    parser.add_argument("--plane-trim", type=float, default=0.85, help="Trim fraction for robust plane Chamfer scoring.")
    parser.add_argument("--score-sample", type=int, default=1200, help="Max points used for plane scoring.")
    parser.add_argument("--normal-angle-deg", type=float, default=None, help="Optional fixed plane normal angle in XY plane. 0 means x-normal, 90 means y-normal.")

    parser.add_argument("--icp-iterations", type=int, default=30, help="Maximum ICP iterations.")
    parser.add_argument("--icp-trim", type=float, default=0.70, help="Fraction of closest correspondences used by ICP.")
    parser.add_argument("--icp-tolerance", type=float, default=1e-5, help="ICP convergence tolerance.")
    parser.add_argument("--max-correspondence-distance", type=float, default=None, help="Optional maximum ICP correspondence distance.")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    report = pipeline(parse_args())
    print(json.dumps(report, indent=2))
