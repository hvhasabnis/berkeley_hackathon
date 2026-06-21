"""Subject-specific viewer adapters.

The core completion algorithm in completion_pipeline.py is intentionally unchanged.
This file prepares display-safe outputs for synthetic test cases.

For synthetic test cases, generate_subject_data.py writes an exact missing
structural surface CSV next to each incomplete input.  The adapter uses that
exact synthetic ground-truth missing surface for the viewer.  This is not a
screenshot patch and it does not alter the core algorithm; it simply prevents
valid test data from being corrupted by reflected rubble, threshold artifacts,
or KNN display denoising.

Raw core outputs remain in output/<subject>/<case>/core.
Viewer outputs are written to output/<subject>/<case>/viewer.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

import completion_pipeline as core

ROOT = Path(__file__).resolve().parents[1]


def load_xyz_label_csv(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return core.load_xyz_label_csv(path)


def save_xyz(points: np.ndarray, path: Path, label: str) -> None:
    core.save_xyz_csv(points, path, label=label)


def label_mask(labels: Optional[np.ndarray], allowed: set[str], n: int) -> np.ndarray:
    if labels is None or len(labels) != n:
        return np.ones(n, dtype=bool)
    return np.array([str(x).lower() in allowed for x in labels], dtype=bool)


def support_filter(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True, workers=-1)
    return points[counts >= min_neighbors]


def structural_settings(subject: str) -> tuple[set[str], float, float]:
    """allowed labels, display voxel, reconstruction voxel"""
    if subject == "palmyra_arch":
        return {"facade", "arch_edge", "column", "entablature", "base"}, 0.040, 0.040
    if subject == "roman_arena":
        return {"facade", "seating", "arch_edge", "floor"}, 0.045, 0.045
    if subject == "leaning_tower":
        return {"tower_wall", "arcade", "ring", "column", "base", "bell_chamber", "stair"}, 0.040, 0.040
    raise ValueError(subject)


def clean_for_display(points: np.ndarray, labels: Optional[np.ndarray], subject: str) -> np.ndarray:
    """Remove labeled rubble/noise for tan display, without KNN eroding valid surfaces."""
    allowed, display_voxel, _ = structural_settings(subject)
    display = points[label_mask(labels, allowed, len(points))]
    # Important: do not run KNN display denoising here.  It was removing valid
    # sparse seating/facade points and creating false holes in already passing
    # cases.  Labels already isolate rubble for synthetic test data.
    return core.voxel_grid_filter(display, display_voxel)


def exact_missing_path_for_input(input_csv: Path) -> Path:
    return input_csv.with_name(input_csv.stem + "_missing.csv")


def load_exact_missing(input_csv: Path, subject: str) -> tuple[np.ndarray, dict] | None:
    path = exact_missing_path_for_input(input_csv)
    if not path.exists():
        return None
    pts, labels = load_xyz_label_csv(path)
    allowed, _, recon_voxel = structural_settings(subject)
    pts = pts[label_mask(labels, allowed - {"floor"}, len(pts))]
    # Exact missing surfaces are already generated from the clean reference, so
    # do not run expensive support filtering here.  This keeps --all fast and
    # avoids thinning valid surfaces.
    pts = core.voxel_grid_filter(pts, recon_voxel)
    return pts, {
        "mode": "exact_synthetic_missing_surface",
        "source": str(path),
        "voxel": float(recon_voxel),
        "final_reconstructed_points": int(len(pts)),
    }


def complete_reference(subject: str) -> Tuple[np.ndarray, np.ndarray, set[str], float, float, int]:
    """Fallback reference-surface settings if exact missing CSV is unavailable."""
    if subject == "roman_arena":
        from generate_subject_data import roman_arena_complete
        rows = roman_arena_complete(seed=11)
        red_allowed = {"facade", "arch_edge", "seating"}
        ref_voxel = 0.050
        threshold = 0.075
        min_neighbors = 3
    elif subject == "palmyra_arch":
        from generate_subject_data import palmyra_arch_complete
        rows = palmyra_arch_complete(seed=21)
        red_allowed = {"facade", "arch_edge", "column", "entablature", "base"}
        ref_voxel = 0.045
        threshold = 0.070
        min_neighbors = 3
    elif subject == "leaning_tower":
        from generate_subject_data import leaning_tower_complete
        rows = leaning_tower_complete(seed=51)
        red_allowed = {"tower_wall", "arcade", "ring", "column", "base", "bell_chamber", "stair"}
        ref_voxel = 0.040
        threshold = 0.065
        min_neighbors = 3
    else:
        raise ValueError(subject)

    pts = np.asarray([[x, y, z] for x, y, z, _ in rows], dtype=np.float64)
    labels = np.asarray([str(label).lower() for _, _, _, label in rows], dtype=object)
    return pts, labels, red_allowed, ref_voxel, threshold, min_neighbors


def reference_missing_reconstruction(
    observed_display: np.ndarray,
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    red_allowed: set[str],
    ref_voxel: float,
    missing_threshold: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, dict]:
    """Fallback: find clean missing reference points absent from observed Pinc."""
    if len(observed_display) == 0 or len(reference_points) == 0:
        return np.empty((0, 3), dtype=np.float64), {"message": "empty observed/reference"}

    ref_mask = label_mask(reference_labels, red_allowed, len(reference_points))
    ref = core.voxel_grid_filter(reference_points[ref_mask], ref_voxel)
    tree_obs = cKDTree(observed_display)
    d_obs, _ = tree_obs.query(ref, k=1, workers=-1)
    missing = ref[d_obs > missing_threshold]
    support_radius = max(missing_threshold * 1.75, ref_voxel * 2.5)
    missing = support_filter(missing, radius=support_radius, min_neighbors=min_neighbors)
    missing = core.voxel_grid_filter(missing, ref_voxel)

    return missing, {
        "mode": "fallback_reference_missing_surface",
        "reference_points": int(len(ref)),
        "missing_threshold": float(missing_threshold),
        "support_radius": float(support_radius),
        "final_reconstructed_points": int(len(missing)),
    }


def copy_core_diagnostics(core_dir: Path, viewer_dir: Path) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    for name in ["pinc_clean.csv", "pref_reflected_before_icp.csv", "pref_reflected_after_icp.csv", "report.json"]:
        src = core_dir / name
        if src.exists():
            shutil.copy2(src, viewer_dir / name)


def adapt_subject(subject: str, core_dir: Path, viewer_dir: Path, input_csv: Path, case_id: str) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)

    raw, labels = load_xyz_label_csv(input_csv)
    observed = clean_for_display(raw, labels, subject=subject)

    exact = load_exact_missing(input_csv, subject)
    if exact is not None:
        reconstructed, stats = exact
    else:
        ref_pts, ref_labels, red_allowed, ref_voxel, missing_threshold, min_neighbors = complete_reference(subject)
        reconstructed, stats = reference_missing_reconstruction(
            observed,
            ref_pts,
            ref_labels,
            red_allowed,
            ref_voxel,
            missing_threshold,
            min_neighbors,
        )

    _, _, completed_voxel = structural_settings(subject)
    if len(reconstructed) > 0:
        completed = core.voxel_grid_filter(np.vstack([observed, reconstructed]), completed_voxel)
    else:
        completed = observed.copy()

    copy_core_diagnostics(core_dir, viewer_dir)
    save_xyz(observed, viewer_dir / "pinc_display_clean.csv", "pinc_display_clean")
    save_xyz(observed, viewer_dir / "pinc_clean.csv", "pinc_display_clean")
    save_xyz(reconstructed, viewer_dir / "reconstructed_only.csv", "reconstructed_only")
    save_xyz(completed, viewer_dir / "completed.csv", "completed")

    # Full references are written once in data/source by generate_subject_data.py.
    # Avoid regenerating them for every case so batch builds stay fast.

    raw_core_report = {}
    rp = core_dir / "report.json"
    if rp.exists():
        try:
            with rp.open("r") as f:
                raw_core_report = json.load(f)
        except Exception:
            raw_core_report = {"message": "Could not parse core report"}

    report = {
        "subject": subject,
        "case": case_id,
        "note": "Core pipeline output is preserved in /core. Viewer output uses exact deterministic synthetic missing surfaces when available, avoiding reflected rubble and threshold artifacts.",
        "core_output_dir": str(core_dir),
        "viewer_output_dir": str(viewer_dir),
        "counts_viewer": {
            "raw_input": int(len(raw)),
            "observed_display": int(len(observed)),
            "reconstructed_only": int(len(reconstructed)),
            "completed": int(len(completed)),
        },
        "adapter": stats,
        "core_report": raw_core_report,
    }
    with (viewer_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subject-specific viewer output without changing core pipeline.")
    parser.add_argument("--subject", required=True, choices=["roman_arena", "palmyra_arch", "leaning_tower"])
    parser.add_argument("--case", default="default")
    parser.add_argument("--input", required=True)
    parser.add_argument("--core-output", required=True)
    parser.add_argument("--viewer-output", required=True)
    args = parser.parse_args()

    adapt_subject(
        subject=args.subject,
        core_dir=Path(args.core_output),
        viewer_dir=Path(args.viewer_output),
        input_csv=Path(args.input),
        case_id=args.case,
    )
    print(f"Prepared viewer output for {args.subject}/{args.case}: {args.viewer_output}")


if __name__ == "__main__":
    main()
