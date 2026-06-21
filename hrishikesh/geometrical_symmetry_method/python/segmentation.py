from __future__ import annotations

import argparse
import os

import numpy as np
import open3d as o3d


def remove_ground(pcd: o3d.geometry.PointCloud,
                  distance_threshold: float = 0.05,
                  ransac_n: int = 3,
                  num_iterations: int = 1000):
    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    ground = pcd.select_by_index(inlier_idx)
    rest = pcd.select_by_index(inlier_idx, invert=True)
    return ground, rest, plane_model


def is_ground_like(plane_model, up_axis: int = 2, angle_tol_deg: float = 15.0) -> bool:
    normal = np.array(plane_model[:3])
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    up = np.zeros(3); up[up_axis] = 1.0
    cos_angle = abs(float(np.dot(normal, up)))  # abs: normal could point down
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
    return angle_deg < angle_tol_deg


# --------------------------------------------------------------------------- #
#  2. Vegetation removal
# --------------------------------------------------------------------------- #
def remove_vegetation(pcd: o3d.geometry.PointCloud,
                      green_dominance: float = 1.08,
                      roughness_radius: float = 0.08,
                      roughness_max_nn: int = 20,
                      roughness_threshold: float = 0.04):
    """Split off points that look like vegetation, using two independent cues:

    Cue 1 -- COLOR: a point is "green-dominant" if its green channel is
    meaningfully higher than red and blue (g > red*green_dominance AND
    g > blue*green_dominance). Cheap, fast, works when color exists.

    Cue 2 -- ROUGHNESS: fit a local plane to each point's neighborhood
    (within `roughness_radius`) and measure how far the point deviates from
    that plane. Tree foliage is geometrically "fuzzy" -- points scatter in
    all directions -- while masonry is locally flat. High deviation -> likely
    vegetation. This cue still works on uncolored scans.

    A point is removed if EITHER cue fires (color OR roughness) -- vegetation
    only needs one tell to be caught; we'd rather over-remove a few stray
    building points near tree-lines than leave whole trees in the "building"
    cluster, since the next stage assumes that cluster is clean architecture.

    Returns (vegetation, rest).
    """
    points = np.asarray(pcd.points)
    n = len(points)

    # --- cue 1: color ---
    green_mask = np.zeros(n, dtype=bool)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)  # already in [0,1]
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        green_mask = (g > r * green_dominance) & (g > b * green_dominance)

    # --- cue 2: local roughness (plane-fit residual) ---
    pcd_tmp = o3d.geometry.PointCloud(pcd)
    pcd_tmp.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=roughness_radius, max_nn=roughness_max_nn
        )
    )
    # Open3D doesn't expose per-point plane-fit residuals directly, so we
    # recompute it ourselves: for each point, look at its neighbors and see
    # how much they deviate from the point's own estimated normal plane.
    kdtree = o3d.geometry.KDTreeFlann(pcd_tmp)
    normals = np.asarray(pcd_tmp.normals)
    roughness = np.zeros(n)
    for i in range(n):
        _, idx, _ = kdtree.search_hybrid_vector_3d(
            points[i], roughness_radius, roughness_max_nn
        )
        if len(idx) < 4:
            continue
        neighbors = points[idx]
        centered = neighbors - neighbors.mean(axis=0)
        # distance of each neighbor from the point's local tangent plane
        dist_from_plane = np.abs(centered @ normals[i])
        roughness[i] = dist_from_plane.std()

    rough_mask = roughness > roughness_threshold

    veg_mask = green_mask | rough_mask
    vegetation = pcd.select_by_index(np.where(veg_mask)[0])
    rest = pcd.select_by_index(np.where(veg_mask)[0], invert=True)
    return vegetation, rest


# --------------------------------------------------------------------------- #
#  3. Clustering (separate disconnected blobs)
# --------------------------------------------------------------------------- #
def cluster_remaining(pcd: o3d.geometry.PointCloud,
                      eps: float = 0.08,
                      min_points: int = 30):
    """DBSCAN: group points into connected blobs by proximity.

    eps: two points closer than this (in the cloud's current units) are
    considered connected. min_points: a blob smaller than this is noise, not
    a real cluster (label -1).

    After ground + vegetation removal, what's left of an outdoor scene is
    usually: the building (one big, dense blob) plus scattered debris/noise
    (small blobs). We sort clusters by size, largest first, since the largest
    coherent leftover blob is almost always the building.

    Returns a list of point clouds, one per cluster, LARGEST FIRST. Noise
    points (label -1) are dropped.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return []

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters = []
    for label in sorted(set(labels) - {-1}):
        idx = np.where(labels == label)[0]
        clusters.append(pcd.select_by_index(idx))
    clusters.sort(key=lambda c: len(c.points), reverse=True)
    return clusters


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #
def segment_scene(pcd: o3d.geometry.PointCloud, verbose: bool = True):
    """Run the full segmentation pipeline.

    Returns a dict:
        {
          "ground":      PointCloud or None,
          "vegetation":  PointCloud,
          "building":    PointCloud   (largest remaining cluster)
          "other":       [PointCloud, ...]   (smaller leftover clusters, if any)
        }
    """
    def log(msg):
        if verbose:
            print(msg)

    n0 = len(pcd.points)
    log(f"  scene:              {n0:,} pts")

    ground, rest, plane_model = remove_ground(pcd)
    ground_ok = is_ground_like(plane_model)
    if not ground_ok:
        # the biggest plane found was NOT roughly horizontal (e.g. a wall or
        # roof dominated) -- don't trust it as "ground", put points back
        log(f"  ground candidate:   rejected (normal not horizontal -- "
            f"likely a wall/roof, not the floor)")
        ground = o3d.geometry.PointCloud()
        rest = pcd
    else:
        log(f"  ground removed:     {len(ground.points):,} pts")

    vegetation, building_candidates = remove_vegetation(rest)
    log(f"  vegetation removed:  {len(vegetation.points):,} pts")

    clusters = cluster_remaining(building_candidates)
    log(f"  remaining clusters:  {len(clusters)} "
        f"({[len(c.points) for c in clusters]} pts each, largest first)")

    building = clusters[0] if clusters else o3d.geometry.PointCloud()
    other = clusters[1:]

    log(f"  -> building cluster: {len(building.points):,} pts "
        f"({100*len(building.points)/max(n0,1):.1f}% of scene)")

    return {"ground": ground, "vegetation": vegetation, "building": building, "other": other}


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import preprocessing

    ap = argparse.ArgumentParser(description="Scene segmentation (geometric base, stage 1.5).")
    ap.add_argument("path", help="path to a .ply scene (partial scan)")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--save_dir", default=None, help="write each segment as a separate .ply here")
    args = ap.parse_args()

    clean, params = preprocessing.preprocess(args.path, voxel_size=args.voxel)
    print(f"\nSegmenting {args.path}")
    result = segment_scene(clean)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        def colored(pcd, rgb):
            out = o3d.geometry.PointCloud(pcd)
            if len(out.points):
                out.paint_uniform_color(rgb)
            return out

        if len(result["ground"].points):
            o3d.io.write_point_cloud(os.path.join(args.save_dir, "ground.ply"),
                                      colored(result["ground"], [0.55, 0.5, 0.4]))
        o3d.io.write_point_cloud(os.path.join(args.save_dir, "vegetation.ply"),
                                  colored(result["vegetation"], [0.3, 0.6, 0.25]))
        o3d.io.write_point_cloud(os.path.join(args.save_dir, "building.ply"),
                                  colored(result["building"], [0.68, 0.64, 0.55]))
        for i, c in enumerate(result["other"]):
            o3d.io.write_point_cloud(os.path.join(args.save_dir, f"other_{i}.ply"),
                                      colored(c, [0.6, 0.3, 0.3]))
        print(f"  wrote segments to {args.save_dir}/")
