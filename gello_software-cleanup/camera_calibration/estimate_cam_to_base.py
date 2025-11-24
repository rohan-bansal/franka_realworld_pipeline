#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path


def load_pairs(json_path):
    """
    Load (tag_xyz, eef_xyz) from your logging JSON.
    tag_xyz: AprilTag translation in camera frame
    eef_xyz: end-effector position in robot base frame
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        data = json.load(f)

    cam_points = []
    base_points = []
    for entry in data:
        cam_points.append(entry["tag_xyz"])
        base_points.append(entry["eef_xyz"])

    cam_points = np.asarray(cam_points, dtype=np.float64)  # shape (N, 3)
    base_points = np.asarray(base_points, dtype=np.float64)  # shape (N, 3)

    return cam_points, base_points


def estimate_rigid_transform(cam_points, base_points):
    """
    Find R, t such that:
        base_points ≈ R @ cam_points + t
    using SVD-based point cloud alignment (Kabsch / Procrustes).
    """
    assert cam_points.shape == base_points.shape
    assert cam_points.shape[1] == 3

    # 1. Compute centroids
    c_cam = cam_points.mean(axis=0)
    c_base = base_points.mean(axis=0)

    # 2. Remove centroids
    X = cam_points - c_cam
    Y = base_points - c_base

    # 3. Cross-covariance
    H = X.T @ Y  # shape (3, 3)

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 5. Ensure a proper rotation (det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6. Translation
    t = c_base - R @ c_cam

    # 7. Build 4x4 homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t


def compute_alignment_error(cam_points, base_points, R, t):
    """
    Compute RMS error after alignment.
    """
    pred_base = (R @ cam_points.T).T + t  # (N,3)
    diffs = base_points - pred_base
    err_per_point = np.linalg.norm(diffs, axis=1)
    rms = np.sqrt(np.mean(err_per_point ** 2))
    return rms, err_per_point


def main():
    parser = argparse.ArgumentParser(
        description="Estimate camera-to-robot-base transform from logged AprilTag/EEF pairs."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to april_eef_pairs_epXXX.json",
    )
    args = parser.parse_args()

    cam_points, base_points = load_pairs(args.json)
    print(f"Loaded {cam_points.shape[0]} pairs from {args.json}")

    # This returns the transform that maps camera frame -> base frame
    T_cam_to_base, R, t = estimate_rigid_transform(cam_points, base_points)

    # Inverse is base frame -> camera frame
    T_base_to_cam = np.linalg.inv(T_cam_to_base)

    rms, err_per_point = compute_alignment_error(cam_points, base_points, R, t)

    np.set_printoptions(precision=6, suppress=True)

    print("\n=== T_cam_to_base (camera -> robot base, 4x4) ===")
    print(T_cam_to_base)

    print("\n=== T_base_to_cam (robot base -> camera, 4x4) ===")
    print(T_base_to_cam)

    print("\nRotation R (used in T_cam_to_base):")
    print(R)

    print("\nTranslation t (used in T_cam_to_base):")
    print(t)

    print(f"\nRMS alignment error: {rms:.6f} (in same units as your logs, e.g. meters)")

    # Example usage for a single camera-frame point:
    example_cam = cam_points[0]
    homog_cam = np.concatenate([example_cam, [1.0]])
    mapped_base = T_cam_to_base @ homog_cam
    print("\nExample (camera -> base):")
    print("  p_cam  =", example_cam)
    print("  T_cam_to_base @ p_cam -> p_base ≈", mapped_base[:3])
    print("  corresponding logged eef_xyz =", base_points[0])

    # And the inverse direction (base -> cam) using the same point
    example_base = base_points[0]
    homog_base = np.concatenate([example_base, [1.0]])
    mapped_cam = T_base_to_cam @ homog_base
    print("\nExample (base -> camera):")
    print("  p_base =", example_base)
    print("  T_base_to_cam @ p_base -> p_cam ≈", mapped_cam[:3])
    print("  corresponding logged tag_xyz =", cam_points[0])

    def pretty_print_matrix(name, M):
        """Print a 4x4 matrix with commas so you can copy-paste it directly."""
        print(f"\n{name} = np.array([")
        for row in M:
            formatted = ", ".join(f"{val:.6f}" for val in row)
            print(f"    [{formatted}],")
        print("])")
    pretty_print_matrix("T_cam_to_base", T_cam_to_base)
    pretty_print_matrix("T_base_to_cam", T_base_to_cam)

if __name__ == "__main__":
    main()
