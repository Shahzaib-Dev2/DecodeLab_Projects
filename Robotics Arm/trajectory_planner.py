"""
=============================================================
  Project 1: Robotic Arm Kinematics & Path Planning
  DecodeLabs Industrial Training Kit | Batch 2026
  File: trajectory_planner.py
  Description: Smooth cubic spline trajectory generation
               between waypoints with collision-free
               time parameterization.
=============================================================
"""

import numpy as np
from scipy.interpolate import CubicSpline
from ik_solver import inverse_kinematics, forward_kinematics


def generate_trajectory(waypoints_xyz, total_time=5.0,
                         num_points=100, q_seed=None):
    """
    Generate a smooth, time-parameterized joint-space trajectory
    through a set of Cartesian waypoints using Cubic Spline interpolation.

    Pipeline:
        XYZ Waypoints → IK Solver (per waypoint) → Joint Space
        → Cubic Spline Fit → Time-Parameterized Trajectory
        → [q(t), q'(t), q''(t)]

    Args:
        waypoints_xyz (list): List of [x, y, z] Cartesian waypoints.
        total_time (float):   Total trajectory duration in seconds.
        num_points (int):     Number of interpolated time steps.
        q_seed (np.ndarray):  Seed joint configuration.

    Returns:
        t_fine (np.ndarray):       Time vector.
        q_traj (np.ndarray):       Joint positions  [N x 6].
        qd_traj (np.ndarray):      Joint velocities [N x 6].
        qdd_traj (np.ndarray):     Joint accels     [N x 6].
        joint_configs (list):      IK solutions at each waypoint.
        success_flags (list):      IK success per waypoint.
    """
    n_waypoints = len(waypoints_xyz)
    t_waypoints = np.linspace(0, total_time, n_waypoints)

    joint_configs = []
    success_flags = []

    q_current = np.zeros(6) if q_seed is None else q_seed.copy()

    for i, xyz in enumerate(waypoints_xyz):
        q_sol, success, err, _ = inverse_kinematics(
            target_pos=np.array(xyz),
            q_init=q_current
        )
        joint_configs.append(q_sol)
        success_flags.append(success)
        q_current = q_sol  # warm start for next waypoint
        status = "✓ OK" if success else f"✗ err={err:.4f}m"
        print(f"  Waypoint {i+1}/{n_waypoints} {xyz}  →  {status}")

    Q = np.array(joint_configs)  # shape: (n_waypoints, 6)

    # Fit cubic spline for each joint
    cs = CubicSpline(t_waypoints, Q, bc_type='clamped')

    t_fine = np.linspace(0, total_time, num_points)
    q_traj   = cs(t_fine)
    qd_traj  = cs(t_fine, 1)   # first derivative  → velocity
    qdd_traj = cs(t_fine, 2)   # second derivative → acceleration

    return t_fine, q_traj, qd_traj, qdd_traj, joint_configs, success_flags


def check_self_collision(q_traj, min_link_dist=0.05):
    """
    Simplified self-collision check along trajectory.
    Flags configurations where link proximity violates threshold.

    Args:
        q_traj: Joint trajectory [N x 6].
        min_link_dist: Minimum safe inter-link distance (m).

    Returns:
        collision_flags (list): True where potential collision detected.
    """
    collision_flags = []
    for q in q_traj:
        _, positions = forward_kinematics(q)
        flag = False
        positions = np.array(positions)
        # Check non-adjacent link distances
        for i in range(len(positions)):
            for j in range(i + 2, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_link_dist:
                    flag = True
        collision_flags.append(flag)
    return collision_flags


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DecodeLabs | Project 1: Trajectory Planner")
    print("=" * 55)

    waypoints = [
        [0.30,  0.00,  0.50],   # Start (Point A)
        [0.40, -0.10,  0.40],   # Via point 1
        [0.45, -0.12,  0.31],   # Target (Point B)
    ]

    print("\nGenerating trajectory through waypoints:")
    t, q, qd, qdd, configs, flags = generate_trajectory(
        waypoints, total_time=5.0, num_points=200
    )

    # Check collisions
    collisions = check_self_collision(q)
    n_collisions = sum(collisions)

    print(f"\nTrajectory Duration  : {t[-1]:.1f} s")
    print(f"Interpolated Points  : {len(t)}")
    print(f"Collision Detections : {n_collisions}")
    print(f"Max Joint Velocity   : {np.max(np.abs(qd)):.4f} rad/s")
    print(f"Max Joint Accel      : {np.max(np.abs(qdd)):.4f} rad/s²")
    print("=" * 55)
