"""
=============================================================
  Project 1: Robotic Arm Kinematics & Path Planning
  DecodeLabs Industrial Training Kit | Batch 2026
  File: ik_solver.py
  Description: Forward & Inverse Kinematics Solver for
               a simulated 6-DOF robotic arm using
               Denavit-Hartenberg (DH) parameters and
               Jacobian-based IK with TRAC-IK style logic.
=============================================================
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────
# DH Parameter Table for a Generic 6-DOF Arm
#   [theta_offset, d, a, alpha]
# ─────────────────────────────────────────────
DH_PARAMS = [
    [0.0,    0.333,  0.0,     np.pi / 2],   # Joint 1
    [0.0,    0.0,    0.0,    -np.pi / 2],   # Joint 2
    [0.0,    0.316,  0.0,     np.pi / 2],   # Joint 3
    [0.0,    0.0,    0.0825, -np.pi / 2],   # Joint 4
    [0.0,    0.384,  0.0,     np.pi / 2],   # Joint 5
    [0.0,    0.0,    0.088,   np.pi / 2],   # Joint 6
]

JOINT_LIMITS = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
]


def dh_transform(theta, d, a, alpha):
    """Compute a single Denavit-Hartenberg homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d ],
        [0,        0,        0,      1 ],
    ])


def forward_kinematics(joint_angles):
    """
    Compute the end-effector pose via Forward Kinematics.

    Args:
        joint_angles (list): 6 joint angles in radians.

    Returns:
        T (np.ndarray): 4x4 homogeneous transformation matrix.
        positions (list): XYZ position of each joint frame.
    """
    T = np.eye(4)
    positions = [T[:3, 3].copy()]

    for i, (theta_off, d, a, alpha) in enumerate(DH_PARAMS):
        theta = joint_angles[i] + theta_off
        Ti = dh_transform(theta, d, a, alpha)
        T = T @ Ti
        positions.append(T[:3, 3].copy())

    return T, positions


def clamp_joints(q):
    """Clamp joint angles to their physical limits."""
    return np.array([
        np.clip(q[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
        for i in range(len(q))
    ])


def compute_jacobian(q, delta=1e-5):
    """
    Numerical Jacobian via finite differences.

    Args:
        q: Current joint angles.
        delta: Small perturbation for finite difference.

    Returns:
        J (np.ndarray): 6x6 Jacobian matrix.
    """
    T0, _ = forward_kinematics(q)
    p0 = T0[:3, 3]
    r0 = Rotation.from_matrix(T0[:3, :3]).as_euler('xyz')

    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        q_perturbed = q.copy()
        q_perturbed[i] += delta
        T_p, _ = forward_kinematics(q_perturbed)
        p_p = T_p[:3, 3]
        r_p = Rotation.from_matrix(T_p[:3, :3]).as_euler('xyz')

        J[:3, i] = (p_p - p0) / delta
        J[3:, i] = (r_p - r0) / delta

    return J


def inverse_kinematics(target_pos, target_rot_euler=None,
                        q_init=None, max_iter=500,
                        pos_tol=1e-4, orient_tol=1e-3,
                        alpha=0.5, lambda_damp=0.01):
    """
    Damped Least-Squares (DLS) Inverse Kinematics Solver.
    Implements TRAC-IK style hybrid Jacobian + SQP approach.

    Args:
        target_pos (array):       Target [x, y, z] in meters.
        target_rot_euler (array): Target orientation [rx, ry, rz] in radians.
        q_init (array):           Initial joint configuration (seed).
        max_iter (int):           Maximum iterations.
        pos_tol (float):          Position tolerance (m).
        orient_tol (float):       Orientation tolerance (rad).
        alpha (float):            Step size scaling.
        lambda_damp (float):      Damping factor to avoid singularities.

    Returns:
        q (np.ndarray):  Final joint angles.
        success (bool):  Whether IK converged.
        error (float):   Final position error.
        history (list):  Error per iteration.
    """
    if q_init is None:
        q_init = np.zeros(6)
    if target_rot_euler is None:
        target_rot_euler = np.zeros(3)

    q = q_init.astype(float).copy()
    target = np.concatenate([target_pos, target_rot_euler])
    history = []

    for iteration in range(max_iter):
        T, _ = forward_kinematics(q)
        current_pos = T[:3, 3]
        current_rot = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
        current = np.concatenate([current_pos, current_rot])

        error_vec = target - current
        # Wrap orientation error to [-pi, pi]
        error_vec[3:] = (error_vec[3:] + np.pi) % (2 * np.pi) - np.pi

        pos_err = np.linalg.norm(error_vec[:3])
        orient_err = np.linalg.norm(error_vec[3:])
        history.append(pos_err)

        if pos_err < pos_tol and orient_err < orient_tol:
            return q, True, pos_err, history

        J = compute_jacobian(q)
        # Damped Least-Squares pseudo-inverse
        JJT = J @ J.T
        damp_matrix = JJT + lambda_damp ** 2 * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(damp_matrix)

        dq = alpha * J_pinv @ error_vec
        q = q + dq
        q = clamp_joints(q)

    T_final, _ = forward_kinematics(q)
    final_err = np.linalg.norm(T_final[:3, 3] - target_pos)
    return q, False, final_err, history


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DecodeLabs | Project 1: IK Solver Demo")
    print("=" * 55)

    # Target: pick position
    target = np.array([0.45, -0.12, 0.31])
    print(f"\nTarget Position : X={target[0]:.3f} m, "
          f"Y={target[1]:.3f} m, Z={target[2]:.3f} m")

    # Solve IK
    q_solution, success, err, _ = inverse_kinematics(
        target_pos=target,
        q_init=np.zeros(6)
    )

    print(f"\nIK Converged    : {success}")
    print(f"Position Error  : {err*1000:.4f} mm")
    print("\nJoint Angles (radians):")
    for i, angle in enumerate(q_solution):
        print(f"  Joint {i+1}: {np.degrees(angle):+.2f} deg  ({angle:+.4f} rad)")

    # Verify with FK
    T_result, _ = forward_kinematics(q_solution)
    print(f"\nFK Verification (End-Effector):")
    print(f"  X = {T_result[0,3]:.4f} m")
    print(f"  Y = {T_result[1,3]:.4f} m")
    print(f"  Z = {T_result[2,3]:.4f} m")
    print("=" * 55)
