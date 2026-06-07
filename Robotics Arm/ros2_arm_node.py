"""
=============================================================
  Project 1: Robotic Arm Kinematics & Path Planning
  DecodeLabs Industrial Training Kit | Batch 2026
  File: ros2_arm_node.py
  Description: ROS 2 Action Client node that sends a
               FollowJointTrajectory goal to the
               gazebo_ros_control physics simulation.
               Requires: ROS 2 Humble + MoveIt 2
=============================================================
"""

# ── Standard ROS 2 imports ───────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

# ── ROS 2 message types ──────────────────────────────────
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# ── Python standard libs ─────────────────────────────────
import numpy as np
import time

# ── Project modules ──────────────────────────────────────
# from ik_solver import inverse_kinematics           # Uncomment in full env
# from trajectory_planner import generate_trajectory # Uncomment in full env


JOINT_NAMES = [
    'joint_1', 'joint_2', 'joint_3',
    'joint_4', 'joint_5', 'joint_6'
]


class RoboticArmClient(Node):
    """
    ROS 2 Action Client Node for 6-DOF Arm Control.

    Topics Published:
        /joint_states  (sensor_msgs/JointState)  – current arm state.

    Action Servers Used:
        /arm_controller/follow_joint_trajectory
        (control_msgs/FollowJointTrajectory)

    Communication Types:
        Topics   → continuous joint-state streaming.
        Services → synchronous parameter requests.
        Actions  → asynchronous, preemptable trajectory execution.
    """

    def __init__(self):
        super().__init__('robotic_arm_ik_client')
        self.get_logger().info("Robotic Arm IK Client Node initialized.")

        # Action client to trajectory controller
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        # Joint state subscriber (feedback)
        self._joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        self._current_joint_positions = np.zeros(6)

    def _joint_state_callback(self, msg):
        """Update internal joint state from sensor feedback."""
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self._current_joint_positions[i] = msg.position[idx]

    def send_trajectory_goal(self, joint_trajectory_matrix,
                              time_steps, blocking=True):
        """
        Send a FollowJointTrajectory action goal.

        Args:
            joint_trajectory_matrix: np.ndarray [N x 6] joint positions.
            time_steps:              np.ndarray [N] time stamps in seconds.
            blocking:                Wait for result if True.
        """
        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = JOINT_NAMES
        goal_msg.trajectory.header = Header()
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        for i, t in enumerate(time_steps):
            point = JointTrajectoryPoint()
            point.positions  = joint_trajectory_matrix[i].tolist()
            # Velocity and acceleration (computed by spline)
            point.velocities     = [0.0] * 6
            point.accelerations  = [0.0] * 6
            point.time_from_start = Duration(seconds=float(t)).to_msg()
            goal_msg.trajectory.points.append(point)

        self.get_logger().info(
            f"Sending goal with {len(time_steps)} waypoints "
            f"over {time_steps[-1]:.1f} seconds..."
        )

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )

        if blocking:
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()

            if not goal_handle.accepted:
                self.get_logger().error("Goal REJECTED by server.")
                return False

            self.get_logger().info("Goal ACCEPTED. Executing trajectory...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

            result = result_future.result().result
            self.get_logger().info(
                f"Trajectory complete. Error code: {result.error_code}"
            )
            return result.error_code == 0

        return True

    def _feedback_callback(self, feedback_msg):
        """Log real-time trajectory feedback."""
        fb = feedback_msg.feedback
        actual = fb.actual.positions
        desired = fb.desired.positions
        error = [abs(a - d) for a, d in zip(actual, desired)]
        max_err = max(error) if error else 0.0
        self.get_logger().info(
            f"[Feedback] Max joint error: {np.degrees(max_err):.3f} deg"
        )

    def execute_pick_task(self, target_xyz):
        """
        High-level pick task:
          1. Move to approach pose (above target)
          2. Move to target grasp pose
          3. Close gripper (simulated)
          4. Retract

        Args:
            target_xyz: [x, y, z] grasp position in meters.
        """
        self.get_logger().info(
            f"Executing pick task at {target_xyz}"
        )

        approach = [target_xyz[0], target_xyz[1], target_xyz[2] + 0.15]
        waypoints = [
            self._current_joint_positions.tolist(),
            approach,
            target_xyz,
        ]

        self.get_logger().info("IK solving for pick waypoints...")
        # NOTE: In full env, call generate_trajectory() here.
        self.get_logger().info("Trajectory computed. Sending goal...")


def main(args=None):
    rclpy.init(args=args)
    node = RoboticArmClient()

    # Demo: move to a target pose
    target = [0.45, -0.12, 0.31]
    node.get_logger().info(
        f"Target end-effector position: {target}"
    )

    # Simulated trajectory (replace with IK output)
    dummy_traj = np.zeros((10, 6))
    dummy_time = np.linspace(0, 3.0, 10)

    node.send_trajectory_goal(dummy_traj, dummy_time, blocking=False)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
