"""
Launch file for MuJoCo simulate UI + Pinocchio impedance controller.
Full MuJoCo 3.x UI (panels, sliders, double-click perturbation, history, keyframes).

Override parameters at runtime:
  ros2 launch robo_description impedance_control.launch.py \
      mjcf_path:=/path/to/model.xml \
      urdf_path:=/path/to/model.urdf \
      Kp:=200.0 Kd:=30.0 M_pos:=2.0
"""
import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Prepend MuJoCo lib path to existing LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    mujoco_lib_path = "/home/kmd/.mujoco/mujoco-3.10.0/lib"
    new_ld_path = f"{mujoco_lib_path}:{ld_path}" if ld_path else mujoco_lib_path

    mujoco_lib = SetEnvironmentVariable("LD_LIBRARY_PATH", new_ld_path)

    # ---- Launch arguments ----
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="/home/kmd/Marvin_Description-Robot_Description/urdf/marvin_pro/marvin_robot.urdf",
        description="Path to the URDF model file"
    )

    mjcf_path_arg = DeclareLaunchArgument(
        "mjcf_path",
        default_value="/home/kmd/my_work_pkg/src/robo_description/model/marvin_pro_mink_with_gripper.xml",
        description="Path to the MuJoCo MJCF model file"
    )

    # Impedance gains – translational
    Kp_arg = DeclareLaunchArgument(
        "Kp", default_value="150.0",
        description="Translational stiffness gain"
    )
    Kd_arg = DeclareLaunchArgument(
        "Kd", default_value="25.0",
        description="Translational damping gain"
    )

    # Impedance gains – rotational
    Kp_rot_arg = DeclareLaunchArgument(
        "Kp_rot", default_value="180.0",
        description="Rotational stiffness gain"
    )
    Kd_rot_arg = DeclareLaunchArgument(
        "Kd_rot", default_value="25.0",
        description="Rotational damping gain"
    )

    # Joint-space damping
    Kd_joint_arg = DeclareLaunchArgument(
        "Kd_joint", default_value="3.0",
        description="Joint-space damping gain"
    )

    # Desired inertia (mass shaping) — NEW
    M_pos_arg = DeclareLaunchArgument(
        "M_pos", default_value="1.0",
        description="Desired translational inertia (mass shaping)"
    )
    M_rot_arg = DeclareLaunchArgument(
        "M_rot", default_value="0.5",
        description="Desired rotational inertia (mass shaping)"
    )

    # Trajectory
    transition_duration_arg = DeclareLaunchArgument(
        "transition_duration", default_value="4.0",
        description="Smoothstep transition duration from initial pose to target (seconds)"
    )

    # Target positions
    left_target_arg = DeclareLaunchArgument(
        "left_target", default_value="[0.0, 1.8, 1.12]",
        description="Left end-effector target position [x, y, z]"
    )
    right_target_arg = DeclareLaunchArgument(
        "right_target", default_value="[0.0, -1.8, 1.12]",
        description="Right end-effector target position [x, y, z]"
    )

    # ---- Node ----
    impedance_simulate_node = Node(
        package="robo_description",
        executable="impedance_simulate",
        name="impedance_simulate",
        output="screen",
        # Command-line args: argv[1]=mjcf_path, argv[2]=urdf_path
        arguments=[
            LaunchConfiguration("mjcf_path"),
            LaunchConfiguration("urdf_path"),
        ],
        parameters=[{
            "urdf_path":           LaunchConfiguration("urdf_path"),
            "mjcf_path":           LaunchConfiguration("mjcf_path"),
            "Kp":                  LaunchConfiguration("Kp"),
            "Kd":                  LaunchConfiguration("Kd"),
            "Kp_rot":              LaunchConfiguration("Kp_rot"),
            "Kd_rot":              LaunchConfiguration("Kd_rot"),
            "Kd_joint":            LaunchConfiguration("Kd_joint"),
            "M_pos":               LaunchConfiguration("M_pos"),
            "M_rot":               LaunchConfiguration("M_rot"),
            "transition_duration": LaunchConfiguration("transition_duration"),
            "left_target":         LaunchConfiguration("left_target"),
            "right_target":        LaunchConfiguration("right_target"),
        }],
    )

    return LaunchDescription([
        mujoco_lib,
        urdf_path_arg,
        mjcf_path_arg,
        Kp_arg,
        Kd_arg,
        Kp_rot_arg,
        Kd_rot_arg,
        Kd_joint_arg,
        M_pos_arg,
        M_rot_arg,
        transition_duration_arg,
        left_target_arg,
        right_target_arg,
        impedance_simulate_node,
    ])
