"""
Launch file for replay_yz_pin_v — MIT-control yz replay.

The first cycle uses hierarchical task-space velocity control and records the
actually executed q/qd/qdd. From the second cycle onward, that recorded joint
trajectory is tracked by sending MIT commands: [q_des, qd_des, kp, kd, tau_ff].
For now tau_ff is fixed to 0.0.

Topics published (subscribe in PlotJuggler):
  /replay_yz_pin_v_node/left_ee_actual        — actual left EE pose
  /replay_yz_pin_v_node/left_ee_desired       — desired left EE pose (planned in cycle 1, recorded FK after)
  /replay_yz_pin_v_node/right_ee_actual       — actual right EE pose
  /replay_yz_pin_v_node/right_ee_desired      — desired right EE pose (static in cycle 1, recorded FK after)
  /joint_states                          — actual joint positions/velocities
  /replay_yz_pin_v_node/joint_actual          — actual joint positions/velocities
  /replay_yz_pin_v_node/joint_desired         — desired joint positions/velocities
  /replay_yz_pin_v_node/left_joint_mit_cmd — left-arm MIT commands [q_des, qd_des, kp, kd, tau_ff]
  /replay_yz_pin_v_node/joint_actual_kinematics   — [q, qd, qdd] per joint
  /replay_yz_pin_v_node/joint_desired_kinematics  — [q_des, qd_des, qdd_des] per joint

Override parameters at runtime:
  ros2 launch robo_description replay_yz_pin_v.launch.py \
      Kp_x:=8.0 Kp_y:=8.0 Kp_z:=8.0 \
      Kq_posture:=2.0 \
      replay_Kp:=8.0 replay_Kd:=1.0 \
      mit_Kp:=220.0 mit_Kd:=20.0 \
      max_desired_qdd:=12.0 \
      figure8_frequency:=0.25 figure8_amplitude_x:=0.08 figure8_amplitude_z:=0.06
"""
import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Prepend MuJoCo lib path
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    mujoco_lib_path = "/home/kmd/.mujoco/mujoco-3.10.0/lib"
    new_ld_path = f"{mujoco_lib_path}:{ld_path}" if ld_path else mujoco_lib_path

    mujoco_lib = SetEnvironmentVariable("LD_LIBRARY_PATH", new_ld_path)

    # ---- Path arguments ----
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="/home/kmd/Marvin_Description-Robot_Description/urdf/marvin_pro/marvin_robot.urdf",
        description="Optional URDF file used only to publish robot_description"
    )
    mjcf_path_arg = DeclareLaunchArgument(
        "mjcf_path",
        default_value="/home/kmd/my_work_pkg/src/robo_description/model/marvin_v.xml",
        description="Path to the MuJoCo MIT-control MJCF model file"
    )

    # ---- Task-space velocity gains ----
    Kp_x_arg = DeclareLaunchArgument(
        "Kp_x", default_value="8.0",
        description="Task-space velocity proportional gain — X axis"
    )
    Kp_y_arg = DeclareLaunchArgument(
        "Kp_y", default_value="8.0",
        description="Task-space velocity proportional gain — Y axis"
    )
    Kp_z_arg = DeclareLaunchArgument(
        "Kp_z", default_value="8.0",
        description="Task-space velocity proportional gain — Z axis"
    )
    # ---- Joint posture velocity gain ----
    Kq_posture_arg = DeclareLaunchArgument(
        "Kq_posture", default_value="2.0",
        description="Posture velocity proportional gain"
    )
    replay_Kp_arg = DeclareLaunchArgument(
        "replay_Kp", default_value="8.0",
        description="Joint-space replay position-to-velocity gain"
    )
    replay_Kd_arg = DeclareLaunchArgument(
        "replay_Kd", default_value="1.0",
        description="Joint-space replay velocity feedback gain"
    )
    mit_Kp_arg = DeclareLaunchArgument(
        "mit_Kp", default_value="220.0",
        description="MIT joint stiffness gain"
    )
    mit_Kd_arg = DeclareLaunchArgument(
        "mit_Kd", default_value="20.0",
        description="MIT joint damping gain"
    )
    max_desired_qdd_arg = DeclareLaunchArgument(
        "max_desired_qdd", default_value="12.0",
        description="Slew-rate limit for published desired joint velocity (rad/s^2)"
    )

    # ---- Figure-8 trajectory parameters ----
    fig8_freq_arg = DeclareLaunchArgument(
        "figure8_frequency", default_value="1.00",
        description="Figure-8 frequency (Hz) — one full cycle per 1/freq seconds"
    )
    fig8_amp_x_arg = DeclareLaunchArgument(
        "figure8_amplitude_x", default_value="0.08",
        description="Figure-8 amplitude in the first trajectory axis (Y for yz, m)"
    )
    fig8_amp_z_arg = DeclareLaunchArgument(
        "figure8_amplitude_z", default_value="0.06",
        description="Figure-8 amplitude in the second axis (m)"
    )
    fig8_plane_arg = DeclareLaunchArgument(
        "figure8_plane", default_value="yz",
        description="Plane of the figure-8: 'xz', 'yz', or 'xy'"
    )

    # ---- Node ----
    replay_yz_pin_v_node = Node(
        package="robo_description",
        executable="replay_yz_pin_v",
        name="replay_yz_pin_v_node",
        output="screen",
        parameters=[{
            "urdf_path":           LaunchConfiguration("urdf_path"),
            "mjcf_path":           LaunchConfiguration("mjcf_path"),
            "Kp_x":                LaunchConfiguration("Kp_x"),
            "Kp_y":                LaunchConfiguration("Kp_y"),
            "Kp_z":                LaunchConfiguration("Kp_z"),
            "Kq_posture":          LaunchConfiguration("Kq_posture"),
            "replay_Kp":           LaunchConfiguration("replay_Kp"),
            "replay_Kd":           LaunchConfiguration("replay_Kd"),
            "mit_Kp":              LaunchConfiguration("mit_Kp"),
            "mit_Kd":              LaunchConfiguration("mit_Kd"),
            "max_desired_qdd":     LaunchConfiguration("max_desired_qdd"),
            "figure8_frequency":   LaunchConfiguration("figure8_frequency"),
            "figure8_amplitude_x": LaunchConfiguration("figure8_amplitude_x"),
            "figure8_amplitude_z": LaunchConfiguration("figure8_amplitude_z"),
            "figure8_plane":       LaunchConfiguration("figure8_plane"),
        }],
    )

    return LaunchDescription([
        mujoco_lib,
        urdf_path_arg,
        mjcf_path_arg,
        Kp_x_arg,
        Kp_y_arg,
        Kp_z_arg,
        Kq_posture_arg,
        replay_Kp_arg,
        replay_Kd_arg,
        mit_Kp_arg,
        mit_Kd_arg,
        max_desired_qdd_arg,
        fig8_freq_arg,
        fig8_amp_x_arg,
        fig8_amp_z_arg,
        fig8_plane_arg,
        replay_yz_pin_v_node,
    ])
