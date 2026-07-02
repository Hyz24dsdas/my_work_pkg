"""
Launch file for replay_yz_pin — record first yz figure-8, then replay in joint space.

The first cycle uses the same planned yz figure-8 controller as sin_yz_pin and
records the actually executed q/qd/qdd. From the second cycle onward, that
recorded real joint trajectory is published as desired joint state and tracked
with joint-space PD + Pinocchio inverse dynamics.

Topics published (subscribe in PlotJuggler):
  /replay_yz_pin_node/left_ee_actual        — actual left EE pose
  /replay_yz_pin_node/left_ee_desired       — desired left EE pose (planned in cycle 1, recorded FK after)
  /replay_yz_pin_node/right_ee_actual       — actual right EE pose
  /replay_yz_pin_node/right_ee_desired      — desired right EE pose (static in cycle 1, recorded FK after)
  /joint_states                          — actual joint positions/velocities
  /replay_yz_pin_node/joint_actual          — actual joint positions/velocities
  /replay_yz_pin_node/joint_desired         — desired joint positions/velocities
  /replay_yz_pin_node/joint_actual_kinematics   — [q, qd, qdd] per joint
  /replay_yz_pin_node/joint_desired_kinematics  — [q_des, qd_des, qdd_des] per joint

Override parameters at runtime:
  ros2 launch robo_description replay_yz_pin.launch.py \
      Kp_x:=2500.0 Kd_x:=350.0 Kp_y:=900.0 Kd_y:=180.0 Kp_z:=1200.0 Kd_z:=240.0 \
      Kq_posture:=300.0 Dq_posture:=45.0 \
      replay_Kp:=120.0 replay_Kd:=25.0 \
      ilc_Lp:=15.0 ilc_Ld:=3.0 ilc_learning_rate:=0.35 \
      max_desired_qdd:=6.0 \
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
        default_value="/home/kmd/my_work_pkg/src/robo_description/model/marvin_pro_mink_with_gripper.xml",
        description="Path to the MuJoCo MJCF model file"
    )

    # ---- Task-space impedance gains ----
    Kp_x_arg = DeclareLaunchArgument(
        "Kp_x", default_value="2500.0",
        description="Task-space stiffness gain — X axis"
    )
    Kp_y_arg = DeclareLaunchArgument(
        "Kp_y", default_value="900.0",
        description="Task-space stiffness gain — Y axis"
    )
    Kp_z_arg = DeclareLaunchArgument(
        "Kp_z", default_value="1200.0",
        description="Task-space stiffness gain — Z axis"
    )
    Kd_x_arg = DeclareLaunchArgument(
        "Kd_x", default_value="350.0",
        description="Task-space damping gain — X axis"
    )
    Kd_y_arg = DeclareLaunchArgument(
        "Kd_y", default_value="180.0",
        description="Task-space damping gain — Y axis"
    )
    Kd_z_arg = DeclareLaunchArgument(
        "Kd_z", default_value="240.0",
        description="Task-space damping gain — Z axis"
    )

    # ---- Joint posture PD gains ----
    Kq_posture_arg = DeclareLaunchArgument(
        "Kq_posture", default_value="300.0",
        # "Kq_posture", default_value="0.0",
        description="Posture stiffness gain (joint-space PD)"
    )
    Dq_posture_arg = DeclareLaunchArgument(
        "Dq_posture", default_value="45.0",
        # "Dq_posture", default_value="0.0",

        description="Posture damping gain (joint-space PD)"
    )
    replay_Kp_arg = DeclareLaunchArgument(
        "replay_Kp", default_value="120.0",
        description="Joint-space replay stiffness gain used after the first recorded cycle"
    )
    replay_Kd_arg = DeclareLaunchArgument(
        "replay_Kd", default_value="25.0",
        description="Joint-space replay damping gain used after the first recorded cycle"
    )
    ilc_Lp_arg = DeclareLaunchArgument(
        "ilc_Lp", default_value="15.0",
        description="ILC proportional learning gain for qdd feed-forward update"
    )
    ilc_Ld_arg = DeclareLaunchArgument(
        "ilc_Ld", default_value="3.0",
        description="ILC derivative learning gain for qdd feed-forward update"
    )
    ilc_learning_rate_arg = DeclareLaunchArgument(
        "ilc_learning_rate", default_value="0.35",
        description="Scale applied to each cycle's learned qdd feed-forward correction"
    )
    ilc_max_delta_qdd_arg = DeclareLaunchArgument(
        "ilc_max_delta_qdd", default_value="8.0",
        description="Per-cycle qdd correction clamp before learning-rate scaling"
    )
    ilc_max_qdd_ff_arg = DeclareLaunchArgument(
        "ilc_max_qdd_ff", default_value="80.0",
        description="Absolute clamp for learned qdd feed-forward"
    )
    max_desired_qdd_arg = DeclareLaunchArgument(
        "max_desired_qdd", default_value="6.0",
        description="Slew-rate limit for published desired joint velocity (rad/s^2)"
    )

    # ---- Desired mass (mass shaping) ----
    m_des_arg = DeclareLaunchArgument(
        "m_des", default_value="5.0",
        description="Desired task-space mass for impedance shaping"
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
    replay_yz_pin_node = Node(
        package="robo_description",
        executable="replay_yz_pin",
        name="replay_yz_pin_node",
        output="screen",
        parameters=[{
            "urdf_path":           LaunchConfiguration("urdf_path"),
            "mjcf_path":           LaunchConfiguration("mjcf_path"),
            "Kp_x":                LaunchConfiguration("Kp_x"),
            "Kp_y":                LaunchConfiguration("Kp_y"),
            "Kp_z":                LaunchConfiguration("Kp_z"),
            "Kd_x":                LaunchConfiguration("Kd_x"),
            "Kd_y":                LaunchConfiguration("Kd_y"),
            "Kd_z":                LaunchConfiguration("Kd_z"),
            "Kq_posture":          LaunchConfiguration("Kq_posture"),
            "Dq_posture":          LaunchConfiguration("Dq_posture"),
            "replay_Kp":           LaunchConfiguration("replay_Kp"),
            "replay_Kd":           LaunchConfiguration("replay_Kd"),
            "ilc_Lp":              LaunchConfiguration("ilc_Lp"),
            "ilc_Ld":              LaunchConfiguration("ilc_Ld"),
            "ilc_learning_rate":   LaunchConfiguration("ilc_learning_rate"),
            "ilc_max_delta_qdd":   LaunchConfiguration("ilc_max_delta_qdd"),
            "ilc_max_qdd_ff":      LaunchConfiguration("ilc_max_qdd_ff"),
            "max_desired_qdd":     LaunchConfiguration("max_desired_qdd"),
            "m_des":               LaunchConfiguration("m_des"),
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
        Kd_x_arg,
        Kd_y_arg,
        Kd_z_arg,
        Kq_posture_arg,
        Dq_posture_arg,
        replay_Kp_arg,
        replay_Kd_arg,
        ilc_Lp_arg,
        ilc_Ld_arg,
        ilc_learning_rate_arg,
        ilc_max_delta_qdd_arg,
        ilc_max_qdd_ff_arg,
        max_desired_qdd_arg,
        m_des_arg,
        fig8_freq_arg,
        fig8_amp_x_arg,
        fig8_amp_z_arg,
        fig8_plane_arg,
        replay_yz_pin_node,
    ])
