"""
Launch file for sin_yz_control — left-arm figure-8 in the yz plane.

The controller uses hierarchical inverse dynamics: left tool x first, left y/z
figure-8 second, right end-effector hold third, then joint posture in the
remaining null-space. Defaults are conservative to avoid torque saturation.

Topics published (subscribe in PlotJuggler):
  /sin_yz_control_node/left_ee_actual    — actual left EE pose
  /sin_yz_control_node/left_ee_desired   — desired left EE pose (figure-8)
  /sin_yz_control_node/right_ee_actual   — actual right EE pose
  /sin_yz_control_node/right_ee_desired  — desired right EE pose (static L-shape)
  /joint_states                            — all joint positions/velocities

Override parameters at runtime:
  ros2 launch robo_description sin_yz_control.launch.py \
      Kp_x:=2500.0 Kd_x:=350.0 Kp_y:=900.0 Kd_y:=180.0 Kp_z:=1200.0 Kd_z:=240.0 \
      Kq_posture:=300.0 Dq_posture:=45.0 \
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
        description="Path to the URDF model file"
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
        description="Posture stiffness gain (joint-space PD)"
    )
    Dq_posture_arg = DeclareLaunchArgument(
        "Dq_posture", default_value="45.0",
        description="Posture damping gain (joint-space PD)"
    )

    # ---- Desired mass (mass shaping) ----
    m_des_arg = DeclareLaunchArgument(
        "m_des", default_value="5.0",
        description="Desired task-space mass for impedance shaping"
    )

    # ---- Figure-8 trajectory parameters ----
    fig8_freq_arg = DeclareLaunchArgument(
        "figure8_frequency", default_value="0.25",
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
    sin_yz_control_node = Node(
        package="robo_description",
        executable="sin_yz_control",
        name="sin_yz_control_node",
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
        m_des_arg,
        fig8_freq_arg,
        fig8_amp_x_arg,
        fig8_amp_z_arg,
        fig8_plane_arg,
        sin_yz_control_node,
    ])
