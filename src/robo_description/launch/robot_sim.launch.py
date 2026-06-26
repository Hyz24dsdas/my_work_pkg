"""
Launch file for the MuJoCo + Pinocchio robot simulator node.
Sets MuJoCo library path and loads the node with model parameters.
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

    robot_sim_node = Node(
        package="robo_description",
        executable="robo_des",
        name="robot_simulator",
        output="screen",
        parameters=[{
            "urdf_path": LaunchConfiguration("urdf_path"),
            "mjcf_path": LaunchConfiguration("mjcf_path"),
        }],
    )

    return LaunchDescription([
        mujoco_lib,
        urdf_path_arg,
        mjcf_path_arg,
        robot_sim_node,
    ])
