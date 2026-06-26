cmake_minimum_required(VERSION 3.8)
project(robo_description)

# set C++17 required by MuJoCo
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
# Workaround: get Pinocchio include dir from its cmake target
get_target_property(PINOCCHIO_INC pinocchio::pinocchio INTERFACE_INCLUDE_DIRECTORIES)
if(NOT PINOCCHIO_INC)
  set(PINOCCHIO_INC "/opt/ros/humble/include")
endif()
message(STATUS "Pinocchio include: ${PINOCCHIO_INC}")
message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIRS}")

# find MuJoCo
set(MUJOCO_ROOT "$ENV{HOME}/.mujoco/mujoco-3.10.0")
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
  PATHS ${MUJOCO_ROOT}/include
  NO_DEFAULT_PATH
)
find_library(MUJOCO_LIBRARY mujoco
  PATHS ${MUJOCO_ROOT}/lib
  NO_DEFAULT_PATH
)
if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
  message(FATAL_ERROR "MuJoCo not found at ${MUJOCO_ROOT}")
endif()
message(STATUS "MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
message(STATUS "MuJoCo library: ${MUJOCO_LIBRARY}")

# find GLFW3
find_package(PkgConfig)
pkg_check_modules(GLFW3 glfw3)

# executable: original MuJoCo physics simulator
add_executable(robo_des src/robo_des.cpp)
target_include_directories(robo_des PRIVATE
  ${MUJOCO_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIRS}
  ${PINOCCHIO_INC}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(robo_des
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
)
target_link_libraries(robo_des
  pinocchio::pinocchio
  ${MUJOCO_LIBRARY}
  ${GLFW3_LIBRARIES}
)

# executable: impedance-controlled simulator (bypassed MuJoCo physics)
add_executable(impedance_control_node src/impedance_control_node.cpp)
target_include_directories(impedance_control_node PRIVATE
  ${MUJOCO_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIRS}
  ${PINOCCHIO_INC}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(impedance_control_node
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
)
target_link_libraries(impedance_control_node
  pinocchio::pinocchio
  ${MUJOCO_LIBRARY}
  ${GLFW3_LIBRARIES}
)

# executable: L-shape + figure-8 trajectory + impedance control (full MuJoCo UI)
add_executable(sin_control
  src/sin_control.cpp
  simulate_mujoco/simulate.cc
  simulate_mujoco/glfw_adapter.cc
  simulate_mujoco/glfw_dispatch.cc
  simulate_mujoco/platform_ui_adapter.cc
  simulate_mujoco/lodepng.cpp
)
target_include_directories(sin_control PRIVATE
  ${MUJOCO_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIRS}
  ${PINOCCHIO_INC}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/simulate_mujoco
)
ament_target_dependencies(sin_control
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
  geometry_msgs
)
target_link_libraries(sin_control
  pinocchio::pinocchio
  ${MUJOCO_LIBRARY}
  ${GLFW3_LIBRARIES}
)

# install
install(TARGETS robo_des impedance_control_node sin_control
  DESTINATION lib/${PROJECT_NAME}
)

# executable: MuJoCo simulate UI + impedance controller
add_executable(impedance_simulate
  src/impedance_simulate_main.cpp
  simulate_mujoco/simulate.cc
  simulate_mujoco/glfw_adapter.cc
  simulate_mujoco/glfw_dispatch.cc
  simulate_mujoco/platform_ui_adapter.cc
  simulate_mujoco/lodepng.cpp
)
target_include_directories(impedance_simulate PRIVATE
  ${MUJOCO_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIRS}
  ${PINOCCHIO_INC}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/simulate_mujoco
)
ament_target_dependencies(impedance_simulate
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
)
target_link_libraries(impedance_simulate
  pinocchio::pinocchio
  ${MUJOCO_LIBRARY}
  ${GLFW3_LIBRARIES}
)

install(TARGETS impedance_simulate
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY model/
  DESTINATION share/${PROJECT_NAME}/model
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# copy model to build dir for running without install
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
