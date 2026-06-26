// Copyright 2024
//
// Impedance-controlled dual-arm robot with MuJoCo visualization.
// - MuJoCo: rendering + integration only (bypassed physics engine)
// - Pinocchio: kinematics (FK, Jacobian, gravity)
// - Impedance control in task-space per end-effector
// - ROS2: joint_states, TF, debug info
//
// Target: arms form a T-shape (extended horizontally sideways)

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

// ============================================================================
// SimulatorContext — MuJoCo rendering state (reused pattern from robo_des.cpp)
// ============================================================================
struct SimulatorContext
{
  mjModel*   mj_model = nullptr;
  mjData*    mj_data  = nullptr;
  mjvCamera  cam;
  mjvOption  opt;
  mjvScene   scn;
  mjrContext con;
  mjvPerturb pert;   // mouse perturbation (select + apply force/torque)

  double last_x = 0, last_y = 0;
  bool   button_left = false, button_right = false, button_middle = false;
  bool   paused = false;

  // Double-click detection for perturbation (force/torque on joints)
  double last_click_time  = 0.0;
  double last_click_x     = 0.0;
  double last_click_y     = 0.0;

  SimulatorContext()
  {
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_defaultPerturb(&pert);
  }
  ~SimulatorContext()
  {
    if (mj_data)  { mj_deleteData(mj_data);   mj_data  = nullptr; }
    if (mj_model) { mj_deleteModel(mj_model); mj_model = nullptr; }
  }
};

// ============================================================================
// GLFW callbacks
// ============================================================================
static void keyboard(GLFWwindow* w, int key, int /*scancode*/, int act, int /*mods*/)
{
  if (act != GLFW_PRESS) return;
  auto* ctx = static_cast<SimulatorContext*>(glfwGetWindowUserPointer(w));

  switch (key) {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(w, GLFW_TRUE);
    break;
  case GLFW_KEY_BACKSPACE:
    if (ctx->mj_model && ctx->mj_data) {
      mj_resetData(ctx->mj_model, ctx->mj_data);
      mj_forward(ctx->mj_model, ctx->mj_data);
    }
    // Also clear any active perturbation
    ctx->pert.active = 0;
    ctx->pert.select = -1;
    break;
  case GLFW_KEY_SPACE:
    ctx->paused = !ctx->paused;
    break;
  }
}

static void mouse_button(GLFWwindow* w, int button, int act, int /*mods*/)
{
  auto* ctx = static_cast<SimulatorContext*>(glfwGetWindowUserPointer(w));

  // ==========================================================================
  // Double-click detection → MuJoCo perturbation (select + apply force/torque)
  // ==========================================================================
  if (act == GLFW_PRESS) {
    double now = glfwGetTime();
    double cx, cy;
    glfwGetCursorPos(w, &cx, &cy);

    double dt   = now - ctx->last_click_time;
    double dist = std::sqrt((cx - ctx->last_click_x) * (cx - ctx->last_click_x) +
                            (cy - ctx->last_click_y) * (cy - ctx->last_click_y));

    ctx->last_click_time = now;
    ctx->last_click_x    = cx;
    ctx->last_click_y    = cy;

    // Double-click: within 350 ms and 5 px of the first click
    if (dt < 0.35 && dist < 5.0 && ctx->mj_model && ctx->mj_data) {
      int ww, wh;
      glfwGetWindowSize(w, &ww, &wh);

      mjtNum sel_pos[3];
      int geomid[1] = {-1}, flexid[1] = {-1}, skinid[1] = {-1};
      mjtNum aspect = static_cast<mjtNum>(ww) / static_cast<mjtNum>(wh);
      int sel_body = mjv_select(ctx->mj_model, ctx->mj_data, &ctx->opt,
                                aspect,
                                static_cast<mjtNum>(cx) / ww,
                                static_cast<mjtNum>(cy) / wh,
                                &ctx->scn, sel_pos,
                                geomid, flexid, skinid);

      if (sel_body > 0) {
        // Found a body — activate perturbation
        ctx->pert.select     = sel_body;
        ctx->pert.skinselect = skinid[0];
        ctx->pert.flexselect = flexid[0];
        ctx->pert.active = (button == GLFW_MOUSE_BUTTON_LEFT)
                               ? mjPERT_TRANSLATE   // force
                               : mjPERT_ROTATE;     // torque

        mjv_initPerturb(ctx->mj_model, ctx->mj_data,
                        &ctx->scn, &ctx->pert);

        RCLCPP_INFO(rclcpp::get_logger("impedance_control_node"),
            "Perturbation: body=%d type=%s (drag mouse to apply)",
            sel_body,
            (button == GLFW_MOUSE_BUTTON_LEFT) ? "force" : "torque");

        // Prevent triple-click from immediately re-triggering
        ctx->last_click_time = 0.0;
        return;  // skip camera-mouse handling for this press
      } else if (ctx->pert.active != 0) {
        // Double-click on empty space → clear perturbation
        ctx->pert.active = 0;
        ctx->pert.select = -1;
        ctx->last_click_time = 0.0;
        return;
      }
    }
  }

  // ==========================================================================
  // Normal camera mouse handling
  // ==========================================================================
  bool alt = (glfwGetKey(w, GLFW_KEY_LEFT_ALT)    == GLFW_PRESS ||
              glfwGetKey(w, GLFW_KEY_RIGHT_ALT)   == GLFW_PRESS);

  bool pressed = (act == GLFW_PRESS);

  if (button == GLFW_MOUSE_BUTTON_LEFT)
    (alt ? ctx->button_right : ctx->button_left) = pressed;
  else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    (alt ? ctx->button_left : ctx->button_right) = pressed;
  else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    ctx->button_middle = pressed;

  if (pressed)
    glfwGetCursorPos(w, &ctx->last_x, &ctx->last_y);
}

static void mouse_move(GLFWwindow* w, double xpos, double ypos)
{
  auto* ctx = static_cast<SimulatorContext*>(glfwGetWindowUserPointer(w));

  // ==========================================================================
  // Perturbation dragging — apply force/torque to the selected body
  // ==========================================================================
  if (ctx->pert.active != 0 &&
      (ctx->button_left || ctx->button_right || ctx->button_middle)) {
    double dx = xpos - ctx->last_x;
    double dy = ypos - ctx->last_y;
    ctx->last_x = xpos;
    ctx->last_y = ypos;

    int action = ctx->pert.active;  // mjPERT_TRANSLATE or mjPERT_ROTATE
    int height;
    glfwGetWindowSize(w, nullptr, &height);
    mjv_movePerturb(ctx->mj_model, ctx->mj_data, action,
                    dx / height, dy / height,
                    &ctx->scn, &ctx->pert);
    return;
  }

  // ==========================================================================
  // Camera movement
  // ==========================================================================
  if (!ctx->button_left && !ctx->button_right && !ctx->button_middle) return;

  double dx = xpos - ctx->last_x;
  double dy = ypos - ctx->last_y;
  ctx->last_x = xpos;
  ctx->last_y = ypos;

  int height;
  glfwGetWindowSize(w, nullptr, &height);
  bool shift = (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
                glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action;
  if      (ctx->button_right)          action = mjMOUSE_MOVE_H;
  else if (ctx->button_left && shift)  action = mjMOUSE_ROTATE_H;
  else if (ctx->button_left)           action = mjMOUSE_ROTATE_V;
  else                                  action = mjMOUSE_ZOOM;

  mjv_moveCamera(ctx->mj_model, action, dx / height, dy / height, &ctx->scn, &ctx->cam);
}

static void scroll(GLFWwindow* w, double /*xoff*/, double yoff)
{
  auto* ctx = static_cast<SimulatorContext*>(glfwGetWindowUserPointer(w));
  mjv_moveCamera(ctx->mj_model, mjMOUSE_ZOOM, 0, -0.05 * yoff, &ctx->scn, &ctx->cam);
}

// ============================================================================
// RobotControlNode — ROS2 node with impedance controller
// ============================================================================
class RobotControlNode : public rclcpp::Node
{
public:
  explicit RobotControlNode(SimulatorContext* ctx)
  : Node("impedance_control_node"), ctx_(ctx)
  {
    // --- ROS2 parameters ---
    declare_parameter("urdf_path", "");
    declare_parameter("mjcf_path", "");

    declare_parameter("Kp", 200.0);
    declare_parameter("Kd", 20.0);

    declare_parameter("left_target", std::vector<double>{0.0, 1.8, 1.12});
    declare_parameter("right_target", std::vector<double>{0.0, -1.8, 1.12});

    declare_parameter("left_ee_frame", "left_tool");
    declare_parameter("right_ee_frame", "right_tool");

    declare_parameter("transition_duration", 4.0);
    declare_parameter("steps_per_frame", 1);

    declare_parameter("null_Kp", 30.0);
    declare_parameter("null_Kd", 5.0);
    declare_parameter("robot_description", "");

    // --- Paths ---
    const char* home = std::getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");

    std::string urdf_path = get_parameter("urdf_path").as_string();
    if (urdf_path.empty())
      urdf_path = std::string(home)
        + "/Marvin_Description-Robot_Description/urdf/marvin_pro/marvin_robot.urdf";

    std::string mjcf_path = get_parameter("mjcf_path").as_string();
    if (mjcf_path.empty())
      mjcf_path = std::string(home)
        + "/my_work_pkg/src/robo_description/model/marvin_pro_mink_with_gripper.xml";

    // --- Load impedance parameters ---
    Kp_ = get_parameter("Kp").as_double();
    Kd_ = get_parameter("Kd").as_double();

    auto left_tgt  = get_parameter("left_target").as_double_array();
    auto right_tgt = get_parameter("right_target").as_double_array();
    left_target_pos_  = Eigen::Vector3d(left_tgt[0],  left_tgt[1],  left_tgt[2]);
    right_target_pos_ = Eigen::Vector3d(right_tgt[0], right_tgt[1], right_tgt[2]);

    steps_per_frame_ = get_parameter("steps_per_frame").as_int();
    if (steps_per_frame_ < 1) steps_per_frame_ = 1;

    transition_duration_ = get_parameter("transition_duration").as_double();
    if (transition_duration_ < 0.1) transition_duration_ = 0.1;

    Kp_null_ = get_parameter("null_Kp").as_double();
    Kd_null_ = get_parameter("null_Kd").as_double();

    std::string left_ee_name  = get_parameter("left_ee_frame").as_string();
    std::string right_ee_name = get_parameter("right_ee_frame").as_string();

    dt_ = 0.001;  // will be read from MJCF timestep

    // --- Load Pinocchio model ---
    RCLCPP_INFO(get_logger(), "Loading URDF: %s", urdf_path.c_str());
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

    // --- Load MuJoCo model ---
    RCLCPP_INFO(get_logger(), "Loading MJCF: %s", mjcf_path.c_str());
    char error[1000] = "";
    ctx_->mj_model = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
    if (!ctx_->mj_model) {
      RCLCPP_FATAL(get_logger(), "MuJoCo load failed: %s", error);
      throw std::runtime_error("MuJoCo load failed");
    }
    ctx_->mj_data = mj_makeData(ctx_->mj_model);

    // read the physics timestep from the model
    dt_ = ctx_->mj_model->opt.timestep;
    RCLCPP_INFO(get_logger(), "MuJoCo timestep = %.4f s", dt_);

    // --- Zero out MuJoCo passive joint dynamics ---
    // We want MuJoCo to only integrate our torques (no extra damping/friction).
    zero_passive_dynamics();

    // --- Build joint and actuator mappings ---
    build_joint_map();
    build_actuator_map();

    // --- Null-space rest posture: will be captured from initial config ---
    q_rest_ = Eigen::VectorXd::Zero(pin_model_.nq);  // placeholder, overwritten on first control_step

    // --- Find end-effector frame IDs in Pinocchio ---
    left_ee_frame_id_  = static_cast<int>(pin_model_.getFrameId(left_ee_name));
    right_ee_frame_id_ = static_cast<int>(pin_model_.getFrameId(right_ee_name));
    RCLCPP_INFO(get_logger(), "Left  EE frame: '%s' id=%d",
                left_ee_name.c_str(),  left_ee_frame_id_);
    RCLCPP_INFO(get_logger(), "Right EE frame: '%s' id=%d",
                right_ee_name.c_str(), right_ee_frame_id_);

    // --- ROS2 publishers ---
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::QoS(10).reliable());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    debug_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/impedance_debug", rclcpp::QoS(10));

    // --- Publish robot_description for RViz ---
    robot_desc_pub_ = create_publisher<std_msgs::msg::String>(
        "robot_description", rclcpp::QoS(1).transient_local());

    // --- Publish robot_description as parameter + topic for RViz ---
    {
      std::ifstream urdf_file(urdf_path);
      if (urdf_file) {
        std::stringstream buf;
        buf << urdf_file.rdbuf();
        std::string urdf_text = buf.str();

        // Resolve relative mesh paths → file:// absolute paths for RViz
        std::filesystem::path urdf_dir =
            std::filesystem::path(urdf_path).parent_path();
        std::regex mesh_re(R"__(filename="(\.\.[^"]*)")__");
        std::string result;
        auto it = std::sregex_iterator(urdf_text.begin(), urdf_text.end(), mesh_re);
        auto end = std::sregex_iterator();
        size_t last = 0;
        for (; it != end; ++it) {
          const auto& m = *it;
          result += urdf_text.substr(last, m.position() - last);
          auto abs_path = std::filesystem::canonical(urdf_dir / m[1].str());
          result += "filename=\"file://" + abs_path.string() + "\"";
          last = m.position() + m.length();
        }
        result += urdf_text.substr(last);
        urdf_text = std::move(result);

        // Set as parameter (RViz reads robot_description from parameter server)
        set_parameter(rclcpp::Parameter("robot_description", urdf_text));

        // Also publish as topic for other subscribers
        robot_desc_pub_ = create_publisher<std_msgs::msg::String>(
            "robot_description", rclcpp::QoS(1).transient_local());
        auto msg = std_msgs::msg::String();
        msg.data = urdf_text;
        robot_desc_pub_->publish(msg);
        RCLCPP_INFO(get_logger(), "URDF published on /robot_description (%zu bytes)",
                    msg.data.size());
      } else {
        RCLCPP_ERROR(get_logger(), "Cannot read URDF for robot_description: %s",
                     urdf_path.c_str());
      }
    }

    RCLCPP_INFO(get_logger(),
        "Ready: %zu active DOFs, %d Pinocchio joints, %ld MuJoCo actuators",
        mj_actuator_of_pin_v_.size(),
        static_cast<int>(pin_model_.njoints - 1),
        static_cast<long>(ctx_->mj_model->nu));
    RCLCPP_INFO(get_logger(),
        "Impedance: Kp=%.1f  Kd=%.1f  dt=%.4f  steps/frame=%d  "
        "null_Kp=%.1f  null_Kd=%.1f",
        Kp_, Kd_, dt_, steps_per_frame_, Kp_null_, Kd_null_);
    RCLCPP_INFO(get_logger(),
        "T-shape targets: L=[%.2f, %.2f, %.2f]  R=[%.2f, %.2f, %.2f]",
        left_target_pos_.x(),  left_target_pos_.y(),  left_target_pos_.z(),
        right_target_pos_.x(), right_target_pos_.y(), right_target_pos_.z());
    RCLCPP_INFO(get_logger(),
        "Transition duration: %.1f s (smoothstep from initial pose to T-shape)",
        transition_duration_);
  }

  // ==========================================================================
  // One control step: read state, compute torques, apply, integrate
  // ==========================================================================
  void control_step()
  {
    // Read current state from MuJoCo
    Eigen::VectorXd q  = read_q_from_mujoco();
    Eigen::VectorXd qd(pin_model_.nv);
    qd.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      qd[pin_v_idx_[i]] = ctx_->mj_data->qvel[mj_dof_adr_[i]];

    // Compute kinematics + Jacobians + gravity via Pinocchio
    pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
    pinocchio::computeJointJacobians(pin_model_, *pin_data_, q);
    pinocchio::computeGeneralizedGravity(pin_model_, *pin_data_, q);

    // Get gravity vector g(q)
    Eigen::VectorXd g_q = pin_data_->g;

    // --- Trajectory: capture initial EE positions and joint configuration on first call ---
    if (!trajectory_initialized_) {
      initial_left_ee_pos_  = pin_data_->oMf[left_ee_frame_id_].translation();
      initial_right_ee_pos_ = pin_data_->oMf[right_ee_frame_id_].translation();
      q_rest_ = q;  // Use current joint configuration as null-space rest posture
      trajectory_start_time_ = std::chrono::steady_clock::now();
      trajectory_initialized_ = true;

      RCLCPP_INFO(get_logger(),
          "Trajectory start: L_ee=[%.3f, %.3f, %.3f]  R_ee=[%.3f, %.3f, %.3f]",
          initial_left_ee_pos_.x(),  initial_left_ee_pos_.y(),  initial_left_ee_pos_.z(),
          initial_right_ee_pos_.x(), initial_right_ee_pos_.y(), initial_right_ee_pos_.z());
      RCLCPP_INFO(get_logger(),
          "Null-space rest posture captured (%d DOFs)", static_cast<int>(q.size()));
    }

    // Compute smoothstep interpolation factor
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - trajectory_start_time_).count();
    double t = std::max(0.0, std::min(1.0, elapsed / transition_duration_));
    double s = smoothstep(t);  // smooth ease-in/out

    // Interpolate target positions
    Eigen::Vector3d current_left_target  = initial_left_ee_pos_  + s * (left_target_pos_  - initial_left_ee_pos_);
    Eigen::Vector3d current_right_target = initial_right_ee_pos_ + s * (right_target_pos_ - initial_right_ee_pos_);

    // Compute impedance torques for both arms (with interpolated targets)
    const int nv = pin_model_.nv;
    Eigen::VectorXd tau_left  = compute_arm_torque(q, qd, left_ee_frame_id_,  current_left_target);
    Eigen::VectorXd tau_right = compute_arm_torque(q, qd, right_ee_frame_id_, current_right_target);
    Eigen::VectorXd tau = tau_left + tau_right + g_q;

    // Clamp torques to safe limits
    const double max_torque = 200.0;
    for (int i = 0; i < nv; ++i) {
      tau[i] = std::max(-max_torque, std::min(max_torque, tau[i]));
    }

    // Apply torques to MuJoCo ctrl array
    apply_torques_to_mujoco(tau);

    // Store for debug publishing
    last_tau_ = tau;
  }

  // ==========================================================================
  // Sync state to Pinocchio and publish ROS2 messages
  // ==========================================================================
  void sync_and_publish()
  {
    Eigen::VectorXd q = read_q_from_mujoco();
    pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
    publish_joint_states();
    publish_tf_frames();
    publish_debug();
  }

  // ==========================================================================
  // Accessors
  // ==========================================================================
  SimulatorContext* ctx() const { return ctx_; }
  int steps_per_frame() const { return steps_per_frame_; }

private:
  // ========================================================================
  // Zero out MuJoCo passive joint dynamics (damping, armature, frictionloss)
  // ========================================================================
  void zero_passive_dynamics()
  {
    mjModel* m = ctx_->mj_model;
    for (int i = 0; i < m->nv; ++i) {
      m->dof_damping[i]      = 0.0;
      m->dof_armature[i]     = 0.0;
      m->dof_frictionloss[i] = 0.0;
    }
    RCLCPP_INFO(get_logger(),
        "Zeroed passive dynamics for %ld DOFs (damping, armature, frictionloss)",
        static_cast<long>(m->nv));
  }

  // ========================================================================
  // Build Pinocchio ↔ MuJoCo joint position mapping
  // ========================================================================
  void build_joint_map()
  {
    pin_joint_names_.clear();
    pin_q_idx_.clear();
    pin_v_idx_.clear();
    mj_qpos_adr_.clear();
    mj_dof_adr_.clear();

    for (int j = 1; j < pin_model_.njoints; ++j) {
      int j_nq = pin_model_.joints[j].nq();
      int j_nv = pin_model_.joints[j].nv();
      if (j_nq < 1 || j_nv < 1) continue;

      int idx_q = pin_model_.joints[j].idx_q();
      int idx_v = pin_model_.joints[j].idx_v();
      const std::string& name = pin_model_.names[j];

      int mj_id = mj_name2id(ctx_->mj_model, mjOBJ_JOINT, name.c_str());
      if (mj_id < 0) {
        RCLCPP_WARN(get_logger(),
            "Joint '%s' not in MuJoCo model, skipping", name.c_str());
        continue;
      }
      int mj_qpos_adr = ctx_->mj_model->jnt_qposadr[mj_id];
      int mj_dof_adr  = ctx_->mj_model->jnt_dofadr[mj_id];
      if (mj_qpos_adr < 0 || mj_dof_adr < 0) continue;

      for (int k = 0; k < j_nq; ++k) {
        pin_joint_names_.push_back(name);
        pin_q_idx_.push_back(idx_q + k);
        pin_v_idx_.push_back(idx_v + k);
        mj_qpos_adr_.push_back(mj_qpos_adr + k);
        mj_dof_adr_.push_back(mj_dof_adr + k);
      }
    }
  }

  // ========================================================================
  // Build Pinocchio v-index → MuJoCo actuator index mapping
  // ========================================================================
  void build_actuator_map()
  {
    mj_actuator_of_pin_v_.resize(pin_model_.nv, -1);

    for (int j = 1; j < pin_model_.njoints; ++j) {
      int j_nv = pin_model_.joints[j].nv();
      if (j_nv < 1) continue;

      int idx_v = pin_model_.joints[j].idx_v();
      const std::string& name = pin_model_.names[j];

      int act_id = mj_name2id(ctx_->mj_model, mjOBJ_ACTUATOR, name.c_str());
      if (act_id < 0) {
        // Not every joint has an actuator (e.g. fixed joints, mimic joints)
        continue;
      }

      for (int k = 0; k < j_nv; ++k) {
        mj_actuator_of_pin_v_[idx_v + k] = act_id;
      }
    }
  }

  // ========================================================================
  // Smoothstep: s(0)=0, s'(0)=0, s(1)=1, s'(1)=0  (cubic ease-in/out)
  // ========================================================================
  static double smoothstep(double t) {
    // t is already clamped to [0, 1]
    return t * t * (3.0 - 2.0 * t);
  }

  // ========================================================================
  // Impedance + null-space posture control:
  //   τ = Jᵀ·F_task  +  (I − J⁺J)·[Kp_null·(q_rest−q) − Kd_null·q̇]
  // Task space:  F_task = Kp·(x_des−x) − Kd·ẋ
  // Null space:  pulls joint posture toward rest (elbows down, arms straight)
  // ========================================================================
  Eigen::VectorXd compute_arm_torque(
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& qd,
      int ee_frame_id,
      const Eigen::Vector3d& target_pos)
  {
    const int nv = pin_model_.nv;

    // Current end-effector pose
    const auto& oMf = pin_data_->oMf[ee_frame_id];
    Eigen::Vector3d current_pos = oMf.translation();

    // Position error
    Eigen::Vector3d e_pos = target_pos - current_pos;

    // End-effector Jacobian (linear part only: top 3 rows)
    Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(
        pin_model_, *pin_data_, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::MatrixXd J_lin = J6.topRows<3>();  // 3 × nv

    // End-effector velocity: ẋ = J_lin * qd
    Eigen::Vector3d ee_vel = J_lin * qd;

    // --- Task-space torque ---
    Eigen::Vector3d F = Kp_ * e_pos - Kd_ * ee_vel;
    Eigen::VectorXd tau_task = J_lin.transpose() * F;

    // --- Null-space posture torque ---
    // Right pseudo-inverse: J⁺ = Jᵀ·(J·Jᵀ)⁻¹   (3×3 solve, numerically cheap)
    Eigen::MatrixXd JJt = J_lin * J_lin.transpose();          // 3×3
    Eigen::MatrixXd J_pinv = J_lin.transpose() * JJt.inverse(); // nv×3

    // Null-space projector: N = I − J⁺·J  (projects into kinematic null space)
    Eigen::MatrixXd I_nv = Eigen::MatrixXd::Identity(nv, nv);
    Eigen::MatrixXd N = I_nv - J_pinv * J_lin;  // nv×nv

    // Posture error (only active joint DOFs — free-flyer base is fixed)
    Eigen::VectorXd q_err = q_rest_ - q;
    Eigen::VectorXd tau_null = N * (Kp_null_ * q_err - Kd_null_ * qd);

    return tau_task + tau_null;
  }

  // ========================================================================
  // Apply Pinocchio generalized torques to MuJoCo ctrl array
  // ========================================================================
  void apply_torques_to_mujoco(const Eigen::VectorXd& tau)
  {
    // Zero out ctrl first
    std::fill(ctx_->mj_data->ctrl,
              ctx_->mj_data->ctrl + ctx_->mj_model->nu, 0.0);

    // Map each Pinocchio DOF to the corresponding MuJoCo actuator
    for (int i = 0; i < pin_model_.nv; ++i) {
      int act_id = mj_actuator_of_pin_v_[i];
      if (act_id >= 0) {
        ctx_->mj_data->ctrl[act_id] = tau[i];
      }
    }
  }

  // ========================================================================
  // Read current joint positions from MuJoCo into a Pinocchio-sized vector
  // ========================================================================
  Eigen::VectorXd read_q_from_mujoco() const
  {
    Eigen::VectorXd q(pin_model_.nq);
    q.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      q[pin_q_idx_[i]] = ctx_->mj_data->qpos[mj_qpos_adr_[i]];
    return q;
  }

  // ========================================================================
  // ROS2 publishers
  // ========================================================================
  void publish_joint_states()
  {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = get_clock()->now();
    const size_t n = pin_joint_names_.size();
    msg.name.resize(n);
    msg.position.resize(n);
    msg.velocity.resize(n);

    for (size_t i = 0; i < n; ++i) {
      msg.name[i]     = pin_joint_names_[i];
      msg.position[i] = ctx_->mj_data->qpos[mj_qpos_adr_[i]];
      msg.velocity[i] = ctx_->mj_data->qvel[mj_dof_adr_[i]];
    }
    joint_state_pub_->publish(msg);
  }

  void publish_tf_frames()
  {
    const auto stamp = get_clock()->now();

    for (int f = 0; f < pin_model_.nframes; ++f) {
      const auto& frame = pin_model_.frames[f];
      const auto& oMf   = pin_data_->oMf[f];

      int parent = frame.previousFrame;

      // Skip self-referencing root frame (Pinocchio's "universe" frame, parent == self)
      if (parent == f) continue;

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.child_frame_id = frame.name;

      // Parent == 0 means Pinocchio root "universe" → use "world" (absolute pose)
      if (parent == 0) {
        tf.header.frame_id = "world";
        tf.transform.translation.x = oMf.translation()[0];
        tf.transform.translation.y = oMf.translation()[1];
        tf.transform.translation.z = oMf.translation()[2];
        Eigen::Quaterniond q_w(oMf.rotation());
        tf.transform.rotation.w = q_w.w();
        tf.transform.rotation.x = q_w.x();
        tf.transform.rotation.y = q_w.y();
        tf.transform.rotation.z = q_w.z();
      } else if (parent >= 0) {
        auto rel = pin_data_->oMf[parent].inverse() * oMf;
        tf.header.frame_id = pin_model_.frames[parent].name;
        tf.transform.translation.x = rel.translation().x();
        tf.transform.translation.y = rel.translation().y();
        tf.transform.translation.z = rel.translation().z();
        Eigen::Quaterniond q_rel(rel.rotation());
        tf.transform.rotation.w = q_rel.w();
        tf.transform.rotation.x = q_rel.x();
        tf.transform.rotation.y = q_rel.y();
        tf.transform.rotation.z = q_rel.z();
      } else {
        tf.header.frame_id = "world";
        tf.transform.translation.x = oMf.translation()[0];
        tf.transform.translation.y = oMf.translation()[1];
        tf.transform.translation.z = oMf.translation()[2];
        Eigen::Quaterniond q_w(oMf.rotation());
        tf.transform.rotation.w = q_w.w();
        tf.transform.rotation.x = q_w.x();
        tf.transform.rotation.y = q_w.y();
        tf.transform.rotation.z = q_w.z();
      }
      tf_broadcaster_->sendTransform(tf);
    }
  }

  void publish_debug()
  {
    // Throttle to ~10 Hz
    static auto last_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count() < 100)
      return;
    last_time = now;

    auto msg = std_msgs::msg::Float64MultiArray();
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "data";
    msg.layout.dim[0].size = 18;  // 6 values per arm (3 pos + 3 err) + 6 for torque norms
    msg.layout.dim[0].stride = 1;

    // Get current EE positions
    Eigen::Vector3d left_pos  = pin_data_->oMf[left_ee_frame_id_].translation();
    Eigen::Vector3d right_pos = pin_data_->oMf[right_ee_frame_id_].translation();

    Eigen::Vector3d left_err  = left_target_pos_  - left_pos;
    Eigen::Vector3d right_err = right_target_pos_ - right_pos;

    // Pack: [left_pos(3), left_err(3), right_pos(3), right_err(3)]
    msg.data = {
      left_pos.x(),  left_pos.y(),  left_pos.z(),
      left_err.x(),  left_err.y(),  left_err.z(),
      right_pos.x(), right_pos.y(), right_pos.z(),
      right_err.x(), right_err.y(), right_err.z(),
    };

    // Append torque RMS for left and right arms
    double left_tau_rms = 0.0, right_tau_rms = 0.0;
    int left_count = 0, right_count = 0;
    for (int i = 0; i < pin_model_.nv; ++i) {
      if (i < pin_model_.nv / 2) {
        left_tau_rms += last_tau_[i] * last_tau_[i];
        left_count++;
      } else {
        right_tau_rms += last_tau_[i] * last_tau_[i];
        right_count++;
      }
    }
    if (left_count > 0)  left_tau_rms  = std::sqrt(left_tau_rms  / left_count);
    if (right_count > 0) right_tau_rms = std::sqrt(right_tau_rms / right_count);
    msg.data.push_back(left_tau_rms);
    msg.data.push_back(right_tau_rms);

    debug_pub_->publish(msg);
  }

  // ========================================================================
  // Member variables
  // ========================================================================
  SimulatorContext* ctx_;

  // Pinocchio
  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;

  // Joint name ↔ index mappings
  std::vector<std::string> pin_joint_names_;
  std::vector<int>         pin_q_idx_;
  std::vector<int>         pin_v_idx_;
  std::vector<int>         mj_qpos_adr_;
  std::vector<int>         mj_dof_adr_;

  // Pinocchio velocity index → MuJoCo actuator index
  std::vector<int>         mj_actuator_of_pin_v_;

  // End-effector frame IDs
  int left_ee_frame_id_  = -1;
  int right_ee_frame_id_ = -1;

  // Impedance parameters
  double Kp_ = 200.0;
  double Kd_ = 20.0;
  double dt_ = 0.001;
  int    steps_per_frame_ = 1;
  double transition_duration_ = 4.0;

  // Null-space posture control
  double Kp_null_ = 30.0;
  double Kd_null_ = 5.0;
  Eigen::VectorXd q_rest_;  // desired joint posture (arms straight, elbows down)

  // Trajectory state
  bool trajectory_initialized_ = false;
  std::chrono::steady_clock::time_point trajectory_start_time_;
  Eigen::Vector3d initial_left_ee_pos_;
  Eigen::Vector3d initial_right_ee_pos_;

  // Target positions for T-shape
  Eigen::Vector3d left_target_pos_;
  Eigen::Vector3d right_target_pos_;

  // Last computed torques for debug
  Eigen::VectorXd last_tau_;

  // ROS2 publishers
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debug_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_desc_pub_;
};

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto ctx = std::make_unique<SimulatorContext>();
  GLFWwindow* window = nullptr;

  try {
    auto node = std::make_shared<RobotControlNode>(ctx.get());

    if (!glfwInit()) {
      RCLCPP_FATAL(node->get_logger(), "GLFW init failed");
      rclcpp::shutdown();
      return 1;
    }
    window = glfwCreateWindow(1400, 900,
        "Impedance Control — T-Shape (MuJoCo + Pinocchio + ROS2)",
        nullptr, nullptr);
    if (!window) {
      RCLCPP_FATAL(node->get_logger(), "GLFW window creation failed");
      glfwTerminate();
      rclcpp::shutdown();
      return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, ctx.get());

    glfwSetKeyCallback(window, keyboard);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);

    mjv_makeScene(ctx->mj_model, &ctx->scn, 5000);
    mjr_makeContext(ctx->mj_model, &ctx->con, mjFONTSCALE_150);

    using Clock = std::chrono::steady_clock;
    auto last_pub = Clock::now();
    constexpr auto pub_interval = std::chrono::milliseconds(16);

    RCLCPP_INFO(node->get_logger(),
        "Entering render loop. Press [Space] to pause, [Esc] to quit.");

    while (!glfwWindowShouldClose(window)) {
      // --- Apply perturbation force (if any) ---------------------------------
      bool perturbing = (ctx->pert.active != 0);
      if (perturbing) {
        // Clear previous xfrc_applied, then apply perturbation
        mju_zero(ctx->mj_data->xfrc_applied, 6 * ctx->mj_model->nbody);
        mjv_applyPerturbForce(ctx->mj_model, ctx->mj_data, &ctx->pert);
      }

      // When perturbing, physics runs even if "paused" so the user sees the
      // effect of their force/torque on the robot in real time.
      if (!ctx->paused || perturbing) {
        // Impedance control step(s)
        node->control_step();

        // Integrate MuJoCo physics (applies our torques from ctrl)
        for (int s = 0; s < node->steps_per_frame(); ++s) {
          mj_step(ctx->mj_model, ctx->mj_data);
        }
      }

      // Publish at fixed rate
      auto now = Clock::now();
      if (now - last_pub >= pub_interval) {
        try {
          node->sync_and_publish();
        } catch (const std::exception& e) {
          RCLCPP_ERROR_STREAM(node->get_logger(),
              "sync_and_publish failed: " << e.what());
        }
        last_pub = now;
      }

      // Render
      int w, h;
      glfwGetFramebufferSize(window, &w, &h);
      mjrRect viewport = {0, 0, w, h};

      mjv_updateScene(ctx->mj_model, ctx->mj_data, &ctx->opt, nullptr,
                      &ctx->cam, mjCAT_ALL, &ctx->scn);
      mjr_render(viewport, &ctx->scn, &ctx->con);
      glfwSwapBuffers(window);
      glfwPollEvents();
      rclcpp::spin_some(node);
    }

    RCLCPP_INFO(node->get_logger(), "Render loop exited normally.");
  } catch (const std::exception& e) {
    RCLCPP_FATAL_STREAM(rclcpp::get_logger("impedance_control_node"),
        "Fatal: " << e.what());
  } catch (...) {
    RCLCPP_FATAL(rclcpp::get_logger("impedance_control_node"),
        "Fatal: unknown exception");
  }

  mjr_freeContext(&ctx->con);
  mjv_freeScene(&ctx->scn);
  if (window) glfwDestroyWindow(window);
  glfwTerminate();
  rclcpp::shutdown();
  return 0;
}

