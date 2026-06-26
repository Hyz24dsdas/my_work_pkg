// Copyright 2024
//
// MuJoCo simulate UI + Pinocchio impedance controller.
// - Full MuJoCo 3.x UI (panels, sliders, double-click perturbation, keyframes, history)
// - Pinocchio: kinematics (FK, Jacobian, gravity)
// - Impedance control in task-space per end-effector
// - ROS2: joint_states, TF, debug info
//
// Architecture:
//   Main thread:  MuJoCo UI (Simulate::RenderLoop) + ROS2 publishing
//   Physics thread: impedance controller + mj_step

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

#include "../simulate_mujoco/glfw_adapter.h"
#include "../simulate_mujoco/simulate.h"
#include "../simulate_mujoco/array_safety.h"

namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// ============================================================================
// ImpedanceController — pure control logic (no ROS2 dependency)
// ============================================================================
class ImpedanceController
{
public:
  ImpedanceController(const std::string& urdf_path,
                      double Kp, double Kd,
                      double Kp_rot, double Kd_rot,
                      double Kd_joint,
                      double M_pos, double M_rot,
                      const Eigen::Vector3d& left_target,
                      const Eigen::Vector3d& right_target,
                      const std::string& left_ee_name,
                      const std::string& right_ee_name,
                      double transition_duration)
    : Kp_(Kp), Kd_(Kd)
    , Kp_rot_(Kp_rot), Kd_rot_(Kd_rot)
    , Kd_joint_(Kd_joint)
    , M_pos_(M_pos), M_rot_(M_rot)
    , left_target_pos_(left_target)
    , right_target_pos_(right_target)
    , transition_duration_(transition_duration)
  {
    // Orientation targets default to identity; will be set to initial EE
    // orientation on the first control step ("maintain current orientation").
    left_target_ori_  = Eigen::Matrix3d::Identity();
    right_target_ori_ = Eigen::Matrix3d::Identity();

    // Load Pinocchio model from URDF
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

    // Find end-effector frame IDs
    left_ee_frame_id_  = static_cast<int>(pin_model_.getFrameId(left_ee_name));
    right_ee_frame_id_ = static_cast<int>(pin_model_.getFrameId(right_ee_name));

    printf("Pinocchio model: %d joints, nq=%d, nv=%d\n",
           static_cast<int>(pin_model_.njoints - 1),
           static_cast<int>(pin_model_.nq),
           static_cast<int>(pin_model_.nv));
    printf("Left  EE frame: '%s' id=%d\n", left_ee_name.c_str(),  left_ee_frame_id_);
    printf("Right EE frame: '%s' id=%d\n", right_ee_name.c_str(), right_ee_frame_id_);
  }

  // Must be called AFTER mjModel is loaded
  void init_mujoco_mappings(mjModel* m)
  {
    mj_model_ = m;
    build_joint_map();
    build_actuator_map();
    // zero_passive_dynamics();  // keep MuJoCo's built-in damping/friction for stability

    dt_ = m->opt.timestep;
    printf("MuJoCo model: %ld actuators, timestep=%.4f\n", (long)m->nu, dt_);
    printf("Mapped %zu DOFs to MuJoCo joints, %ld actuators\n",
           pin_joint_names_.size(),
           static_cast<long>(m->nu));
  }

  // One control step: read state, compute torques, write to mjData.ctrl
  void control_step(mjData* d)
  {
    if (!mj_model_) return;

    // Read current state from MuJoCo
    Eigen::VectorXd q  = read_q_from_mujoco(d);
    Eigen::VectorXd qd(pin_model_.nv);
    qd.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      qd[pin_v_idx_[i]] = d->qvel[mj_dof_adr_[i]];

    // Read joint accelerations from previous MuJoCo step (for inertia feedback)
    Eigen::VectorXd qdd(pin_model_.nv);
    qdd.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      qdd[pin_v_idx_[i]] = d->qacc[mj_dof_adr_[i]];

    // Compute kinematics (pos+vel+accel) + Jacobians + gravity via Pinocchio
    pinocchio::forwardKinematics(pin_model_, *pin_data_, q, qd, qdd);
    pinocchio::updateFramePlacements(pin_model_, *pin_data_);  // explicitly refresh oMf
    pinocchio::computeJointJacobians(pin_model_, *pin_data_, q);
    pinocchio::computeGeneralizedGravity(pin_model_, *pin_data_, q);

    Eigen::VectorXd g_q = pin_data_->g;

    // Trajectory: capture initial EE positions and joint configuration on first call
    if (!trajectory_initialized_) {
      initial_left_ee_pos_  = pin_data_->oMf[left_ee_frame_id_].translation();
      initial_right_ee_pos_ = pin_data_->oMf[right_ee_frame_id_].translation();
      initial_left_ee_ori_  = pin_data_->oMf[left_ee_frame_id_].rotation();
      initial_right_ee_ori_ = pin_data_->oMf[right_ee_frame_id_].rotation();
      left_target_ori_  = initial_left_ee_ori_;
      right_target_ori_ = initial_right_ee_ori_;
      trajectory_start_time_ = std::chrono::steady_clock::now();
      trajectory_initialized_ = true;

      printf("Trajectory start: L_ee=[%.3f, %.3f, %.3f]  R_ee=[%.3f, %.3f, %.3f]\n",
             initial_left_ee_pos_.x(),  initial_left_ee_pos_.y(),  initial_left_ee_pos_.z(),
             initial_right_ee_pos_.x(), initial_right_ee_pos_.y(), initial_right_ee_pos_.z());
    }

    // Compute smoothstep interpolation
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - trajectory_start_time_).count();
    double t = std::max(0.0, std::min(1.0, elapsed / transition_duration_));
    double s = smoothstep(t);

    Eigen::Vector3d current_left_target  = initial_left_ee_pos_  + s * (left_target_pos_  - initial_left_ee_pos_);
    Eigen::Vector3d current_right_target = initial_right_ee_pos_ + s * (right_target_pos_ - initial_right_ee_pos_);

    // Compute impedance torques for both arms
    const int nv = pin_model_.nv;
    Eigen::VectorXd tau_left  = compute_arm_torque(qd, left_ee_frame_id_,  current_left_target,  left_target_ori_);
    Eigen::VectorXd tau_right = compute_arm_torque(qd, right_ee_frame_id_, current_right_target, right_target_ori_);
    // Torque ramp: gradually release control authority over the first second
    // to avoid a sudden torque spike on startup.
    double torque_ramp = smoothstep(std::max(0.0, std::min(1.0, elapsed / 1.0)));

    Eigen::VectorXd tau_task = tau_left + tau_right;
    Eigen::VectorXd tau_damp = -Kd_joint_ * qd;
    Eigen::VectorXd tau = g_q + torque_ramp * (tau_task + tau_damp);

    // Clamp torques
    const double max_torque = 200.0;
    for (int i = 0; i < nv; ++i)
      tau[i] = std::max(-max_torque, std::min(max_torque, tau[i]));

    // Apply torques to MuJoCo ctrl array
    apply_torques_to_mujoco(d, tau);

    // Store for debug
    last_tau_ = tau;
    last_q_ = q;
    last_qd_ = qd;
  }

  // Accessors for ROS2 publishing
  const Eigen::VectorXd& last_q()  const { return last_q_; }
  const Eigen::VectorXd& last_qd() const { return last_qd_; }
  const Eigen::VectorXd& last_tau() const { return last_tau_; }
  const std::vector<std::string>& joint_names() const { return pin_joint_names_; }
  const std::vector<int>& mj_qpos_adr() const { return mj_qpos_adr_; }
  const std::vector<int>& mj_dof_adr()  const { return mj_dof_adr_; }

  const pinocchio::Model& pin_model() const { return pin_model_; }
  const pinocchio::Data&  pin_data()  const { return *pin_data_; }
  int left_ee_frame_id()  const { return left_ee_frame_id_; }
  int right_ee_frame_id() const { return right_ee_frame_id_; }
  const Eigen::Vector3d& left_target()  const { return left_target_pos_; }
  const Eigen::Vector3d& right_target() const { return right_target_pos_; }
  const Eigen::Matrix3d& left_target_ori()  const { return left_target_ori_; }
  const Eigen::Matrix3d& right_target_ori() const { return right_target_ori_; }

private:
  static double smoothstep(double t) { return t * t * (3.0 - 2.0 * t); }

  void zero_passive_dynamics()
  {
    for (int i = 0; i < mj_model_->nv; ++i) {
      mj_model_->dof_damping[i]      = 0.0;
      mj_model_->dof_armature[i]     = 0.0;
      mj_model_->dof_frictionloss[i] = 0.0;
    }
    printf("Zeroed passive dynamics for %ld DOFs\n", (long)mj_model_->nv);
  }

  void build_joint_map()
  {
    for (int j = 1; j < pin_model_.njoints; ++j) {
      int j_nq = pin_model_.joints[j].nq();
      int j_nv = pin_model_.joints[j].nv();
      if (j_nq < 1 || j_nv < 1) continue;

      int idx_q = pin_model_.joints[j].idx_q();
      int idx_v = pin_model_.joints[j].idx_v();
      const std::string& name = pin_model_.names[j];

      int mj_id = mj_name2id(mj_model_, mjOBJ_JOINT, name.c_str());
      if (mj_id < 0) {
        printf("WARNING: Joint '%s' not in MuJoCo model, skipping\n", name.c_str());
        continue;
      }
      int mj_qpos_adr = mj_model_->jnt_qposadr[mj_id];
      int mj_dof_adr  = mj_model_->jnt_dofadr[mj_id];
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

  void build_actuator_map()
  {
    mj_actuator_of_pin_v_.resize(pin_model_.nv, -1);
    for (int j = 1; j < pin_model_.njoints; ++j) {
      int j_nv = pin_model_.joints[j].nv();
      if (j_nv < 1) continue;
      int idx_v = pin_model_.joints[j].idx_v();
      const std::string& name = pin_model_.names[j];
      int act_id = mj_name2id(mj_model_, mjOBJ_ACTUATOR, name.c_str());
      if (act_id < 0) {
        printf("WARNING: No actuator found for joint '%s'\n", name.c_str());
        continue;
      }
      printf("  actuator map: joint '%s' -> act_id %d\n", name.c_str(), act_id);
      for (int k = 0; k < j_nv; ++k)
        mj_actuator_of_pin_v_[idx_v + k] = act_id;
    }
  }

  Eigen::VectorXd read_q_from_mujoco(const mjData* d) const
  {
    Eigen::VectorXd q(pin_model_.nq);
    q.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      q[pin_q_idx_[i]] = d->qpos[mj_qpos_adr_[i]];
    return q;
  }

  void apply_torques_to_mujoco(mjData* d, const Eigen::VectorXd& tau)
  {
    std::fill(d->ctrl, d->ctrl + mj_model_->nu, 0.0);
    for (int i = 0; i < pin_model_.nv; ++i) {
      int act_id = mj_actuator_of_pin_v_[i];
      if (act_id >= 0) d->ctrl[act_id] = tau[i];
    }
  }

  Eigen::VectorXd compute_arm_torque(
      const Eigen::VectorXd& qd,
      int ee_frame_id,
      const Eigen::Vector3d& target_pos,
      const Eigen::Matrix3d& target_ori)
  {
    const auto& oMf = pin_data_->oMf[ee_frame_id];

    // --- Position error (3D) ---
    Eigen::Vector3d current_pos = oMf.translation();
    Eigen::Vector3d e_pos = target_pos - current_pos;

    // --- Orientation error (3D, logSO3: axis * angle) ---
    Eigen::Matrix3d current_ori = oMf.rotation();
    Eigen::Matrix3d R_err = target_ori * current_ori.transpose();
    Eigen::Vector3d e_ori = pinocchio::log3(R_err);

    // --- Jacobian and task-space velocity ---
    Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(
        pin_model_, *pin_data_, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::Matrix<double, 6, 1> x_dot = J6 * qd;

    // --- Task-space acceleration (inertia feedback) ---
    // Requires forwardKinematics(q, qd, qdd) to have been called
    // getFrameClassicalAcceleration returns Motion (spatial accel): [angular, linear]
    auto accel = pinocchio::getFrameClassicalAcceleration(
        pin_model_, *pin_data_, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::Vector3d x_ddot_lin = accel.linear();   // linear acceleration
    Eigen::Vector3d x_ddot_ang = accel.angular();  // angular acceleration

    // --- Full impedance wrench ---
    // F = M·(ẍ_des − ẍ) + D·(ẋ_des − ẋ) + K·(x_des − x)
    // Static targets: ẍ_des = 0, ẋ_des = 0
    Eigen::Matrix<double, 6, 1> W;
    W.head<3>() = Kp_     * e_pos - Kd_     * x_dot.head<3>() - M_pos_ * x_ddot_lin;
    W.tail<3>() = Kp_rot_ * e_ori - Kd_rot_ * x_dot.tail<3>() - M_rot_ * x_ddot_ang;

    return J6.transpose() * W;
  }

  // Pinocchio
  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;
  mjModel* mj_model_ = nullptr;

  // Joint mappings
  std::vector<std::string> pin_joint_names_;
  std::vector<int> pin_q_idx_, pin_v_idx_;
  std::vector<int> mj_qpos_adr_, mj_dof_adr_;
  std::vector<int> mj_actuator_of_pin_v_;

  // EE frame IDs
  int left_ee_frame_id_ = -1, right_ee_frame_id_ = -1;

  // Impedance params
  double Kp_, Kd_, Kp_rot_, Kd_rot_, dt_ = 0.001;
  double Kd_joint_;
  double M_pos_, M_rot_;  // desired inertia (mass shaping)

  // Trajectory
  bool trajectory_initialized_ = false;
  std::chrono::steady_clock::time_point trajectory_start_time_;
  Eigen::Vector3d initial_left_ee_pos_, initial_right_ee_pos_;
  Eigen::Matrix3d initial_left_ee_ori_, initial_right_ee_ori_;
  Eigen::Vector3d left_target_pos_, right_target_pos_;
  Eigen::Matrix3d left_target_ori_, right_target_ori_;
  double transition_duration_;

  // Last computed state (for ROS2 publishing)
  Eigen::VectorXd last_q_, last_qd_, last_tau_;
};

// ============================================================================
// Global state (shared between physics thread and main thread)
// ============================================================================
static mjModel* g_m = nullptr;
static mjData*  g_d = nullptr;
static std::unique_ptr<ImpedanceController> g_ctrl;

// ROS2 node (created in main thread, read from anywhere)
static std::shared_ptr<rclcpp::Node> g_ros_node;
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr g_joint_pub;
static std::unique_ptr<tf2_ros::TransformBroadcaster> g_tf_broadcaster;
static rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr g_debug_pub;

// ============================================================================
// Physics loop (runs in background thread, replaces simulate's PhysicsLoop)
// ============================================================================

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

const double syncMisalign = 0.1;
const double simRefreshFraction = 0.7;

static const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0)
        return mju_warningText(w, d->warning[w].lastinfo);
    }
  }
  return nullptr;
}

static void PhysicsLoop(mj::Simulate& sim)
{
  auto syncCPU = Clock::now();
  mjtNum syncSim = 0;
  int last_run = -1;

  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      // In our custom version, we don't support dynamic model reloading
      // (Pinocchio model would need to be rebuilt)
      sim.droploadrequest.store(false);
      sim.LoadMessageClear();
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      // Same — no dynamic reload
      sim.LoadMessageClear();
    }

    // Sleep/yield to let main thread run
    if (sim.run && sim.busywait)
      std::this_thread::yield();
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

    {
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      if (g_m) {
        if (sim.run != last_run) {
          if (last_run != -1) {
            std::memset(g_d->timer, 0, sizeof(g_d->timer));
            std::memset(sim.timer_prev_, 0, sizeof(sim.timer_prev_));
          }
          last_run = sim.run;
        }

        if (sim.run) {
          bool stepped = false;
          const auto startCPU = Clock::now();
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = g_d->time - syncSim;
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];
          bool misaligned =
              std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > syncMisalign;

          if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
              syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim.speed_changed) {
            syncCPU = startCPU;
            syncSim = g_d->time;
            sim.speed_changed = false;
            sim.InjectNoise(sim.key);

            // === IMPEDANCE CONTROL STEP ===
            g_ctrl->control_step(g_d);
            mj_step(g_m, g_d);

            const char* msg = Diverged(g_m->opt.disableflags, g_d);
            if (msg) { sim.run = 0; mju::strcpy_arr(sim.load_error, msg); }
            else stepped = true;
          } else {
            bool measured = false;
            mjtNum prevSim = g_d->time;
            double refreshTime = simRefreshFraction / sim.refresh_rate;

            while (Seconds((g_d->time - syncSim)*slowdown) < Clock::now() - syncCPU &&
                   Clock::now() - startCPU < Seconds(refreshTime)) {
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              sim.InjectNoise(sim.key);

              // === IMPEDANCE CONTROL STEP ===
              g_ctrl->control_step(g_d);
              mj_step(g_m, g_d);

              const char* msg = Diverged(g_m->opt.disableflags, g_d);
              if (msg) { sim.run = 0; mju::strcpy_arr(sim.load_error, msg); }
              else stepped = true;

              if (g_d->time < prevSim) break;
            }
          }

          if (stepped) sim.AddToHistory();
        } else {
          // Paused: forward kinematics only (for rendering + joint sliders)
          mj_forward(g_m, g_d);
          if (sim.pause_update)
            mju_copy(g_d->qacc_warmstart, g_d->qacc, g_m->nv);
          sim.speed_changed = true;
        }
      }
    }
  }
}

// ============================================================================
// Model loading
// ============================================================================

static const int kErrorLength = 1024;

static mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  if (!filename[0]) return nullptr;

  char loadError[kErrorLength] = "";
  mjModel* mnew = nullptr;
  auto load_start = Clock::now();

  std::string filename_str(filename);
  size_t dot_pos = filename_str.rfind('.');
  std::string extension = (dot_pos != std::string::npos) ? filename_str.substr(dot_pos) : "";

  if (extension == ".mjb")
    mnew = mj_loadModel(filename, nullptr);
  else if (extension == ".xml")
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
  else {
    mjSpec* spec = mj_parse(filename, nullptr, nullptr, loadError, kErrorLength);
    if (spec) { mnew = mj_compile(spec, nullptr); mj_deleteSpec(spec); }
  }

  if (loadError[0]) {
    int len = mju::strlen_arr(loadError);
    if (loadError[len-1] == '\n') loadError[len-1] = '\0';
  }

  auto load_interval = Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  if (loadError[0]) {
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  } else if (load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);
  return mnew;
}

// ============================================================================
// ROS2 publishing (called from main thread)
// ============================================================================

static void publish_ros2(mj::Simulate& /*sim*/)
{
  if (!g_ros_node || !g_ctrl || !g_m || !g_d) return;

  // Throttle to ~60 Hz
  static auto last_pub = Clock::now();
  auto now = Clock::now();
  if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_pub).count() < 16)
    return;
  last_pub = now;

  const auto& joint_names = g_ctrl->joint_names();
  const auto& qpos_adr    = g_ctrl->mj_qpos_adr();
  const auto& dof_adr     = g_ctrl->mj_dof_adr();
  size_t n = joint_names.size();

  // Joint states
  {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = g_ros_node->get_clock()->now();
    msg.name.resize(n);
    msg.position.resize(n);
    msg.velocity.resize(n);
    for (size_t i = 0; i < n; ++i) {
      msg.name[i]     = joint_names[i];
      msg.position[i] = g_d->qpos[qpos_adr[i]];
      msg.velocity[i] = g_d->qvel[dof_adr[i]];
    }
    g_joint_pub->publish(msg);
  }

  // TF frames
  {
    // We need FK to be up-to-date. The physics thread runs FK via Pinocchio
    // each control step, but for TF we use Pinocchio's last computation.
    // For safety, re-run FK on the current MuJoCo state.
    Eigen::VectorXd q(g_ctrl->pin_model().nq);
    q.setZero();
    for (size_t i = 0; i < n; ++i)
      q[g_ctrl->pin_model().getJointId(joint_names[i]) - 1] = g_d->qpos[qpos_adr[i]];  // approximate — use pin q idx

    // Actually, just use the last computed state from control_step
    // This is close enough for visualization
    const auto& pin_model = g_ctrl->pin_model();
    const auto& pin_data  = g_ctrl->pin_data();
    const auto stamp = g_ros_node->get_clock()->now();

    // Publish only major frames to avoid flooding TF
    for (int f : {g_ctrl->left_ee_frame_id(), g_ctrl->right_ee_frame_id()}) {
      if (f < 0) continue;
      const auto& oMf = pin_data.oMf[f];
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = "world";
      tf.child_frame_id = pin_model.frames[f].name;
      tf.transform.translation.x = oMf.translation()[0];
      tf.transform.translation.y = oMf.translation()[1];
      tf.transform.translation.z = oMf.translation()[2];
      Eigen::Quaterniond qq(oMf.rotation());
      tf.transform.rotation.w = qq.w();
      tf.transform.rotation.x = qq.x();
      tf.transform.rotation.y = qq.y();
      tf.transform.rotation.z = qq.z();
      g_tf_broadcaster->sendTransform(tf);
    }
  }

  // Debug info (throttle to ~10 Hz within the 60 Hz throttle)
  {
    static auto last_debug = Clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_debug).count() >= 100) {
      last_debug = now;
      auto msg = std_msgs::msg::Float64MultiArray();
      const auto& pin_data = g_ctrl->pin_data();

      Eigen::Vector3d left_pos  = pin_data.oMf[g_ctrl->left_ee_frame_id()].translation();
      Eigen::Vector3d right_pos = pin_data.oMf[g_ctrl->right_ee_frame_id()].translation();
      Eigen::Vector3d left_err  = g_ctrl->left_target()  - left_pos;
      Eigen::Vector3d right_err = g_ctrl->right_target() - right_pos;

      // Orientation error magnitude (rad) — 6D control resists rotation
      double left_ori_err = 0.0, right_ori_err = 0.0;
      if (g_ctrl->left_target_ori().determinant() > 0.5) {  // valid rotation matrix
        Eigen::Matrix3d L_R_err = g_ctrl->left_target_ori()
                                * pin_data.oMf[g_ctrl->left_ee_frame_id()].rotation().transpose();
        left_ori_err = pinocchio::log3(L_R_err).norm();
      }
      if (g_ctrl->right_target_ori().determinant() > 0.5) {
        Eigen::Matrix3d R_R_err = g_ctrl->right_target_ori()
                                * pin_data.oMf[g_ctrl->right_ee_frame_id()].rotation().transpose();
        right_ori_err = pinocchio::log3(R_R_err).norm();
      }

      msg.data = {
        left_pos.x(), left_pos.y(), left_pos.z(),
        left_err.x(), left_err.y(), left_err.z(),
        right_pos.x(), right_pos.y(), right_pos.z(),
        right_err.x(), right_err.y(), right_err.z(),
        left_ori_err, right_ori_err,
      };

      const auto& tau = g_ctrl->last_tau();
      int nv = tau.size();
      double left_rms = 0, right_rms = 0;
      for (int i = 0; i < nv/2; ++i) left_rms  += tau[i] * tau[i];
      for (int i = nv/2; i < nv; ++i) right_rms += tau[i] * tau[i];
      left_rms = std::sqrt(left_rms / (nv/2));
      right_rms = std::sqrt(right_rms / (nv/2));
      msg.data.push_back(left_rms);
      msg.data.push_back(right_rms);

      g_debug_pub->publish(msg);
    }
  }
}

// ============================================================================
// Physics thread entry point
// ============================================================================

static void PhysicsThread(mj::Simulate* sim, const char* filename)
{
  if (filename) {
    sim->LoadMessage(filename);
    g_m = LoadModel(filename, *sim);
    if (g_m) {
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
      g_d = mj_makeData(g_m);

      int home_id = mj_name2id(g_m, mjOBJ_KEY, "home");
      if (home_id >= 0) {
        mj_resetDataKeyframe(g_m, g_d, home_id);
        printf("[INIT] Loaded home keyframe\n");
      } else {
        mj_resetData(g_m, g_d);
        printf("[INIT] home keyframe not found, reset data\n");
      }
    }
    if (g_d) {
      // Initialize the impedance controller's MuJoCo mappings
      g_ctrl->init_mujoco_mappings(g_m);

      sim->Load(g_m, g_d, filename);

      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
      mj_forward(g_m, g_d);
    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  mj_deleteData(g_d);
  mj_deleteModel(g_m);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv)
{
  // --- ROS2 init ---
  rclcpp::init(argc, argv);
  g_ros_node = std::make_shared<rclcpp::Node>("impedance_simulate_node");

  // --- Parameters (ROS2 declare_parameter, overridable from launch file) ---
  const char* home = std::getenv("HOME");
  if (!home) { fprintf(stderr, "HOME not set\n"); return 1; }

  // Paths
  g_ros_node->declare_parameter("urdf_path",
      std::string(home) + "/Marvin_Description-Robot_Description/urdf/marvin_pro/marvin_robot.urdf");
  g_ros_node->declare_parameter("mjcf_path",
      std::string(home) + "/my_work_pkg/src/robo_description/model/marvin_pro_mink_with_gripper.xml");

  // Impedance gains – translational
  g_ros_node->declare_parameter("Kp", 80.0);
  g_ros_node->declare_parameter("Kd", 20.0);

  // Impedance gains – rotational
  g_ros_node->declare_parameter("Kp_rot", 220.0);
  g_ros_node->declare_parameter("Kd_rot", 35.0);

  // Joint-space damping
  g_ros_node->declare_parameter("Kd_joint", 8.0);

  // Desired inertia (mass shaping) — set to 0 to disable acceleration feedback,
  // which can amplify qacc noise at startup and cause jerky motion.
  g_ros_node->declare_parameter("M_pos", 0.0);
  g_ros_node->declare_parameter("M_rot", 0.0);

  // Trajectory
  g_ros_node->declare_parameter("transition_duration", 4.0);

  // Target positions
  g_ros_node->declare_parameter("left_target",
      std::vector<double>{0.0, 1.8, 1.12});
  g_ros_node->declare_parameter("right_target",
      std::vector<double>{0.0, -1.8, 1.12});

  // Read back all parameters
  std::string urdf_path = g_ros_node->get_parameter("urdf_path").as_string();
  std::string mjcf_path = g_ros_node->get_parameter("mjcf_path").as_string();

  double Kp   = g_ros_node->get_parameter("Kp").as_double();
  double Kd   = g_ros_node->get_parameter("Kd").as_double();
  double Kp_rot = g_ros_node->get_parameter("Kp_rot").as_double();
  double Kd_rot = g_ros_node->get_parameter("Kd_rot").as_double();
  double Kd_joint = g_ros_node->get_parameter("Kd_joint").as_double();
  double M_pos = g_ros_node->get_parameter("M_pos").as_double();
  double M_rot = g_ros_node->get_parameter("M_rot").as_double();
  double transition_duration = g_ros_node->get_parameter("transition_duration").as_double();

  auto left_tgt  = g_ros_node->get_parameter("left_target").as_double_array();
  auto right_tgt = g_ros_node->get_parameter("right_target").as_double_array();
  Eigen::Vector3d left_target(left_tgt[0],  left_tgt[1],  left_tgt[2]);
  Eigen::Vector3d right_target(right_tgt[0], right_tgt[1], right_tgt[2]);

  // Allow command-line overrides for paths (argc>1: argv[1]=mjcf, argv[2]=urdf)
  if (argc > 1) mjcf_path = argv[1];
  if (argc > 2) urdf_path = argv[2];

  printf("MJCF:  %s\n", mjcf_path.c_str());
  printf("URDF:  %s\n", urdf_path.c_str());
  printf("Impedance: Kp=%.1f Kd=%.1f Kp_rot=%.1f Kd_rot=%.1f Kd_joint=%.1f M_pos=%.1f M_rot=%.1f\n",
         Kp, Kd, Kp_rot, Kd_rot, Kd_joint, M_pos, M_rot);
  printf("Targets: L=[%.2f,%.2f,%.2f] R=[%.2f,%.2f,%.2f]\n",
         left_target.x(), left_target.y(), left_target.z(),
         right_target.x(), right_target.y(), right_target.z());

  // --- Create impedance controller ---
  g_ctrl = std::make_unique<ImpedanceController>(
      urdf_path, Kp, Kd, Kp_rot, Kd_rot, Kd_joint, M_pos, M_rot,
      left_target, right_target,
      "left_tool", "right_tool",
      transition_duration);

  // --- ROS2 publishers ---
  g_joint_pub = g_ros_node->create_publisher<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::QoS(10).reliable());
  g_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*g_ros_node);
  g_debug_pub = g_ros_node->create_publisher<std_msgs::msg::Float64MultiArray>(
      "~/impedance_debug", rclcpp::QoS(10));

  // Publish robot_description for RViz
  {
    std::ifstream urdf_file(urdf_path);
    if (urdf_file) {
      std::stringstream buf;
      buf << urdf_file.rdbuf();
      std::string urdf_text = buf.str();

      // Resolve relative mesh paths → file://
      std::filesystem::path urdf_dir = std::filesystem::path(urdf_path).parent_path();
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

      auto pub = g_ros_node->create_publisher<std_msgs::msg::String>(
          "robot_description", rclcpp::QoS(1).transient_local());
      auto msg = std_msgs::msg::String();
      msg.data = result;
      pub->publish(msg);
      printf("URDF published on /robot_description (%zu bytes)\n", msg.data.size());
    }
  }

  // --- MuJoCo UI setup ---
  printf("MuJoCo version %s\n", mj_versionString());

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  auto sim = std::make_unique<mj::Simulate>(
      std::make_unique<mj::GlfwAdapter>(),
      &cam, &opt, &pert, /* is_passive = */ false);

  // --- Start physics thread ---
  const char* filename = mjcf_path.c_str();
  std::thread physics_thread(&PhysicsThread, sim.get(), filename);

  // --- ROS2 publishing: run from a timer in a background thread ---
  // We let sim->RenderLoop() own the main thread (it handles all UI init +
  // event loop). ROS2 publishing happens on a separate thread via a wall timer.
  auto ros_timer = g_ros_node->create_wall_timer(
      std::chrono::milliseconds(16),
      [&sim]() { publish_ros2(*sim); });

  std::thread ros_spin_thread([]() {
    rclcpp::spin(g_ros_node);
  });

  printf("Entering MuJoCo UI render loop...\n");

  // This blocks until the window is closed or exit is requested.
  // RenderLoop() does all the GLFW/MuJoCo UI initialization internally.
  sim->RenderLoop();

  // Shutdown
  sim->exitrequest.store(1);
  rclcpp::shutdown();
  physics_thread.join();
  if (ros_spin_thread.joinable()) ros_spin_thread.join();

  return 0;

}
