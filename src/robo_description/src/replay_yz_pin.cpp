// Pinocchio逆动力学补偿+PD关节姿态+yz平面8字任务空间控制
// MuJoCo负责仿真和可视化，Pinocchio从同一个MJCF模型计算逆动力学

//Copyright 2024
//
// L-shape dual-arm robot with figure-8 left-gripper trajectory.
// - Full MuJoCo 3.x UI (panels, sliders, double-click perturbation, keyframes, history)
// - Pinocchio: FULL dynamics — M(q), C(q,q̇), g(q), FK, Jacobian, J̇
// - Operational-space impedance control with dynamic decoupling
// - ROS2: joint_states, TF, EE pose topics for PlotJuggler
//
// Architecture:
//   Main thread:  MuJoCo UI (Simulate::RenderLoop) + ROS2 publishing
//   Physics thread: Pinocchio dynamics + impedance controller + mj_step
//
// Control law (per arm):
//   τ = M(q)·J̄·(ẍ_des − J̇·q̇) + C(q,q̇)·q̇ + g(q)  +  Nᵀ·τ_posture
//   where J̄ = M⁻¹·Jᵀ·(J·M⁻¹·Jᵀ)⁻¹  (dynamically consistent pseudo-inverse)
//         ẍ_des = target_acc + M_des⁻¹·(Kp·Δx + Kd·Δẋ)  (with feed-forward)
//         Nᵀ = I − Jᵀ·J̄ᵀ  (null-space projector)
//
// Target: arms form L-shape (upper arm horizontal, forearm down).
//         Left gripper traces a figure-8 (∞) in a plane.
//         Right arm holds static L-shape.

/*
ros2 launch robo_description replay_yz_pin.launch.py \
  Kp_x:=2500.0 \
  Kp_y:=900.0 \
  Kp_z:=1200.0 \
  Kd_x:=350.0 \
  Kd_y:=180.0 \
  Kd_z:=240.0 \
  Kq_posture:=300.0 \
  Dq_posture:=45.0 \
  figure8_frequency:=0.25 \
  figure8_amplitude_x:=0.08 \
  figure8_amplitude_z:=0.06
*/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/mjcf.hpp>

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
  ImpedanceController(const std::string& mjcf_path,
                      const Eigen::Vector3d& Kp_task,
                      const Eigen::Vector3d& Kd_task,
                      double Kp_null, double Kd_null,
                      double Kq_posture, double Dq_posture,
                      double replay_Kp, double replay_Kd,
                      double ilc_Lp, double ilc_Ld,
                      double ilc_learning_rate,
                      double ilc_max_delta_qdd,
                      double ilc_max_qdd_ff,
                      double max_desired_qdd,
                      double m_des,
                      const std::string& left_ee_name,
                      const std::string& right_ee_name,
                      double fig8_freq, double fig8_amp_x, double fig8_amp_z,
                      const std::string& fig8_plane)
    : Kp_task_(Kp_task), Kd_task_(Kd_task)
    , Kp_null_(Kp_null), Kd_null_(Kd_null)
    , Kq_posture_(Kq_posture), Dq_posture_(Dq_posture)
    , replay_Kp_(replay_Kp), replay_Kd_(replay_Kd)
    , ilc_Lp_(ilc_Lp), ilc_Ld_(ilc_Ld)
    , ilc_learning_rate_(ilc_learning_rate)
    , ilc_max_delta_qdd_(ilc_max_delta_qdd)
    , ilc_max_qdd_ff_(ilc_max_qdd_ff)
    , max_desired_qdd_(max_desired_qdd)
    , m_des_(m_des)
    , fig8_freq_(fig8_freq)
    , fig8_amp_x_(fig8_amp_x)
    , fig8_amp_z_(fig8_amp_z)
    , fig8_plane_(fig8_plane)
  {
    // Load Pinocchio model from the same MJCF used by MuJoCo.
    pinocchio::mjcf::buildModel(mjcf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    pin_data_bias_ = std::make_unique<pinocchio::Data>(pin_model_);
    pin_data_desired_ = std::make_unique<pinocchio::Data>(pin_model_);

    left_ee_frame_id_  = static_cast<int>(pin_model_.getFrameId(left_ee_name));
    right_ee_frame_id_ = static_cast<int>(pin_model_.getFrameId(right_ee_name));
    left_upper_frame_id_ = static_cast<int>(pin_model_.getFrameId("Arm_L3_Link"));
    left_forearm_frame_id_ = static_cast<int>(pin_model_.getFrameId("Arm_L5_Link"));

    printf("Pinocchio model: %d joints, nq=%d, nv=%d\n",
           static_cast<int>(pin_model_.njoints - 1),
           static_cast<int>(pin_model_.nq),
           static_cast<int>(pin_model_.nv));
    printf("Left  EE frame: '%s' id=%d\n", left_ee_name.c_str(),  left_ee_frame_id_);
    printf("Right EE frame: '%s' id=%d\n", right_ee_name.c_str(), right_ee_frame_id_);
    printf("Left  upper frame: 'Arm_L3_Link' id=%d\n", left_upper_frame_id_);
    printf("Left  forearm frame: 'Arm_L5_Link' id=%d\n", left_forearm_frame_id_);
    printf("Desired mass (M_des): %.1f kg\n", m_des_);
    printf("Replay joint PD: Kp=%.1f Kd=%.1f\n", replay_Kp_, replay_Kd_);
    printf("ILC: Lp=%.2f Ld=%.2f learning_rate=%.2f max_delta_qdd=%.2f max_qdd_ff=%.2f\n",
           ilc_Lp_, ilc_Ld_, ilc_learning_rate_, ilc_max_delta_qdd_, ilc_max_qdd_ff_);
  }

  // Must be called AFTER mjModel is loaded
  void init_mujoco_mappings(mjModel* m)
  {
    mj_model_ = m;
    build_joint_map();
    build_actuator_map();
    find_elbow_indices();

    dt_ = m->opt.timestep;
    printf("MuJoCo model: %ld actuators, timestep=%.4f\n", (long)m->nu, dt_);
    printf("Mapped %zu DOFs to MuJoCo joints, %ld actuators\n",
           pin_joint_names_.size(), (long)m->nu);
  }

  // One control step: read state, compute full Pinocchio dynamics,
  // impedance torques, write to mjData.ctrl
  void control_step(mjData* d)
  {
    if (!mj_model_) return;

    // Read current state from MuJoCo
    Eigen::VectorXd q  = read_q_from_mujoco(d);
    Eigen::VectorXd qd(pin_model_.nv);
    qd.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      qd[pin_v_idx_[i]] = d->qvel[mj_dof_adr_[i]];

    // Pinocchio: full dynamics computation
    pinocchio::forwardKinematics(pin_model_, *pin_data_, q, qd);
    pinocchio::updateFramePlacements(pin_model_, *pin_data_);
    pinocchio::computeJointJacobians(pin_model_, *pin_data_, q);

    // dynamic 
    pinocchio::crba(pin_model_, *pin_data_, q);                       // M(q)
    pinocchio::nonLinearEffects(pin_model_, *pin_data_, q, qd);       // nle = C·q̇ + g

    // Initialize L-shape configuration on first call
    if (!trajectory_initialized_) {
      printf("[INIT-Q] current left  q = %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
             q[0], q[1], q[2], q[3], q[4], q[5], q[6]);

      printf("[INIT-Q] current right q = %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
             q[9], q[10], q[11], q[12], q[13], q[14], q[15]);

      init_l_shape_configuration(q);

      // Verify: FK on current q vs q_rest
      {
        pinocchio::Data check_data(pin_model_);
        pinocchio::forwardKinematics(pin_model_, check_data, q);
        pinocchio::updateFramePlacements(pin_model_, check_data);

        Eigen::Vector3d actual_init =
            check_data.oMf[left_ee_frame_id_].translation();

        printf("[CHECK] FK(q_current) left z = %.6f\n", actual_init.z());
        printf("[CHECK] FK(q_rest)    left z = %.6f\n", l_shape_left_ee_.z());
        printf("[CHECK] dz = %.6f\n", l_shape_left_ee_.z() - actual_init.z());
      }

      start_sim_time_ = d->time;
      desired_q_ = q_rest_;
      desired_qd_ = Eigen::VectorXd::Zero(pin_model_.nv);
      desired_qdd_ = Eigen::VectorXd::Zero(pin_model_.nv);
      trajectory_initialized_ = true;
    }

    // =====================================================================
    // Figure-8 trajectory for left EE (compute targets even in test mode)
    // =====================================================================
    double t = d->time - start_sim_time_;

    TrajectoryPoint fig8 = compute_figure8_trajectory(t);

    // 左臂：8字轨迹
    Eigen::Vector3d left_target_pos = l_shape_left_ee_ + fig8.pos;
    Eigen::Vector3d left_target_vel = fig8.vel;
    Eigen::Vector3d left_target_acc = fig8.acc;

    // 右臂：静止目标
    Eigen::Vector3d right_target_pos = l_shape_right_ee_;
    Eigen::Vector3d right_target_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_target_acc = Eigen::Vector3d::Zero();

    if (!recording_complete_) {
      current_left_target_  = left_target_pos;
      current_right_target_ = right_target_pos;
      update_desired_joint_state(left_target_pos, left_target_vel,
                                 right_target_pos, right_target_vel);
    }

    const double cycle_duration = 1.0 / std::max(fig8_freq_, 1e-6);
    if (!recording_complete_ && t >= cycle_duration && recorded_trajectory_.size() >= 2) {
      finish_recording(cycle_duration);
    }

    if (recording_complete_) {
      control_replay_step(d, q, qd, t);
      return;
    }

    // ===== Stable controller: MuJoCo bias + J^T F + joint posture PD =====
    {
      mj_forward(mj_model_, d);

      Eigen::VectorXd tau(pin_model_.nv);
      tau.setZero();

      const int nv = pin_model_.nv;

      Eigen::VectorXd qdd_cmd = Eigen::VectorXd::Zero(nv);
      Eigen::MatrixXd N = Eigen::MatrixXd::Identity(nv, nv);

      auto add_acc_task = [&](const Eigen::MatrixXd& J_task,
                              const Eigen::VectorXd& b_task,
                              double lambda)
      {
        const int task_dim = static_cast<int>(J_task.rows());
        Eigen::MatrixXd J_eff = J_task * N;
        Eigen::MatrixXd regularized =
            J_eff * J_eff.transpose()
          + lambda * Eigen::MatrixXd::Identity(task_dim, task_dim);
        Eigen::MatrixXd J_eff_pinv =
            J_eff.transpose() * regularized.ldlt().solve(
                Eigen::MatrixXd::Identity(task_dim, task_dim));

        Eigen::VectorXd residual = b_task - J_task * qdd_cmd;
        qdd_cmd += N * J_eff_pinv * residual;
        N = N * (Eigen::MatrixXd::Identity(nv, nv) - J_eff_pinv * J_eff);
      };

      auto task_accel_target = [&](int ee_frame_id,
                                   const Eigen::Vector3d& target_pos,
                                   const Eigen::Vector3d& target_vel,
                                   const Eigen::Vector3d& target_acc,
                                   Eigen::MatrixXd& J_lin,
                                   Eigen::Vector3d& b)
      {
        const auto& oMf = pin_data_->oMf[ee_frame_id];
        Eigen::Vector3d x = oMf.translation();

        Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(
            pin_model_, *pin_data_, ee_frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED);
        J_lin = J6.topRows<3>();

        Eigen::Vector3d x_dot = J_lin * qd;
        Eigen::Vector3d e = target_pos - x;
        Eigen::Vector3d e_dot = target_vel - x_dot;

        Eigen::VectorXd qdd_zero = Eigen::VectorXd::Zero(nv);
        pinocchio::forwardKinematics(pin_model_, *pin_data_bias_, q, qd, qdd_zero);
        auto bias = pinocchio::getFrameClassicalAcceleration(
            pin_model_, *pin_data_bias_, ee_frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED);

        Eigen::Vector3d x_ddot_des =
            target_acc
          + (Kp_task_.cwiseProduct(e) + Kd_task_.cwiseProduct(e_dot)) / m_des_;
        b = x_ddot_des - bias.linear();
      };

      Eigen::MatrixXd J_left, J_right;
      Eigen::Vector3d b_left, b_right;
      task_accel_target(left_ee_frame_id_,
                        left_target_pos, left_target_vel, left_target_acc,
                        J_left, b_left);
      task_accel_target(right_ee_frame_id_,
                        right_target_pos, right_target_vel, right_target_acc,
                        J_right, b_right);

      Eigen::MatrixXd J_left_x(1, nv);
      J_left_x.row(0) = J_left.row(0);
      Eigen::VectorXd b_left_x(1);
      b_left_x << b_left.x();

      Eigen::MatrixXd J_left_yz(2, nv);
      J_left_yz.row(0) = J_left.row(1);
      J_left_yz.row(1) = J_left.row(2);
      Eigen::Vector2d b_left_yz;
      b_left_yz << b_left.y(), b_left.z();

      // 2) Hierarchical acceleration tasks:
      //    left tool x > left yz path > right hold > posture.
      // Keep this close to the stable xz controller: do not over-constrain
      // intermediate links in yz, otherwise Arm_L1/R2/R3 saturate near turns.
      add_acc_task(J_left_x, b_left_x, 1e-8);
      add_acc_task(J_left_yz, b_left_yz, 5e-5);
      add_acc_task(J_right, b_right, 1e-3);

      Eigen::VectorXd qdd_posture = Eigen::VectorXd::Zero(nv);
      for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
        const std::string& name = pin_joint_names_[i];

        // 不要控制 gripper finger
        if (name.find("Arm_") != 0) {
          continue;
        }

        int pin_q = pin_q_idx_[i];
        int pin_v = pin_v_idx_[i];

        qdd_posture[pin_v] = Kq_posture_ * (q_rest_[pin_q] - q[pin_q])
                            - Dq_posture_ * qd[pin_v];
      }

      qdd_cmd += N * qdd_posture;
      tau = pinocchio_inverse_dynamics(q, qd, qdd_cmd);

      record_actual_joint_sample(t, q, qd);

      apply_torques_to_mujoco(d, tau);

      last_tau_ = tau;
      last_q_ = q;
      last_qd_ = qd;
      last_qdd_ = qdd_cmd;
      return;
    }

    // =====================================================================
    // Full dynamics operational-space impedance control
    // =====================================================================
    Eigen::VectorXd tau_left = compute_arm_torque(
        q, qd,
        left_ee_frame_id_,
        left_target_pos,
        left_target_vel,
        left_target_acc);

    Eigen::VectorXd tau_right = compute_arm_torque(
        q, qd,
        right_ee_frame_id_,
        right_target_pos,
        right_target_vel,
        right_target_acc);
    Eigen::VectorXd tau = tau_left + tau_right - pin_data_-> nle;  // nle already contains C·q̇ + g for all joints

    // Clamp torques
    const double max_torque = 200.0;
    for (int i = 0; i < pin_model_.nv; ++i)
      tau[i] = std::max(-max_torque, std::min(max_torque, tau[i]));

    apply_torques_to_mujoco(d, tau);

    last_tau_ = tau;
    last_q_ = q;
    last_qd_ = qd;
    last_qdd_ = Eigen::VectorXd::Zero(pin_model_.nv);
  }

  // Accessors for ROS2 publishing
  const Eigen::VectorXd& last_q()  const { return last_q_; }
  const Eigen::VectorXd& last_qd() const { return last_qd_; }
  const Eigen::VectorXd& last_qdd() const { return last_qdd_; }
  const Eigen::VectorXd& desired_q() const { return desired_q_; }
  const Eigen::VectorXd& desired_qd() const { return desired_qd_; }
  const Eigen::VectorXd& desired_qdd() const { return desired_qdd_; }
  const Eigen::VectorXd& last_tau() const { return last_tau_; }
  const std::vector<std::string>& joint_names() const { return pin_joint_names_; }
  const std::vector<int>& pin_q_idx() const { return pin_q_idx_; }
  const std::vector<int>& pin_v_idx() const { return pin_v_idx_; }
  const std::vector<int>& mj_qpos_adr() const { return mj_qpos_adr_; }
  const std::vector<int>& mj_dof_adr()  const { return mj_dof_adr_; }

  const pinocchio::Model& pin_model() const { return pin_model_; }
  const pinocchio::Data&  pin_data()  const { return *pin_data_; }
  int left_ee_frame_id()  const { return left_ee_frame_id_; }
  int right_ee_frame_id() const { return right_ee_frame_id_; }
  const Eigen::Vector3d& current_left_target()  const { return current_left_target_; }
  const Eigen::Vector3d& current_right_target() const { return current_right_target_; }
  const Eigen::Vector3d& l_shape_left_ee()  const { return l_shape_left_ee_; }
  const Eigen::Vector3d& l_shape_right_ee() const { return l_shape_right_ee_; }
  double fig8_freq() const { return fig8_freq_; }

private:
  // ========================================================================
  // Figure-8 (lemniscate of Gerono) offset
  // ========================================================================
  struct TrajectoryPoint
  {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d acc;
  };
  
  
  
  TrajectoryPoint compute_figure8_trajectory(double t) const
  {
    const double omega = 2.0 * M_PI * fig8_freq_;
    const double theta = omega * t;

    // Gerono 8字轨迹:
    // a = Ax * sin(theta)
    // b = Az * sin(theta) * cos(theta)
    const double a      = fig8_amp_x_ * std::sin(theta);
    const double b      = fig8_amp_z_ * std::sin(theta) * std::cos(theta);

    const double a_dot  = fig8_amp_x_ * omega * std::cos(theta);
    const double b_dot  = fig8_amp_z_ * omega * std::cos(2.0 * theta);

    const double a_ddot = -fig8_amp_x_ * omega * omega * std::sin(theta);
    const double b_ddot = -2.0 * fig8_amp_z_ * omega * omega * std::sin(2.0 * theta);

    TrajectoryPoint traj;
    traj.pos.setZero();
    traj.vel.setZero();
    traj.acc.setZero();

    if (fig8_plane_ == "xz") {
      traj.pos.x() = a;
      traj.pos.z() = b;

      traj.vel.x() = a_dot;
      traj.vel.z() = b_dot;

      traj.acc.x() = a_ddot;
      traj.acc.z() = b_ddot;
    }
    else if (fig8_plane_ == "yz") {
      traj.pos.y() = a;
      traj.pos.z() = b;

      traj.vel.y() = a_dot;
      traj.vel.z() = b_dot;

      traj.acc.y() = a_ddot;
      traj.acc.z() = b_ddot;
    }
    else if (fig8_plane_ == "xy") {
      traj.pos.x() = a;
      traj.pos.y() = b;

      traj.vel.x() = a_dot;
      traj.vel.y() = b_dot;

      traj.acc.x() = a_ddot;
      traj.acc.y() = b_ddot;
    }

    return traj;
  }

  // ========================================================================
  // Initialize L-shape: bend elbows to -90°, compute target EE positions
  // ========================================================================
  void init_l_shape_configuration(const Eigen::VectorXd& q_current)
  {
    (void)q_current;  // not used — we set the full home posture explicitly
    q_rest_ = Eigen::VectorXd::Zero(pin_model_.nq);

    auto set_joint_q = [&](const std::string& name, double val) {
      for (int j = 1; j < pin_model_.njoints; ++j) {
        if (pin_model_.names[j] == name) {
          int q_idx = pin_model_.joints[j].idx_q();
          q_rest_[q_idx] = val;
          printf("Set %s q[%d] = %.2f rad (%.0f deg)\n",
                 name.c_str(), q_idx, val, val * 180.0 / M_PI);
          break;
        }
      }
    };

    // Left arm: full home posture
    set_joint_q("Arm_L1_Joint",  1.7);
    set_joint_q("Arm_L2_Joint", -1.1);
    set_joint_q("Arm_L3_Joint", -1.1);
    set_joint_q("Arm_L4_Joint", -2.0);
    set_joint_q("Arm_L5_Joint", -0.37);
    set_joint_q("Arm_L6_Joint",  0.13);
    set_joint_q("Arm_L7_Joint",  0.55);

    // Right arm: full home posture (mirrored)
    set_joint_q("Arm_R1_Joint", -1.7);
    set_joint_q("Arm_R2_Joint", -1.1);
    set_joint_q("Arm_R3_Joint",  1.1);
    set_joint_q("Arm_R4_Joint", -2.0);
    set_joint_q("Arm_R5_Joint",  0.37);
    set_joint_q("Arm_R6_Joint",  0.13);
    set_joint_q("Arm_R7_Joint", -0.55);

    // FK with L-shape config to get target EE positions
    pinocchio::Data data_lshape(pin_model_);
    pinocchio::forwardKinematics(pin_model_, data_lshape, q_rest_);
    pinocchio::updateFramePlacements(pin_model_, data_lshape);

    l_shape_left_ee_  = data_lshape.oMf[left_ee_frame_id_].translation();
    l_shape_right_ee_ = data_lshape.oMf[right_ee_frame_id_].translation();
    l_shape_left_upper_ = data_lshape.oMf[left_upper_frame_id_].translation();
    l_shape_left_forearm_ = data_lshape.oMf[left_forearm_frame_id_].translation();

    printf("L-shape targets: L_ee=[%.3f, %.3f, %.3f]  R_ee=[%.3f, %.3f, %.3f]\n",
           l_shape_left_ee_.x(),  l_shape_left_ee_.y(),  l_shape_left_ee_.z(),
           l_shape_right_ee_.x(), l_shape_right_ee_.y(), l_shape_right_ee_.z());
    printf("L-shape left shape y: upper=%.3f  forearm=%.3f\n",
           l_shape_left_upper_.y(), l_shape_left_forearm_.y());
  }

  // ========================================================================
  // Joint & actuator mapping
  // ========================================================================
  void build_joint_map()
  {
    pin_v_is_arm_.assign(pin_model_.nv, false);

    for (int j = 1; j < pin_model_.njoints; ++j) {
      int j_nq = pin_model_.joints[j].nq();
      int j_nv = pin_model_.joints[j].nv();
      if (j_nq < 1 || j_nv < 1) continue;

      int idx_q = pin_model_.joints[j].idx_q();
      int idx_v = pin_model_.joints[j].idx_v();
      const std::string& name = pin_model_.names[j];

      int mj_id = mj_name2id(mj_model_, mjOBJ_JOINT, name.c_str());
      if (mj_id < 0) continue;
      int mj_qpos_adr = mj_model_->jnt_qposadr[mj_id];
      int mj_dof_adr  = mj_model_->jnt_dofadr[mj_id];
      if (mj_qpos_adr < 0 || mj_dof_adr < 0) continue;

      for (int k = 0; k < j_nq; ++k) {
        pin_joint_names_.push_back(name);
        pin_q_idx_.push_back(idx_q + k);
        pin_v_idx_.push_back(idx_v + k);
        mj_qpos_adr_.push_back(mj_qpos_adr + k);
        mj_dof_adr_.push_back(mj_dof_adr + k);
        pin_v_is_arm_[idx_v + k] = (name.find("Arm_") == 0);
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
      if (act_id < 0) continue;
      for (int k = 0; k < j_nv; ++k)
        mj_actuator_of_pin_v_[idx_v + k] = act_id;
    }
  }

  void find_elbow_indices()
  {
    for (int j = 1; j < pin_model_.njoints; ++j) {
      const std::string& name = pin_model_.names[j];
      if (name == "Arm_L4_Joint")
        printf("Left  elbow (Arm_L4_Joint) at v_idx=%d\n",
               pin_model_.joints[j].idx_v());
      if (name == "Arm_R4_Joint")
        printf("Right elbow (Arm_R4_Joint) at v_idx=%d\n",
               pin_model_.joints[j].idx_v());
    }
  }

  // ========================================================================
  // State I/O
  // ========================================================================
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
      if (act_id < 0) continue;
      if (i >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[i]) continue;

      double cmd = tau[i];

      // MuJoCo actuator ctrlrange
      double lo = mj_model_->actuator_ctrlrange[2 * act_id + 0];
      double hi = mj_model_->actuator_ctrlrange[2 * act_id + 1];

      bool saturated = false;
      if (cmd < lo) {
        cmd = lo;
        saturated = true;
      }
      if (cmd > hi) {
        cmd = hi;
        saturated = true;
      }

      if (saturated && should_log_saturation(act_id, d->time)) {
        const char* act_name = mj_id2name(mj_model_, mjOBJ_ACTUATOR, act_id);
        printf("[SAT] actuator=%s tau=%.2f clipped to [%.2f, %.2f]\n",
               act_name ? act_name : "unknown", tau[i], lo, hi);
      }

      d->ctrl[act_id] = cmd;
    }
  }

  Eigen::VectorXd pinocchio_inverse_dynamics(const Eigen::VectorXd& q,
                                             const Eigen::VectorXd& qd,
                                             const Eigen::VectorXd& qdd_cmd)
  {
    Eigen::VectorXd tau = pinocchio::rnea(pin_model_, *pin_data_, q, qd, qdd_cmd);

    for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
      const int pin_v = pin_v_idx_[i];
      const int mj_dof = mj_dof_adr_[i];
      if (pin_v < 0 || pin_v >= pin_model_.nv ||
          pin_v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[pin_v]) {
        continue;
      }

      // Pinocchio's MJCF parser does not model MuJoCo's joint-side passive
      // terms. Add them explicitly so the torque is closer to mj_inverse while
      // keeping Pinocchio as the rigid-body inverse dynamics engine.
      const double armature = mj_model_->dof_armature[mj_dof];
      const double damping = mj_model_->dof_damping[mj_dof];
      const double frictionloss = mj_model_->dof_frictionloss[mj_dof];
      const double v = qd[pin_v];
      const double friction_sign = std::tanh(v / 0.01);

      tau[pin_v] += armature * qdd_cmd[pin_v]
                  + damping * v
                  + frictionloss * friction_sign;
    }

    for (int i = 0; i < pin_model_.nv; ++i) {
      if (i >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[i]) {
        tau[i] = 0.0;
      }
    }
    return tau;
  }

  struct JointSample
  {
    double t = 0.0;
    Eigen::VectorXd q;
    Eigen::VectorXd qd;
    Eigen::VectorXd qdd;
  };

  struct ReplayReference
  {
    JointSample sample;
    Eigen::VectorXd qdd_ff;
    size_t lo = 0;
    size_t hi = 0;
    double alpha = 0.0;
  };

  void record_actual_joint_sample(double t,
                                  const Eigen::VectorXd& q,
                                  const Eigen::VectorXd& qd)
  {
    JointSample sample;
    sample.t = std::max(0.0, t);
    sample.q = q;
    sample.qd = qd;
    sample.qdd = Eigen::VectorXd::Zero(pin_model_.nv);

    if (!recorded_trajectory_.empty()) {
      const JointSample& prev = recorded_trajectory_.back();
      const double dt = std::max(sample.t - prev.t, 1e-6);
      sample.qdd = (sample.qd - prev.qd) / dt;
      recorded_trajectory_.back().qdd = sample.qdd;
    }

    recorded_trajectory_.push_back(sample);
  }

  void rebuild_recorded_trajectory_derivatives()
  {
    const size_t n = recorded_trajectory_.size();
    if (n < 3) return;

    const double period = std::max(replay_cycle_duration_, 1e-6);
    for (size_t i = 0; i < n; ++i) {
      recorded_trajectory_[i].qd = Eigen::VectorXd::Zero(pin_model_.nv);
      recorded_trajectory_[i].qdd = Eigen::VectorXd::Zero(pin_model_.nv);
    }

    auto segment_dt = [&](size_t lo, size_t hi) {
      double dt = recorded_trajectory_[hi].t - recorded_trajectory_[lo].t;
      if (dt <= 0.0) dt += period;
      return std::max(dt, 1e-6);
    };

    auto pchip_slope = [](double d_prev, double d_next,
                          double h_prev, double h_next) {
      if (d_prev == 0.0 || d_next == 0.0 ||
          std::signbit(d_prev) != std::signbit(d_next)) {
        return 0.0;
      }
      const double w1 = 2.0 * h_next + h_prev;
      const double w2 = h_next + 2.0 * h_prev;
      return (w1 + w2) / (w1 / d_prev + w2 / d_next);
    };

    for (size_t j = 0; j < pin_joint_names_.size(); ++j) {
      const int pin_q = pin_q_idx_[j];
      const int pin_v = pin_v_idx_[j];
      if (pin_q < 0 || pin_q >= pin_model_.nq ||
          pin_v < 0 || pin_v >= pin_model_.nv ||
          pin_v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[pin_v]) {
        continue;
      }

      for (size_t i = 0; i < n - 1; ++i) {
        const size_t prev = (i == 0) ? n - 2 : i - 1;
        const size_t next = i + 1;
        const double h_prev = (i == 0)
            ? std::max(period - recorded_trajectory_[prev].t, 1e-6)
            : segment_dt(prev, i);
        const double h_next = segment_dt(i, next);
        const double d_prev =
            (recorded_trajectory_[i].q[pin_q] -
             recorded_trajectory_[prev].q[pin_q]) / h_prev;
        const double d_next =
            (recorded_trajectory_[next].q[pin_q] -
             recorded_trajectory_[i].q[pin_q]) / h_next;

        recorded_trajectory_[i].qd[pin_v] =
            pchip_slope(d_prev, d_next, h_prev, h_next);
      }
    }

    recorded_trajectory_.back().qd = recorded_trajectory_.front().qd;
  }

  void finish_recording(double cycle_duration)
  {
    replay_cycle_duration_ = cycle_duration;
    recorded_trajectory_.front().t = 0.0;
    recorded_trajectory_.back().t = replay_cycle_duration_;
    recorded_trajectory_.back().q = recorded_trajectory_.front().q;
    rebuild_recorded_trajectory_derivatives();

    recording_complete_ = true;
    replay_started_ = false;
    replay_start_time_ = start_sim_time_ + replay_cycle_duration_;
    replay_cycle_index_ = 1;
    initialize_ilc_feedforward();
    reset_replay_metrics();

    printf("[REPLAY] Recorded first real joint trajectory: samples=%zu duration=%.3f s\n",
           recorded_trajectory_.size(), replay_cycle_duration_);
    printf("[REPLAY] From cycle 2 onward, using C1 shape-preserving recorded joints as desired and tracking with joint PD + Pinocchio RNEA.\n");
    printf("[ILC] Feed-forward qdd will be updated after each replay cycle from the previous cycle tracking error.\n");
  }

  ReplayReference sample_replay_reference(double t_in_cycle) const
  {
    ReplayReference out;
    out.sample.t = t_in_cycle;
    out.sample.q = Eigen::VectorXd::Zero(pin_model_.nq);
    out.sample.qd = Eigen::VectorXd::Zero(pin_model_.nv);
    out.sample.qdd = Eigen::VectorXd::Zero(pin_model_.nv);
    out.qdd_ff = Eigen::VectorXd::Zero(pin_model_.nv);

    if (recorded_trajectory_.empty()) {
      return out;
    }
    if (recorded_trajectory_.size() == 1 || t_in_cycle <= recorded_trajectory_.front().t) {
      out.sample = recorded_trajectory_.front();
      out.qdd_ff = ilc_qdd_ff_.empty() ? out.sample.qdd : ilc_qdd_ff_.front();
      return out;
    }
    if (t_in_cycle >= recorded_trajectory_.back().t) {
      out.sample = recorded_trajectory_.back();
      out.lo = recorded_trajectory_.size() - 1;
      out.hi = out.lo;
      out.qdd_ff = ilc_qdd_ff_.empty() ? out.sample.qdd : ilc_qdd_ff_.back();
      return out;
    }

    size_t hi = 1;
    while (hi < recorded_trajectory_.size() &&
           recorded_trajectory_[hi].t < t_in_cycle) {
      ++hi;
    }
    const JointSample& a = recorded_trajectory_[hi - 1];
    const JointSample& b = recorded_trajectory_[hi];
    const double span = std::max(b.t - a.t, 1e-6);
    const double alpha = std::max(0.0, std::min(1.0, (t_in_cycle - a.t) / span));

    out.lo = hi - 1;
    out.hi = hi;
    out.alpha = alpha;
    const double s = alpha;
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    const double h10 = s3 - 2.0 * s2 + s;
    const double h01 = -2.0 * s3 + 3.0 * s2;
    const double h11 = s3 - s2;
    const double dh00 = (6.0 * s2 - 6.0 * s) / span;
    const double dh10 = 3.0 * s2 - 4.0 * s + 1.0;
    const double dh01 = (-6.0 * s2 + 6.0 * s) / span;
    const double dh11 = 3.0 * s2 - 2.0 * s;
    const double ddh00 = (12.0 * s - 6.0) / (span * span);
    const double ddh10 = (6.0 * s - 4.0) / span;
    const double ddh01 = (-12.0 * s + 6.0) / (span * span);
    const double ddh11 = (6.0 * s - 2.0) / span;

    Eigen::VectorXd a_q_tangent = Eigen::VectorXd::Zero(pin_model_.nq);
    Eigen::VectorXd b_q_tangent = Eigen::VectorXd::Zero(pin_model_.nq);
    for (size_t j = 0; j < pin_joint_names_.size(); ++j) {
      const int pin_q = pin_q_idx_[j];
      const int pin_v = pin_v_idx_[j];
      if (pin_q < 0 || pin_q >= pin_model_.nq ||
          pin_v < 0 || pin_v >= pin_model_.nv) {
        continue;
      }
      a_q_tangent[pin_q] = a.qd[pin_v];
      b_q_tangent[pin_q] = b.qd[pin_v];
    }

    out.sample.q = h00 * a.q + h10 * span * a_q_tangent
                 + h01 * b.q + h11 * span * b_q_tangent;
    Eigen::VectorXd q_dot_as_q = dh00 * a.q + dh10 * a_q_tangent
                               + dh01 * b.q + dh11 * b_q_tangent;
    Eigen::VectorXd q_ddot_as_q = ddh00 * a.q + ddh10 * a_q_tangent
                                + ddh01 * b.q + ddh11 * b_q_tangent;

    out.sample.qd.setZero(pin_model_.nv);
    out.sample.qdd.setZero(pin_model_.nv);
    for (size_t j = 0; j < pin_joint_names_.size(); ++j) {
      const int pin_q = pin_q_idx_[j];
      const int pin_v = pin_v_idx_[j];
      if (pin_q < 0 || pin_q >= pin_model_.nq ||
          pin_v < 0 || pin_v >= pin_model_.nv) {
        continue;
      }
      out.sample.qd[pin_v] = q_dot_as_q[pin_q];
      out.sample.qdd[pin_v] = q_ddot_as_q[pin_q];
    }
    if (!ilc_qdd_ff_.empty() && hi < ilc_qdd_ff_.size()) {
      out.qdd_ff = (1.0 - alpha) * ilc_qdd_ff_[hi - 1] + alpha * ilc_qdd_ff_[hi];
    } else {
      out.qdd_ff = out.sample.qdd;
    }
    return out;
  }

  void initialize_ilc_feedforward()
  {
    ilc_qdd_ff_.clear();
    ilc_qdd_delta_sum_.clear();
    ilc_weight_sum_.clear();
    ilc_qdd_ff_.reserve(recorded_trajectory_.size());
    for (const auto& sample : recorded_trajectory_) {
      (void)sample;
      ilc_qdd_ff_.push_back(Eigen::VectorXd::Zero(pin_model_.nv));
    }
    reset_ilc_accumulators();
  }

  void smooth_ilc_feedforward()
  {
    const size_t n = ilc_qdd_ff_.size();
    if (n < 3) return;

    auto cyclic_sample_index = [&](int idx) -> size_t {
      if (idx < 0) return n - 2;
      if (idx >= static_cast<int>(n)) return 1;
      return static_cast<size_t>(idx);
    };

    for (int pass = 0; pass < 3; ++pass) {
      std::vector<Eigen::VectorXd> filtered = ilc_qdd_ff_;
      for (size_t i = 0; i < n; ++i) {
        const size_t prev = cyclic_sample_index(static_cast<int>(i) - 1);
        const size_t next = cyclic_sample_index(static_cast<int>(i) + 1);
        filtered[i] = 0.25 * ilc_qdd_ff_[prev] + 0.50 * ilc_qdd_ff_[i]
                    + 0.25 * ilc_qdd_ff_[next];
      }
      ilc_qdd_ff_.swap(filtered);
    }

    for (auto& qdd : ilc_qdd_ff_) {
      for (int v = 0; v < qdd.size(); ++v) {
        if (v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[v]) {
          qdd[v] = 0.0;
        } else {
          qdd[v] = std::max(-ilc_max_qdd_ff_,
                            std::min(ilc_max_qdd_ff_, qdd[v]));
        }
      }
    }
    ilc_qdd_ff_.back() = ilc_qdd_ff_.front();
  }

  void reset_ilc_accumulators()
  {
    ilc_qdd_delta_sum_.assign(
        recorded_trajectory_.size(), Eigen::VectorXd::Zero(pin_model_.nv));
    ilc_weight_sum_.assign(recorded_trajectory_.size(), 0.0);
  }

  void accumulate_ilc_error(const ReplayReference& ref,
                            const Eigen::VectorXd& q,
                            const Eigen::VectorXd& qd)
  {
    if (recorded_trajectory_.empty() ||
        ilc_qdd_delta_sum_.size() != recorded_trajectory_.size()) {
      return;
    }

    Eigen::VectorXd delta = Eigen::VectorXd::Zero(pin_model_.nv);
    for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
      const int pin_q = pin_q_idx_[i];
      const int pin_v = pin_v_idx_[i];
      if (pin_v < 0 || pin_v >= pin_model_.nv ||
          pin_v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[pin_v]) {
        continue;
      }
      delta[pin_v] = ilc_Lp_ * (ref.sample.q[pin_q] - q[pin_q])
                   + ilc_Ld_ * (ref.sample.qd[pin_v] - qd[pin_v]);
      delta[pin_v] = std::max(-ilc_max_delta_qdd_,
                              std::min(ilc_max_delta_qdd_, delta[pin_v]));
    }

    auto add_weighted = [&](size_t idx, double weight) {
      if (idx >= ilc_qdd_delta_sum_.size() || weight <= 0.0) return;
      ilc_qdd_delta_sum_[idx] += weight * delta;
      ilc_weight_sum_[idx] += weight;
    };

    if (ref.lo == ref.hi) {
      add_weighted(ref.lo, 1.0);
    } else {
      add_weighted(ref.lo, 1.0 - ref.alpha);
      add_weighted(ref.hi, ref.alpha);
    }
  }

  void apply_ilc_update(int completed_cycle)
  {
    if (ilc_qdd_ff_.size() != recorded_trajectory_.size() ||
        ilc_qdd_delta_sum_.size() != recorded_trajectory_.size()) {
      return;
    }

    double rms_delta = 0.0;
    double max_delta = 0.0;
    int count = 0;
    for (size_t idx = 0; idx < ilc_qdd_ff_.size(); ++idx) {
      if (ilc_weight_sum_[idx] <= 1e-12) continue;
      Eigen::VectorXd avg_delta = ilc_qdd_delta_sum_[idx] / ilc_weight_sum_[idx];
      for (int v = 0; v < avg_delta.size(); ++v) {
        if (v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[v]) {
          avg_delta[v] = 0.0;
          continue;
        }
        const double learned_delta = ilc_learning_rate_ * avg_delta[v];
        ilc_qdd_ff_[idx][v] += learned_delta;
        ilc_qdd_ff_[idx][v] = std::max(-ilc_max_qdd_ff_,
                                       std::min(ilc_max_qdd_ff_, ilc_qdd_ff_[idx][v]));
        rms_delta += learned_delta * learned_delta;
        max_delta = std::max(max_delta, std::abs(learned_delta));
        ++count;
      }
    }

    if (count > 0) {
      rms_delta = std::sqrt(rms_delta / static_cast<double>(count));
    }
    smooth_ilc_feedforward();
    printf("[ILC][after cycle %d] updated next-cycle qdd_ff: delta_rms=%.6f rad/s^2 delta_max=%.6f rad/s^2 active_samples=%zu\n",
           completed_cycle, rms_delta, max_delta, ilc_qdd_ff_.size());
    reset_ilc_accumulators();
  }

  void reset_replay_metrics()
  {
    metric_samples_ = 0;
    metric_q_sq_sum_ = 0.0;
    metric_qd_sq_sum_ = 0.0;
    metric_ee_sq_sum_ = 0.0;
    metric_q_max_ = 0.0;
    metric_qd_max_ = 0.0;
    metric_ee_max_ = 0.0;
  }

  void update_replay_metrics(const Eigen::VectorXd& q,
                             const Eigen::VectorXd& qd,
                             const JointSample& ref)
  {
    double q_sq = 0.0;
    double qd_sq = 0.0;
    double q_max = 0.0;
    double qd_max = 0.0;
    int n = 0;

    for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
      const int pin_q = pin_q_idx_[i];
      const int pin_v = pin_v_idx_[i];
      if (pin_v < 0 || pin_v >= pin_model_.nv ||
          pin_v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[pin_v]) {
        continue;
      }
      const double qe = ref.q[pin_q] - q[pin_q];
      const double qde = ref.qd[pin_v] - qd[pin_v];
      q_sq += qe * qe;
      qd_sq += qde * qde;
      q_max = std::max(q_max, std::abs(qe));
      qd_max = std::max(qd_max, std::abs(qde));
      ++n;
    }

    const Eigen::Vector3d left_pos = pin_data_->oMf[left_ee_frame_id_].translation();
    pinocchio::Data ref_data(pin_model_);
    pinocchio::forwardKinematics(pin_model_, ref_data, ref.q, ref.qd);
    pinocchio::updateFramePlacements(pin_model_, ref_data);
    const Eigen::Vector3d left_ref = ref_data.oMf[left_ee_frame_id_].translation();
    const double ee_err = (left_ref - left_pos).norm();

    if (n > 0) {
      metric_q_sq_sum_ += q_sq / n;
      metric_qd_sq_sum_ += qd_sq / n;
    }
    metric_ee_sq_sum_ += ee_err * ee_err;
    metric_q_max_ = std::max(metric_q_max_, q_max);
    metric_qd_max_ = std::max(metric_qd_max_, qd_max);
    metric_ee_max_ = std::max(metric_ee_max_, ee_err);
    ++metric_samples_;
  }

  void print_replay_metrics(int cycle_index)
  {
    if (metric_samples_ <= 0) return;
    const double inv = 1.0 / static_cast<double>(metric_samples_);
    printf("[REPLAY][cycle %d] tracking: q_rms=%.6f rad q_max=%.6f rad | qd_rms=%.6f rad/s qd_max=%.6f rad/s | left_ee_rms=%.6f m left_ee_max=%.6f m samples=%d\n",
           cycle_index,
           std::sqrt(metric_q_sq_sum_ * inv), metric_q_max_,
           std::sqrt(metric_qd_sq_sum_ * inv), metric_qd_max_,
           std::sqrt(metric_ee_sq_sum_ * inv), metric_ee_max_,
           metric_samples_);
  }

  Eigen::VectorXd limit_joint_velocity_reference(const Eigen::VectorXd& qd_ref,
                                                 double max_qdd)
  {
    Eigen::VectorXd limited = qd_ref;
    if (desired_qd_.size() != pin_model_.nv) {
      desired_qd_ = Eigen::VectorXd::Zero(pin_model_.nv);
    }

    const double max_delta_qd = std::max(0.0, max_qdd) * std::max(dt_, 1e-6);
    for (int v = 0; v < limited.size(); ++v) {
      if (v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[v]) {
        limited[v] = 0.0;
        continue;
      }
      const double delta = limited[v] - desired_qd_[v];
      limited[v] = desired_qd_[v]
                 + std::max(-max_delta_qd, std::min(max_delta_qd, delta));
    }
    return limited;
  }

  void control_replay_step(mjData* d,
                           const Eigen::VectorXd& q,
                           const Eigen::VectorXd& qd,
                           double t)
  {
    (void)t;
    if (!replay_started_) {
      replay_started_ = true;
      replay_start_time_ = d->time;
      replay_cycle_index_ = 2;
      reset_replay_metrics();
      printf("[REPLAY] Starting replay cycle %d at sim_time=%.3f\n",
             replay_cycle_index_, d->time);
    }

    const double replay_elapsed = std::max(0.0, d->time - replay_start_time_);
    const int completed_cycles = static_cast<int>(std::floor(
        replay_elapsed / std::max(replay_cycle_duration_, 1e-6)));
    const int current_cycle = 2 + completed_cycles;
    if (current_cycle != replay_cycle_index_) {
      print_replay_metrics(replay_cycle_index_);
      apply_ilc_update(replay_cycle_index_);
      replay_cycle_index_ = current_cycle;
      reset_replay_metrics();
      printf("[REPLAY] Starting replay cycle %d at sim_time=%.3f\n",
             replay_cycle_index_, d->time);
    }

    double t_in_cycle = std::fmod(replay_elapsed, std::max(replay_cycle_duration_, 1e-6));
    if (t_in_cycle < 0.0) t_in_cycle += replay_cycle_duration_;
    ReplayReference ref = sample_replay_reference(t_in_cycle);

    desired_q_ = ref.sample.q;
    Eigen::VectorXd limited_qd =
        limit_joint_velocity_reference(ref.sample.qd, max_desired_qdd_);
    desired_qdd_ = (limited_qd - desired_qd_) / std::max(dt_, 1e-6);
    desired_qd_ = limited_qd;

    pinocchio::Data ref_data(pin_model_);
    pinocchio::forwardKinematics(pin_model_, ref_data, ref.sample.q, desired_qd_);
    pinocchio::updateFramePlacements(pin_model_, ref_data);
    current_left_target_ = ref_data.oMf[left_ee_frame_id_].translation();
    current_right_target_ = ref_data.oMf[right_ee_frame_id_].translation();

    Eigen::VectorXd qdd_cmd = ref.qdd_ff;
    for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
      const int pin_q = pin_q_idx_[i];
      const int pin_v = pin_v_idx_[i];
      if (pin_v < 0 || pin_v >= pin_model_.nv ||
          pin_v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[pin_v]) {
        continue;
      }
      qdd_cmd[pin_v] += replay_Kp_ * (ref.sample.q[pin_q] - q[pin_q])
                      + replay_Kd_ * (desired_qd_[pin_v] - qd[pin_v]);
    }

    Eigen::VectorXd tau = pinocchio_inverse_dynamics(q, qd, qdd_cmd);
    apply_torques_to_mujoco(d, tau);

    last_tau_ = tau;
    last_q_ = q;
    last_qd_ = qd;
    last_qdd_ = qdd_cmd;
    JointSample limited_ref = ref.sample;
    limited_ref.qd = desired_qd_;
    limited_ref.qdd = desired_qdd_;
    update_replay_metrics(q, qd, limited_ref);
    ref.sample = limited_ref;
    accumulate_ilc_error(ref, q, qd);
  }

  void update_desired_joint_state(const Eigen::Vector3d& left_target_pos,
                                  const Eigen::Vector3d& left_target_vel,
                                  const Eigen::Vector3d& right_target_pos,
                                  const Eigen::Vector3d& right_target_vel)
  {
    if (desired_q_.size() != pin_model_.nq || desired_qd_.size() != pin_model_.nv) {
      if (q_rest_.size() == pin_model_.nq) {
        desired_q_ = q_rest_;
      } else {
        desired_q_ = Eigen::VectorXd::Zero(pin_model_.nq);
      }
      desired_qd_ = Eigen::VectorXd::Zero(pin_model_.nv);
      desired_qdd_ = Eigen::VectorXd::Zero(pin_model_.nv);
    }

    auto clamp_desired_q = [&]() {
      for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
        const int pin_q = pin_q_idx_[i];
        if (pin_q < 0 || pin_q >= desired_q_.size()) continue;

        if (pin_q < pin_model_.lowerPositionLimit.size() &&
            std::isfinite(pin_model_.lowerPositionLimit[pin_q]) &&
            std::isfinite(pin_model_.upperPositionLimit[pin_q])) {
          const double lower = pin_model_.lowerPositionLimit[pin_q];
          const double upper = pin_model_.upperPositionLimit[pin_q];
          desired_q_[pin_q] = std::max(lower, std::min(upper, desired_q_[pin_q]));
        }
      }
    };

    auto build_task = [&](Eigen::MatrixXd& J, Eigen::VectorXd& e) {
      pinocchio::forwardKinematics(pin_model_, *pin_data_desired_, desired_q_);
      pinocchio::updateFramePlacements(pin_model_, *pin_data_desired_);
      pinocchio::computeJointJacobians(pin_model_, *pin_data_desired_, desired_q_);

      Eigen::MatrixXd J_left6 = pinocchio::getFrameJacobian(
          pin_model_, *pin_data_desired_, left_ee_frame_id_,
          pinocchio::LOCAL_WORLD_ALIGNED);
      Eigen::MatrixXd J_right6 = pinocchio::getFrameJacobian(
          pin_model_, *pin_data_desired_, right_ee_frame_id_,
          pinocchio::LOCAL_WORLD_ALIGNED);

      J.setZero(6, pin_model_.nv);
      J.topRows<3>() = J_left6.topRows<3>();
      J.bottomRows<3>() = J_right6.topRows<3>();

      for (int v = 0; v < pin_model_.nv; ++v) {
        if (v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[v]) {
          J.col(v).setZero();
        }
      }

      e.resize(6);
      e.head<3>() = left_target_pos -
          pin_data_desired_->oMf[left_ee_frame_id_].translation();
      e.tail<3>() = right_target_pos -
          pin_data_desired_->oMf[right_ee_frame_id_].translation();
    };

    // Damped least-squares IK for the desired EE positions.
    for (int iter = 0; iter < 8; ++iter) {
      Eigen::MatrixXd J;
      Eigen::VectorXd e;
      build_task(J, e);
      if (e.norm() < 1e-5) break;

      const double lambda = 1e-4;
      Eigen::MatrixXd A = J * J.transpose()
        + lambda * Eigen::MatrixXd::Identity(J.rows(), J.rows());
      Eigen::VectorXd dq = J.transpose() * A.ldlt().solve(e);

      const double max_step = 0.03;
      for (int v = 0; v < dq.size(); ++v) {
        dq[v] = std::max(-max_step, std::min(max_step, dq[v]));
      }

      for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
        const int pin_q = pin_q_idx_[i];
        const int pin_v = pin_v_idx_[i];
        if (pin_q >= 0 && pin_q < desired_q_.size() &&
            pin_v >= 0 && pin_v < dq.size()) {
          desired_q_[pin_q] += dq[pin_v];
        }
      }
      clamp_desired_q();
    }

    // Desired joint velocity from task-space desired velocity at q_des.
    Eigen::MatrixXd J;
    Eigen::VectorXd e_unused;
    build_task(J, e_unused);

    Eigen::VectorXd task_vel(6);
    task_vel.head<3>() = left_target_vel;
    task_vel.tail<3>() = right_target_vel;

    const double lambda = 1e-4;
    Eigen::MatrixXd A = J * J.transpose()
      + lambda * Eigen::MatrixXd::Identity(J.rows(), J.rows());
    Eigen::VectorXd qd_ref = J.transpose() * A.ldlt().solve(task_vel);

    for (int v = 0; v < qd_ref.size(); ++v) {
      if (v >= static_cast<int>(pin_v_is_arm_.size()) || !pin_v_is_arm_[v]) {
        qd_ref[v] = 0.0;
      } else {
        qd_ref[v] = std::max(-2.0, std::min(2.0, qd_ref[v]));
      }
    }

    qd_ref = limit_joint_velocity_reference(qd_ref, max_desired_qdd_);

    desired_qdd_ = (qd_ref - desired_qd_) / std::max(dt_, 1e-6);
    desired_qd_ = qd_ref;
  }

  bool should_log_saturation(int act_id, double sim_time)
  {
    if (act_id < 0) return false;
    if (last_saturation_log_time_.size() != static_cast<size_t>(mj_model_->nu)) {
      last_saturation_log_time_.assign(mj_model_->nu, -1e9);
    }

    if (sim_time - last_saturation_log_time_[act_id] < 0.5) {
      return false;
    }

    last_saturation_log_time_[act_id] = sim_time;
    return true;
  }

  // ========================================================================
  // Full operational-space impedance control with dynamic decoupling
  //
  //   τ = M·J̄·(ẍ_des − J̇·q̇)  +  (I − Jᵀ·J̄ᵀ)·τ_posture  +  nle
  //
  //   J̄   = M⁻¹·Jᵀ·(J·M⁻¹·Jᵀ)⁻¹    dynamically consistent pseudo-inverse
  //   ẍ_des = target_acc + M_des⁻¹·(Kp·Δx + Kd·Δẋ)   with feed-forward
  //   nle = C·q̇ + g                   non-linear effects (Coriolis + gravity)
  //
  // The (I − Jᵀ·J̄ᵀ) term projects τ_posture into the null-space of the
  // task, so posture control never interferes with EE tracking.
  // ========================================================================
  Eigen::VectorXd compute_arm_torque(
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& qd,
      int ee_frame_id,
      const Eigen::Vector3d& target_pos,
      const Eigen::Vector3d& target_vel,
      const Eigen::Vector3d& target_acc)
  {
    const int nv = pin_model_.nv;

    // --- Current EE state ---
    const auto& oMf = pin_data_->oMf[ee_frame_id];
    Eigen::Vector3d x = oMf.translation();
    Eigen::Vector3d e = target_pos - x;                   // position error

    // --- Jacobian (linear part, 3×N) ---
    Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(
        pin_model_, *pin_data_, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::MatrixXd J_lin = J6.topRows<3>();
    Eigen::Vector3d x_dot = J_lin * qd;                   // EE velocity

    // 速度误差
    Eigen::Vector3d e_dot = target_vel - x_dot;

    // --- Pinocchio-computed dynamics ---
    const Eigen::MatrixXd& M   = pin_data_->M;            // nv×nv mass matrix
    const Eigen::VectorXd& nle = pin_data_->nle;          // C·q̇ + g

    // --- Task-space inertia: Λ = (J·M⁻¹·Jᵀ)⁻¹ ---
    // Solve M·X = Jᵀ  →  X = M⁻¹·Jᵀ  (3 linear systems)
    Eigen::MatrixXd M_inv_Jt = M.llt().solve(J_lin.transpose());  // nv×3
    Eigen::Matrix3d Lambda_inv = J_lin * M_inv_Jt;                 // 3×3

    double det = Lambda_inv.determinant();
    if (std::abs(det) < 1e-12) {
      // Near kinematic singularity — fall back to joint-space PD + gravity
      Eigen::VectorXd q_err = q_rest_ - q;
      return Kp_null_ * q_err - Kd_null_ * qd + nle;
    }

    Eigen::Matrix3d Lambda = Lambda_inv.inverse();        // task-space inertia

    // --- Dynamically consistent pseudo-inverse: J̄ = M⁻¹·Jᵀ·Λ ---
    Eigen::MatrixXd J_bar = M_inv_Jt * Lambda;            // nv×3

    // --- Desired task-space acceleration (impedance law with feed-forward) ---
    // ẍ_des = target_acc + M_des⁻¹·(Kp·Δx + Kd·Δẋ)
    double inv_m = 1.0 / m_des_;
    Eigen::Vector3d x_ddot_des =
        target_acc + inv_m * (Kp_task_.cwiseProduct(e) + Kd_task_.cwiseProduct(e_dot));

    // --- Bias acceleration: J̇·q̇ (velocity-product term at EE) ---
    // Run FK with q̈=0 on the bias data; getFrameClassicalAcceleration
    // then returns J·0 + J̇·q̇ = J̇·q̇.
    Eigen::VectorXd qdd_zero = Eigen::VectorXd::Zero(nv);
    pinocchio::forwardKinematics(pin_model_, *pin_data_bias_, q, qd, qdd_zero);
    auto bias = pinocchio::getFrameClassicalAcceleration(
        pin_model_, *pin_data_bias_, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::Vector3d J_dot_qd = bias.linear();             // J̇·q̇

    // --- Desired joint acceleration (task contribution) ---
    Eigen::VectorXd q_ddot_des = J_bar * (x_ddot_des - J_dot_qd);

    // --- Null-space posture acceleration ---
    // Nᵀ = I − Jᵀ·J̄ᵀ
    Eigen::MatrixXd I_nv = Eigen::MatrixXd::Identity(nv, nv);
    Eigen::MatrixXd N_transpose = I_nv - J_lin.transpose() * J_bar.transpose();

    Eigen::VectorXd q_err = q_rest_ - q;
    Eigen::VectorXd q_null_ddot = N_transpose * (Kp_null_ * q_err - Kd_null_ * qd);
    q_ddot_des += q_null_ddot;

    // --- Computed torque: τ = M·q̈_des + C·q̇ + g ---
    Eigen::VectorXd tau = M * q_ddot_des + nle;

    return tau;
  }

  // ========================================================================
  // Members
  // ========================================================================
  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;
  std::unique_ptr<pinocchio::Data> pin_data_bias_;   // for J̇·q̇ computation
  std::unique_ptr<pinocchio::Data> pin_data_desired_;
  mjModel* mj_model_ = nullptr;

  std::vector<std::string> pin_joint_names_;
  std::vector<int> pin_q_idx_, pin_v_idx_;
  std::vector<int> mj_qpos_adr_, mj_dof_adr_;
  std::vector<int> mj_actuator_of_pin_v_;
  std::vector<bool> pin_v_is_arm_;
  std::vector<double> last_saturation_log_time_;

  int left_ee_frame_id_ = -1, right_ee_frame_id_ = -1;
  int left_upper_frame_id_ = -1, left_forearm_frame_id_ = -1;

  Eigen::Vector3d Kp_task_, Kd_task_;
  double dt_ = 0.001;
  double Kp_null_, Kd_null_;
  double Kq_posture_, Dq_posture_;
  double replay_Kp_ = 120.0;
  double replay_Kd_ = 25.0;
  double ilc_Lp_ = 15.0;
  double ilc_Ld_ = 3.0;
  double ilc_learning_rate_ = 0.35;
  double ilc_max_delta_qdd_ = 8.0;
  double ilc_max_qdd_ff_ = 80.0;
  double max_desired_qdd_ = 6.0;
  double m_des_ = 1.0;           // desired task-space mass (mass shaping)

  // Figure-8
  double fig8_freq_, fig8_amp_x_, fig8_amp_z_;
  std::string fig8_plane_;

  // L-shape / trajectory
  bool trajectory_initialized_ = false;
  double start_sim_time_ = 0.0;
  Eigen::VectorXd q_rest_;
  Eigen::Vector3d l_shape_left_ee_, l_shape_right_ee_;
  Eigen::Vector3d l_shape_left_upper_, l_shape_left_forearm_;
  Eigen::Vector3d current_left_target_, current_right_target_;

  // Last state
  Eigen::VectorXd last_q_, last_qd_, last_qdd_, last_tau_;
  Eigen::VectorXd desired_q_, desired_qd_, desired_qdd_;

  // Recorded real joint trajectory and replay metrics
  std::vector<JointSample> recorded_trajectory_;
  std::vector<Eigen::VectorXd> ilc_qdd_ff_;
  std::vector<Eigen::VectorXd> ilc_qdd_delta_sum_;
  std::vector<double> ilc_weight_sum_;
  bool recording_complete_ = false;
  bool replay_started_ = false;
  double replay_cycle_duration_ = 0.0;
  double replay_start_time_ = 0.0;
  int replay_cycle_index_ = 0;
  int metric_samples_ = 0;
  double metric_q_sq_sum_ = 0.0;
  double metric_qd_sq_sum_ = 0.0;
  double metric_ee_sq_sum_ = 0.0;
  double metric_q_max_ = 0.0;
  double metric_qd_max_ = 0.0;
  double metric_ee_max_ = 0.0;
};

// ============================================================================
// Global state (shared between physics thread and main thread)
// ============================================================================
static mjModel* g_m = nullptr;
static mjData*  g_d = nullptr;
static std::unique_ptr<ImpedanceController> g_ctrl;

static std::shared_ptr<rclcpp::Node> g_ros_node;
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr g_joint_pub;
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr g_joint_actual_pub;
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr g_joint_desired_pub;
static rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr g_joint_actual_kinematics_pub;
static rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr g_joint_desired_kinematics_pub;
static std::unique_ptr<tf2_ros::TransformBroadcaster> g_tf_broadcaster;
static rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr g_debug_pub;

static rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr g_left_actual_pub;
static rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr g_left_desired_pub;
static rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr g_right_actual_pub;
static rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr g_right_desired_pub;

// ============================================================================
// Physics loop (runs in background thread)
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
      sim.droploadrequest.store(false);
      sim.LoadMessageClear();
    }
    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      sim.LoadMessageClear();
    }

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
          // Paused: forward kinematics for rendering + joint sliders
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
// Convert Eigen to PoseStamped
// ============================================================================
static geometry_msgs::msg::PoseStamped make_pose_stamped(
    const Eigen::Vector3d& pos, const Eigen::Matrix3d& rot)
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.frame_id = "world";
  msg.pose.position.x = pos.x();
  msg.pose.position.y = pos.y();
  msg.pose.position.z = pos.z();
  Eigen::Quaterniond q_rot(rot);
  msg.pose.orientation.w = q_rot.w();
  msg.pose.orientation.x = q_rot.x();
  msg.pose.orientation.y = q_rot.y();
  msg.pose.orientation.z = q_rot.z();
  return msg;
}

// ============================================================================
// ROS2 publishing (called from main thread via wall timer)
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
  const auto& pin_v_idx    = g_ctrl->pin_v_idx();
  size_t n = joint_names.size();
  const auto stamp = g_ros_node->get_clock()->now();

  // Actual joint states: also publish to /joint_states for standard tools.
  {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = stamp;
    msg.name.resize(n);
    msg.position.resize(n);
    msg.velocity.resize(n);
    for (size_t i = 0; i < n; ++i) {
      msg.name[i]     = joint_names[i];
      msg.position[i] = g_d->qpos[qpos_adr[i]];
      msg.velocity[i] = g_d->qvel[dof_adr[i]];
    }
    g_joint_pub->publish(msg);
    g_joint_actual_pub->publish(msg);
  }

  // Desired joint states from damped least-squares IK on the desired EE path.
  // The order matches /joint_states.name.
  {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = stamp;
    msg.name.resize(n);
    msg.position.resize(n, 0.0);
    msg.velocity.resize(n, 0.0);

    const auto& q_des = g_ctrl->desired_q();
    const auto& qd_des = g_ctrl->desired_qd();
    const auto& pin_q_idx = g_ctrl->pin_q_idx();
    for (size_t i = 0; i < n; ++i) {
      msg.name[i] = joint_names[i];
      const int pin_q = pin_q_idx[i];
      const int pin_v = pin_v_idx[i];
      if (pin_q >= 0 && pin_q < q_des.size()) {
        msg.position[i] = q_des[pin_q];
      }
      if (pin_v >= 0 && pin_v < qd_des.size()) {
        msg.velocity[i] = qd_des[pin_v];
      }
    }
    g_joint_desired_pub->publish(msg);
  }

  // Actual joint kinematics for PlotJuggler:
  // data = [q0, qd0, qdd0, q1, qd1, qdd1, ...] following joint_states.name.
  {
    auto msg = std_msgs::msg::Float64MultiArray();
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "joint";
    msg.layout.dim[0].size = n;
    msg.layout.dim[0].stride = n * 3;
    msg.layout.dim[1].label = "q_qd_qdd";
    msg.layout.dim[1].size = 3;
    msg.layout.dim[1].stride = 3;
    msg.data.resize(n * 3, 0.0);

    for (size_t i = 0; i < n; ++i) {
      msg.data[3 * i + 0] = g_d->qpos[qpos_adr[i]];
      msg.data[3 * i + 1] = g_d->qvel[dof_adr[i]];
      msg.data[3 * i + 2] = g_d->qacc[dof_adr[i]];
    }
    g_joint_actual_kinematics_pub->publish(msg);
  }

  // Desired joint kinematics for PlotJuggler:
  // data = [q_des0, qd_des0, qdd_des0, ...] following joint_states.name.
  {
    auto msg = std_msgs::msg::Float64MultiArray();
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "joint";
    msg.layout.dim[0].size = n;
    msg.layout.dim[0].stride = n * 3;
    msg.layout.dim[1].label = "q_qd_qdd";
    msg.layout.dim[1].size = 3;
    msg.layout.dim[1].stride = 3;
    msg.data.resize(n * 3, 0.0);

    const auto& q_des = g_ctrl->desired_q();
    const auto& qd_des = g_ctrl->desired_qd();
    const auto& qdd_des = g_ctrl->desired_qdd();
    const auto& pin_q_idx = g_ctrl->pin_q_idx();
    for (size_t i = 0; i < n; ++i) {
      const int pin_q = pin_q_idx[i];
      const int pin_v = pin_v_idx[i];
      if (pin_q >= 0 && pin_q < q_des.size()) {
        msg.data[3 * i + 0] = q_des[pin_q];
      }
      if (pin_v >= 0 && pin_v < qd_des.size()) {
        msg.data[3 * i + 1] = qd_des[pin_v];
      }
      if (pin_v >= 0 && pin_v < qdd_des.size()) {
        msg.data[3 * i + 2] = qdd_des[pin_v];
      }
    }
    g_joint_desired_kinematics_pub->publish(msg);
  }

  // TF frames (EE frames only)
  {
    const auto& pin_model = g_ctrl->pin_model();
    const auto& pin_data  = g_ctrl->pin_data();
    const auto stamp = g_ros_node->get_clock()->now();

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

  // ==========================================================================
  // EE pose topics for PlotJuggler
  // ==========================================================================
  {
    const auto& pin_data = g_ctrl->pin_data();
    auto stamp = g_ros_node->get_clock()->now();

    // Left actual
    {
      const auto& oMf = pin_data.oMf[g_ctrl->left_ee_frame_id()];
      auto msg = make_pose_stamped(oMf.translation(), oMf.rotation());
      msg.header.stamp = stamp;
      g_left_actual_pub->publish(msg);
    }

    // Left desired (figure-8)
    {
      const auto& oMf = pin_data.oMf[g_ctrl->left_ee_frame_id()];
      auto msg = make_pose_stamped(g_ctrl->current_left_target(), oMf.rotation());
      msg.header.stamp = stamp;
      g_left_desired_pub->publish(msg);
    }

    // Right actual
    {
      const auto& oMf = pin_data.oMf[g_ctrl->right_ee_frame_id()];
      auto msg = make_pose_stamped(oMf.translation(), oMf.rotation());
      msg.header.stamp = stamp;
      g_right_actual_pub->publish(msg);
    }

    // Right desired (static L-shape)
    {
      const auto& oMf = pin_data.oMf[g_ctrl->right_ee_frame_id()];
      auto msg = make_pose_stamped(g_ctrl->current_right_target(), oMf.rotation());
      msg.header.stamp = stamp;
      g_right_desired_pub->publish(msg);
    }
  }

  // Debug info (~10 Hz within the 60 Hz throttle)
  {
    static auto last_debug = Clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_debug).count() >= 100) {
      last_debug = now;
      auto msg = std_msgs::msg::Float64MultiArray();
      const auto& pin_data = g_ctrl->pin_data();

      Eigen::Vector3d left_pos  = pin_data.oMf[g_ctrl->left_ee_frame_id()].translation();
      Eigen::Vector3d right_pos = pin_data.oMf[g_ctrl->right_ee_frame_id()].translation();
      Eigen::Vector3d left_err  = g_ctrl->current_left_target()  - left_pos;
      Eigen::Vector3d right_err = g_ctrl->current_right_target() - right_pos;

      msg.data = {
        left_pos.x(),  left_pos.y(),  left_pos.z(),
        left_err.x(),  left_err.y(),  left_err.z(),
        right_pos.x(), right_pos.y(), right_pos.z(),
        right_err.x(), right_err.y(), right_err.z(),
      };

      const auto& tau = g_ctrl->last_tau();
      int nv = tau.size();
      double left_rms = 0, right_rms = 0;
      for (int i = 0; i < nv/2; ++i) left_rms += tau[i]*tau[i];
      for (int i = nv/2; i < nv; ++i) right_rms += tau[i]*tau[i];
      left_rms  = std::sqrt(left_rms  / (nv/2));
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
    }

    if (g_d) {
      g_ctrl->init_mujoco_mappings(g_m);

      // Let UI/Simulate take ownership of model and data first
      sim->Load(g_m, g_d, filename);

      {
        const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

        // Load home keyframe AFTER sim->Load to avoid being overwritten
        int home_id = mj_name2id(g_m, mjOBJ_KEY, "home");
        if (home_id >= 0) {
          mj_resetDataKeyframe(g_m, g_d, home_id);
          printf("[INIT] Loaded home keyframe AFTER sim->Load\n");
        } else {
          mj_resetData(g_m, g_d);
          printf("[INIT] home keyframe not found, reset data\n");
        }

        mj_forward(g_m, g_d);
      }

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
  g_ros_node = std::make_shared<rclcpp::Node>("replay_yz_pin_node");

  // --- Parameters ---
  const char* home = std::getenv("HOME");
  if (!home) { fprintf(stderr, "HOME not set\n"); return 1; }

  g_ros_node->declare_parameter("mjcf_path",
      std::string(home) + "/my_work_pkg/src/robo_description/model/marvin_pro_mink_with_gripper.xml");
  g_ros_node->declare_parameter("urdf_path",
      std::string(home) + "/Marvin_Description-Robot_Description/urdf/marvin_pro/marvin_robot.urdf");

  // Task-space impedance gains — per-axis: x not too strong, z stronger
  // so that height-keeping dominates horizontal tracking
  g_ros_node->declare_parameter("Kp_x", 2500.0);
  g_ros_node->declare_parameter("Kp_y", 900.0);
  g_ros_node->declare_parameter("Kp_z", 1200.0);
  g_ros_node->declare_parameter("Kd_x", 350.0);
  g_ros_node->declare_parameter("Kd_y", 180.0);
  g_ros_node->declare_parameter("Kd_z", 240.0);
  g_ros_node->declare_parameter("null_Kp", 5.0);
  g_ros_node->declare_parameter("null_Kd", 3.0);
  g_ros_node->declare_parameter("Kq_posture", 300.0);
  g_ros_node->declare_parameter("Dq_posture", 45.0);
  g_ros_node->declare_parameter("replay_Kp", 120.0);
  g_ros_node->declare_parameter("replay_Kd", 25.0);
  g_ros_node->declare_parameter("ilc_Lp", 15.0);
  g_ros_node->declare_parameter("ilc_Ld", 3.0);
  g_ros_node->declare_parameter("ilc_learning_rate", 0.35);
  g_ros_node->declare_parameter("ilc_max_delta_qdd", 8.0);
  g_ros_node->declare_parameter("ilc_max_qdd_ff", 80.0);
  g_ros_node->declare_parameter("max_desired_qdd", 6.0);
  g_ros_node->declare_parameter("m_des", 5.0);    // desired mass (mass shaping)

  // Figure-8 (small, slow — first get control stable, then scale up)
  g_ros_node->declare_parameter("figure8_frequency", 0.25);
  g_ros_node->declare_parameter("figure8_amplitude_x", 0.08);
  g_ros_node->declare_parameter("figure8_amplitude_z", 0.06);
  g_ros_node->declare_parameter("figure8_plane", "yz");

  // Read back
  std::string mjcf_path = g_ros_node->get_parameter("mjcf_path").as_string();
  std::string urdf_path = g_ros_node->get_parameter("urdf_path").as_string();

  Eigen::Vector3d Kp_task;
  Kp_task << g_ros_node->get_parameter("Kp_x").as_double(),
             g_ros_node->get_parameter("Kp_y").as_double(),
             g_ros_node->get_parameter("Kp_z").as_double();
  Eigen::Vector3d Kd_task;
  Kd_task << g_ros_node->get_parameter("Kd_x").as_double(),
             g_ros_node->get_parameter("Kd_y").as_double(),
             g_ros_node->get_parameter("Kd_z").as_double();
  double Kp_null   = g_ros_node->get_parameter("null_Kp").as_double();
  double Kd_null   = g_ros_node->get_parameter("null_Kd").as_double();
  double Kq_posture = g_ros_node->get_parameter("Kq_posture").as_double();
  double Dq_posture = g_ros_node->get_parameter("Dq_posture").as_double();
  double replay_Kp = g_ros_node->get_parameter("replay_Kp").as_double();
  double replay_Kd = g_ros_node->get_parameter("replay_Kd").as_double();
  double ilc_Lp = g_ros_node->get_parameter("ilc_Lp").as_double();
  double ilc_Ld = g_ros_node->get_parameter("ilc_Ld").as_double();
  double ilc_learning_rate = g_ros_node->get_parameter("ilc_learning_rate").as_double();
  double ilc_max_delta_qdd = g_ros_node->get_parameter("ilc_max_delta_qdd").as_double();
  double ilc_max_qdd_ff = g_ros_node->get_parameter("ilc_max_qdd_ff").as_double();
  double max_desired_qdd = g_ros_node->get_parameter("max_desired_qdd").as_double();
  double m_des     = g_ros_node->get_parameter("m_des").as_double();

  double fig8_freq   = g_ros_node->get_parameter("figure8_frequency").as_double();
  double fig8_amp_x  = g_ros_node->get_parameter("figure8_amplitude_x").as_double();
  double fig8_amp_z  = g_ros_node->get_parameter("figure8_amplitude_z").as_double();
  std::string fig8_plane = g_ros_node->get_parameter("figure8_plane").as_string();

  // Allow command-line override, but skip ROS2 internal args (they start with -- or -)
  if (argc > 1 && argv[1][0] != '-') mjcf_path = argv[1];
  if (argc > 2 && argv[2][0] != '-') {
    fprintf(stderr, "[WARN] replay_yz_pin ignores the second model argument; Pinocchio also uses MJCF.\n");
  }

  printf("MJCF:  %s\n", mjcf_path.c_str());
  printf("Pinocchio model source: MJCF\n");
  printf("Impedance: Kp_task=[%.0f,%.0f,%.0f] Kd_task=[%.0f,%.0f,%.0f] null_Kp=%.1f null_Kd=%.1f Kq_posture=%.1f Dq_posture=%.1f m_des=%.1f\n",
         Kp_task.x(), Kp_task.y(), Kp_task.z(),
         Kd_task.x(), Kd_task.y(), Kd_task.z(),
         Kp_null, Kd_null, Kq_posture, Dq_posture, m_des);
  printf("Replay joint tracking: Kp=%.1f Kd=%.1f max_desired_qdd=%.2f\n",
         replay_Kp, replay_Kd, max_desired_qdd);
  printf("ILC: Lp=%.2f Ld=%.2f learning_rate=%.2f max_delta_qdd=%.2f max_qdd_ff=%.2f\n",
         ilc_Lp, ilc_Ld, ilc_learning_rate, ilc_max_delta_qdd, ilc_max_qdd_ff);
  printf("Figure-8: freq=%.2f Hz  amp_first=%.2f m  amp_second=%.2f m  plane=%s\n",
         fig8_freq, fig8_amp_x, fig8_amp_z, fig8_plane.c_str());

  // --- Create impedance controller ---
  g_ctrl = std::make_unique<ImpedanceController>(
      mjcf_path, Kp_task, Kd_task, Kp_null, Kd_null, Kq_posture, Dq_posture,
      replay_Kp, replay_Kd,
      ilc_Lp, ilc_Ld, ilc_learning_rate, ilc_max_delta_qdd, ilc_max_qdd_ff,
      max_desired_qdd,
      m_des,
      "left_tool", "right_tool",
      fig8_freq, fig8_amp_x, fig8_amp_z, fig8_plane);

  // --- ROS2 publishers ---
  g_joint_pub = g_ros_node->create_publisher<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::QoS(10).reliable());
  g_joint_actual_pub = g_ros_node->create_publisher<sensor_msgs::msg::JointState>(
      "~/joint_actual", rclcpp::QoS(10).reliable());
  g_joint_desired_pub = g_ros_node->create_publisher<sensor_msgs::msg::JointState>(
      "~/joint_desired", rclcpp::QoS(10).reliable());
  g_joint_actual_kinematics_pub = g_ros_node->create_publisher<std_msgs::msg::Float64MultiArray>(
      "~/joint_actual_kinematics", rclcpp::QoS(10).reliable());
  g_joint_desired_kinematics_pub = g_ros_node->create_publisher<std_msgs::msg::Float64MultiArray>(
      "~/joint_desired_kinematics", rclcpp::QoS(10).reliable());
  g_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*g_ros_node);
  g_debug_pub = g_ros_node->create_publisher<std_msgs::msg::Float64MultiArray>(
      "~/debug", rclcpp::QoS(10));

  g_left_actual_pub = g_ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/left_ee_actual", rclcpp::QoS(10).reliable());
  g_left_desired_pub = g_ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/left_ee_desired", rclcpp::QoS(10).reliable());
  g_right_actual_pub = g_ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/right_ee_actual", rclcpp::QoS(10).reliable());
  g_right_desired_pub = g_ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/right_ee_desired", rclcpp::QoS(10).reliable());

  // Publish robot_description for RViz
  {
    std::ifstream urdf_file(urdf_path);
    if (urdf_file) {
      std::stringstream buf;
      buf << urdf_file.rdbuf();
      std::string urdf_text = buf.str();

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

  // --- ROS2 publishing via wall timer ---
  auto ros_timer = g_ros_node->create_wall_timer(
      std::chrono::milliseconds(16),
      [&sim]() { publish_ros2(*sim); });

  std::thread ros_spin_thread([]() {
    rclcpp::spin(g_ros_node);
  });


  // Blocking — MuJoCo full UI
  sim->RenderLoop();

  // Shutdown
  sim->exitrequest.store(1);
  rclcpp::shutdown();
  physics_thread.join();
  if (ros_spin_thread.joinable()) ros_spin_thread.join();

  return 0;
}
