// mj动力学补偿+PD关节姿态+z优先级任务空间控制


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

// ros2 launch robo_description sin_control.launch.py \
//   Kp_x:=100.0 \
//   Kp_y:=80.0 \
//   Kp_z:=3000.0 \
//   Kd_x:=40.0 \
//   Kd_y:=25.0 \
//   Kd_z:=600.0 \
//   Kq_posture:=150.0 \
//   Dq_posture:=25.0 \
//   figure8_frequency:=0.1 \
//   figure8_amplitude_x:=0.05 \
//   figure8_amplitude_z:=0.05

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
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

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
                      const Eigen::Vector3d& Kp_task,
                      const Eigen::Vector3d& Kd_task,
                      double Kp_null, double Kd_null,
                      double Kq_posture, double Dq_posture,
                      double m_des,
                      const std::string& left_ee_name,
                      const std::string& right_ee_name,
                      double fig8_freq, double fig8_amp_x, double fig8_amp_z,
                      const std::string& fig8_plane)
    : Kp_task_(Kp_task), Kd_task_(Kd_task)
    , Kp_null_(Kp_null), Kd_null_(Kd_null)
    , Kq_posture_(Kq_posture), Dq_posture_(Dq_posture)
    , m_des_(m_des)
    , fig8_freq_(fig8_freq)
    , fig8_amp_x_(fig8_amp_x)
    , fig8_amp_z_(fig8_amp_z)
    , fig8_plane_(fig8_plane)
  {
    // Load Pinocchio model from URDF
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    pin_data_bias_ = std::make_unique<pinocchio::Data>(pin_model_);

    left_ee_frame_id_  = static_cast<int>(pin_model_.getFrameId(left_ee_name));
    right_ee_frame_id_ = static_cast<int>(pin_model_.getFrameId(right_ee_name));

    printf("Pinocchio model: %d joints, nq=%d, nv=%d\n",
           static_cast<int>(pin_model_.njoints - 1),
           static_cast<int>(pin_model_.nq),
           static_cast<int>(pin_model_.nv));
    printf("Left  EE frame: '%s' id=%d\n", left_ee_name.c_str(),  left_ee_frame_id_);
    printf("Right EE frame: '%s' id=%d\n", right_ee_name.c_str(), right_ee_frame_id_);
    printf("Desired mass (M_des): %.1f kg\n", m_des_);
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

    current_left_target_  = left_target_pos;
    current_right_target_ = right_target_pos;

    // ===== Stable controller: MuJoCo bias + J^T F + joint posture PD =====
    {
      mj_forward(mj_model_, d);

      Eigen::VectorXd tau(pin_model_.nv);
      tau.setZero();

      // 1) MuJoCo native bias compensation
      for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
        int pin_v = pin_v_idx_[i];
        int mj_dof = mj_dof_adr_[i];
        tau[pin_v] = d->qfrc_bias[mj_dof];
      }

      // 2) Cartesian task force: tau += J^T F
      auto add_task = [&](int ee_frame_id,
                          const Eigen::Vector3d& target_pos,
                          const Eigen::Vector3d& target_vel,
                          const Eigen::Vector3d& target_acc)
      {
        const auto& oMf = pin_data_->oMf[ee_frame_id];
        Eigen::Vector3d x = oMf.translation();

        Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(
            pin_model_, *pin_data_, ee_frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED);

        Eigen::MatrixXd J_lin = J6.topRows<3>();
        Eigen::Vector3d x_dot = J_lin * qd;

        Eigen::Vector3d e = target_pos - x;
        Eigen::Vector3d e_dot = target_vel - x_dot;

        Eigen::Vector3d F =
            Kp_task_.cwiseProduct(e)
          + Kd_task_.cwiseProduct(e_dot)
          + m_des_ * target_acc;

        tau += J_lin.transpose() * F;
      };

      // Hierarchical z-priority task for left arm:
      //   1) z (height) gets highest priority with strong gains
      //   2) x, y are projected into the null-space of z so they
      //      never disturb height tracking
      auto add_task_priority_xz = [&](int ee_frame_id,
                                      const Eigen::Vector3d& target_pos,
                                      const Eigen::Vector3d& target_vel,
                                      const Eigen::Vector3d& target_acc)
      {
        const auto& oMf = pin_data_->oMf[ee_frame_id];
        Eigen::Vector3d x = oMf.translation();

        Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(
            pin_model_, *pin_data_, ee_frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED);

        Eigen::MatrixXd J_lin = J6.topRows<3>();

        Eigen::RowVectorXd Jx = J_lin.row(0);
        Eigen::RowVectorXd Jy = J_lin.row(1);
        Eigen::RowVectorXd Jz = J_lin.row(2);

        double x_dot = (Jx * qd)(0);
        double y_dot = (Jy * qd)(0);
        double z_dot = (Jz * qd)(0);

        double ex = target_pos.x() - x.x();
        double ey = target_pos.y() - x.y();
        double ez = target_pos.z() - x.z();

        double ex_dot = target_vel.x() - x_dot;
        double ey_dot = target_vel.y() - y_dot;
        double ez_dot = target_vel.z() - z_dot;

        const int nv = pin_model_.nv;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nv, nv);

        // 1) z high priority
        double Fz =
            Kp_task_.z() * ez
          + Kd_task_.z() * ez_dot
          + m_des_ * target_acc.z();

        tau += Jz.transpose() * Fz;

        // 2) construct z null-space so x/y only move in directions
        //    that don't disturb z
        double lambda = 1e-4;
        Eigen::VectorXd Jz_pinv =
            Jz.transpose() / ((Jz * Jz.transpose())(0, 0) + lambda);

        Eigen::MatrixXd Nz = I - Jz_pinv * Jz;

        // 3) x secondary task
        double Fx =
            Kp_task_.x() * ex
          + Kd_task_.x() * ex_dot
          + m_des_ * target_acc.x();

        tau += Nz.transpose() * Jx.transpose() * Fx;

        // 4) y low stiffness hold
        double Fy =
            Kp_task_.y() * ey
          + Kd_task_.y() * ey_dot
          + m_des_ * target_acc.y();

        tau += Nz.transpose() * Jy.transpose() * Fy;
      };

      add_task_priority_xz(left_ee_frame_id_, left_target_pos, left_target_vel, left_target_acc);
      add_task(right_ee_frame_id_, right_target_pos, right_target_vel, right_target_acc);

      // 3) Joint posture PD
      for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
        const std::string& name = pin_joint_names_[i];

        // 不要控制 gripper finger
        if (name.find("Arm_") != 0) {
          continue;
        }

        int pin_q = pin_q_idx_[i];
        int pin_v = pin_v_idx_[i];

        tau[pin_v] += Kq_posture_ * (q_rest_[pin_q] - q[pin_q])
                    - Dq_posture_ * qd[pin_v];
      }

      apply_torques_to_mujoco(d, tau);

      last_tau_ = tau;
      last_q_ = q;
      last_qd_ = qd;
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

    printf("L-shape targets: L_ee=[%.3f, %.3f, %.3f]  R_ee=[%.3f, %.3f, %.3f]\n",
           l_shape_left_ee_.x(),  l_shape_left_ee_.y(),  l_shape_left_ee_.z(),
           l_shape_right_ee_.x(), l_shape_right_ee_.y(), l_shape_right_ee_.z());
  }

  // ========================================================================
  // Joint & actuator mapping
  // ========================================================================
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

      if (saturated) {
        const char* act_name = mj_id2name(mj_model_, mjOBJ_ACTUATOR, act_id);
        printf("[SAT] actuator=%s tau=%.2f clipped to [%.2f, %.2f]\n",
               act_name ? act_name : "unknown", tau[i], lo, hi);
      }

      d->ctrl[act_id] = cmd;
    }
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
  mjModel* mj_model_ = nullptr;

  std::vector<std::string> pin_joint_names_;
  std::vector<int> pin_q_idx_, pin_v_idx_;
  std::vector<int> mj_qpos_adr_, mj_dof_adr_;
  std::vector<int> mj_actuator_of_pin_v_;

  int left_ee_frame_id_ = -1, right_ee_frame_id_ = -1;

  Eigen::Vector3d Kp_task_, Kd_task_;
  double dt_ = 0.001;
  double Kp_null_, Kd_null_;
  double Kq_posture_, Dq_posture_;
  double m_des_ = 1.0;           // desired task-space mass (mass shaping)

  // Figure-8
  double fig8_freq_, fig8_amp_x_, fig8_amp_z_;
  std::string fig8_plane_;

  // L-shape / trajectory
  bool trajectory_initialized_ = false;
  double start_sim_time_ = 0.0;
  Eigen::VectorXd q_rest_;
  Eigen::Vector3d l_shape_left_ee_, l_shape_right_ee_;
  Eigen::Vector3d current_left_target_, current_right_target_;

  // Last state
  Eigen::VectorXd last_q_, last_qd_, last_tau_;
};

// ============================================================================
// Global state (shared between physics thread and main thread)
// ============================================================================
static mjModel* g_m = nullptr;
static mjData*  g_d = nullptr;
static std::unique_ptr<ImpedanceController> g_ctrl;

static std::shared_ptr<rclcpp::Node> g_ros_node;
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr g_joint_pub;
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
  g_ros_node = std::make_shared<rclcpp::Node>("sin_control_node");

  // --- Parameters ---
  const char* home = std::getenv("HOME");
  if (!home) { fprintf(stderr, "HOME not set\n"); return 1; }

  g_ros_node->declare_parameter("urdf_path",
      std::string(home) + "/Marvin_Description-Robot_Description/urdf/marvin_pro/marvin_robot.urdf");
  g_ros_node->declare_parameter("mjcf_path",
      std::string(home) + "/my_work_pkg/src/robo_description/model/marvin_pro_mink_with_gripper.xml");

  // Task-space impedance gains — per-axis: x not too strong, z stronger
  // so that height-keeping dominates horizontal tracking
  g_ros_node->declare_parameter("Kp_x", 300.0);
  g_ros_node->declare_parameter("Kp_y", 100.0);
  g_ros_node->declare_parameter("Kp_z", 1200.0);
  g_ros_node->declare_parameter("Kd_x", 80.0);
  g_ros_node->declare_parameter("Kd_y", 30.0);
  g_ros_node->declare_parameter("Kd_z", 240.0);
  g_ros_node->declare_parameter("null_Kp", 5.0);
  g_ros_node->declare_parameter("null_Kd", 3.0);
  g_ros_node->declare_parameter("Kq_posture", 150.0);
  g_ros_node->declare_parameter("Dq_posture", 25.0);
  g_ros_node->declare_parameter("m_des", 5.0);    // desired mass (mass shaping)

  // Figure-8 (small, slow — first get control stable, then scale up)
  g_ros_node->declare_parameter("figure8_frequency", 0.5);
  g_ros_node->declare_parameter("figure8_amplitude_x", 0.15);
  g_ros_node->declare_parameter("figure8_amplitude_z", 0.10);
  g_ros_node->declare_parameter("figure8_plane", "xz");

  // Read back
  std::string urdf_path = g_ros_node->get_parameter("urdf_path").as_string();
  std::string mjcf_path = g_ros_node->get_parameter("mjcf_path").as_string();

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
  double m_des     = g_ros_node->get_parameter("m_des").as_double();

  double fig8_freq   = g_ros_node->get_parameter("figure8_frequency").as_double();
  double fig8_amp_x  = g_ros_node->get_parameter("figure8_amplitude_x").as_double();
  double fig8_amp_z  = g_ros_node->get_parameter("figure8_amplitude_z").as_double();
  std::string fig8_plane = g_ros_node->get_parameter("figure8_plane").as_string();

  // Allow command-line override, but skip ROS2 internal args (they start with -- or -)
  if (argc > 1 && argv[1][0] != '-') mjcf_path = argv[1];
  if (argc > 2 && argv[2][0] != '-') urdf_path = argv[2];

  printf("MJCF:  %s\n", mjcf_path.c_str());
  printf("URDF:  %s\n", urdf_path.c_str());
  printf("Impedance: Kp_task=[%.0f,%.0f,%.0f] Kd_task=[%.0f,%.0f,%.0f] null_Kp=%.1f null_Kd=%.1f Kq_posture=%.1f Dq_posture=%.1f m_des=%.1f\n",
         Kp_task.x(), Kp_task.y(), Kp_task.z(),
         Kd_task.x(), Kd_task.y(), Kd_task.z(),
         Kp_null, Kd_null, Kq_posture, Dq_posture, m_des);
  printf("Figure-8: freq=%.2f Hz  amp_x=%.2f m  amp_z=%.2f m  plane=%s\n",
         fig8_freq, fig8_amp_x, fig8_amp_z, fig8_plane.c_str());

  // --- Create impedance controller ---
  g_ctrl = std::make_unique<ImpedanceController>(
      urdf_path, Kp_task, Kd_task, Kp_null, Kd_null, Kq_posture, Dq_posture, m_des,
      "left_tool", "right_tool",
      fig8_freq, fig8_amp_x, fig8_amp_z, fig8_plane);

  // --- ROS2 publishers ---
  g_joint_pub = g_ros_node->create_publisher<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::QoS(10).reliable());
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
