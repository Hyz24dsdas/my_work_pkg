// Copyright 2024
//
// ROS2 + Pinocchio + MuJoCo robot visualization node.

#include <chrono>
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
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

struct SimulatorContext
{
  mjModel*   mj_model = nullptr;
  mjData*    mj_data  = nullptr;
  mjvCamera  cam;
  mjvOption  opt;
  mjvScene   scn;
  mjrContext con;

  double last_x = 0, last_y = 0;
  bool   button_left = false, button_right = false, button_middle = false;
  bool   paused = false;

  SimulatorContext()
  {
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
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
    break;
  case GLFW_KEY_SPACE:
    ctx->paused = !ctx->paused;
    break;
  }
}

static void mouse_button(GLFWwindow* w, int button, int act, int /*mods*/)
{
  auto* ctx = static_cast<SimulatorContext*>(glfwGetWindowUserPointer(w));

  // query current modifier state
  bool alt = (glfwGetKey(w, GLFW_KEY_LEFT_ALT)    == GLFW_PRESS ||
              glfwGetKey(w, GLFW_KEY_RIGHT_ALT)   == GLFW_PRESS);

  bool pressed = (act == GLFW_PRESS);

  // map GLFW button, swapping left/right when Alt is held
  if (button == GLFW_MOUSE_BUTTON_LEFT)
    (alt ? ctx->button_right : ctx->button_left) = pressed;
  else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    (alt ? ctx->button_left : ctx->button_right) = pressed;
  else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    ctx->button_middle = pressed;

  // record cursor position on press
  if (pressed)
    glfwGetCursorPos(w, &ctx->last_x, &ctx->last_y);
}

static void mouse_move(GLFWwindow* w, double xpos, double ypos)
{
  auto* ctx = static_cast<SimulatorContext*>(glfwGetWindowUserPointer(w));
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
// RobotSimulatorNode
// ============================================================================
class RobotSimulatorNode : public rclcpp::Node
{
public:
  explicit RobotSimulatorNode(SimulatorContext* ctx)
  : Node("robot_simulator"), ctx_(ctx)
  {
    declare_parameter("urdf_path", "");
    declare_parameter("mjcf_path", "");

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

    // Pinocchio
    RCLCPP_INFO(get_logger(), "Loading URDF: %s", urdf_path.c_str());
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

    // MuJoCo
    RCLCPP_INFO(get_logger(), "Loading MJCF: %s", mjcf_path.c_str());
    char error[1000] = "";
    ctx_->mj_model = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
    if (!ctx_->mj_model) {
      RCLCPP_FATAL(get_logger(), "MuJoCo load failed: %s", error);
      throw std::runtime_error("MuJoCo load failed");
    }
    ctx_->mj_data = mj_makeData(ctx_->mj_model);

    build_joint_map();

    // --- Publish robot_description for RViz ---
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
          result += "filename=\"" + abs_path.string() + "\"";
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

    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::QoS(10).reliable());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(get_logger(), "Ready: %zu DOFs, %d Pinocchio joints, %ld MuJoCo joints",
                pin_joint_names_.size(),
                static_cast<int>(pin_model_.njoints - 1),
                static_cast<long>(ctx_->mj_model->njnt));
  }

  void sync_and_publish()
  {
    Eigen::VectorXd q(pin_model_.nq);
    q.setZero();
    for (size_t i = 0; i < pin_joint_names_.size(); ++i)
      q[pin_q_idx_[i]] = ctx_->mj_data->qpos[mj_qpos_adr_[i]];

    pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
    publish_joint_states();
    publish_tf_frames();
  }

private:
  void build_joint_map()
  {
    pin_joint_names_.clear();
    pin_q_idx_.clear();
    mj_qpos_adr_.clear();

    for (int j = 1; j < pin_model_.njoints; ++j) {
      int j_nq = pin_model_.joints[j].nq();
      if (j_nq < 1) continue;

      int idx_q = pin_model_.joints[j].idx_q();
      const std::string& name = pin_model_.names[j];

      int mj_id = mj_name2id(ctx_->mj_model, mjOBJ_JOINT, name.c_str());
      if (mj_id < 0) {
        RCLCPP_WARN(get_logger(), "Joint '%s' not in MuJoCo model, skipping", name.c_str());
        continue;
      }
      int mj_adr = ctx_->mj_model->jnt_qposadr[mj_id];
      if (mj_adr < 0) continue;

      for (int k = 0; k < j_nq; ++k) {
        pin_joint_names_.push_back(name);
        pin_q_idx_.push_back(idx_q + k);
        mj_qpos_adr_.push_back(mj_adr + k);
      }
    }
  }

  void publish_joint_states()
  {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = get_clock()->now();
    const size_t n = pin_joint_names_.size();
    msg.name.resize(n);
    msg.position.resize(n);
    msg.velocity.resize(n);

    for (size_t i = 0; i < n; ++i) {
      msg.position[i] = ctx_->mj_data->qpos[mj_qpos_adr_[i]];
      int mj_id = mj_name2id(ctx_->mj_model, mjOBJ_JOINT, pin_joint_names_[i].c_str());
      msg.velocity[i] = ctx_->mj_data->qvel[ctx_->mj_model->jnt_dofadr[mj_id]];
    }
    joint_state_pub_->publish(msg);
  }

  void publish_tf_frames()
  {
    const auto stamp = get_clock()->now();

    for (int f = 0; f < pin_model_.nframes; ++f) {
      const auto& frame = pin_model_.frames[f];
      const auto& oMf   = pin_data_->oMf[f];

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.child_frame_id = frame.name;

      int parent = frame.previousFrame;
      if (parent >= 0) {
        auto rel = pin_data_->oMf[parent].inverse() * oMf;
        tf.header.frame_id = pin_model_.frames[parent].name;
        tf.transform.translation.x = rel.translation().x();
        tf.transform.translation.y = rel.translation().y();
        tf.transform.translation.z = rel.translation().z();
        Eigen::Quaterniond q(rel.rotation());
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
      } else {
        tf.header.frame_id = "world";
        tf.transform.translation.x = oMf.translation()[0];
        tf.transform.translation.y = oMf.translation()[1];
        tf.transform.translation.z = oMf.translation()[2];
        Eigen::Quaterniond q(oMf.rotation());
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
      }
      tf_broadcaster_->sendTransform(tf);
    }
  }

  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;

  std::vector<std::string> pin_joint_names_;
  std::vector<int>         pin_q_idx_;
  std::vector<int>         mj_qpos_adr_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_desc_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  SimulatorContext* ctx_;
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
    auto node = std::make_shared<RobotSimulatorNode>(ctx.get());

    if (!glfwInit()) {
      RCLCPP_FATAL(node->get_logger(), "GLFW init failed");
      rclcpp::shutdown();
      return 1;
    }
    window = glfwCreateWindow(1400, 900,
        "Marvin Pro Mink — MuJoCo + Pinocchio + ROS2", nullptr, nullptr);
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

    RCLCPP_INFO(node->get_logger(), "Entering render loop...");

    while (!glfwWindowShouldClose(window)) {
      if (!ctx->paused)
        mj_step(ctx->mj_model, ctx->mj_data);

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
    RCLCPP_FATAL_STREAM(rclcpp::get_logger("robot_simulator"),
        "Fatal: " << e.what());
  } catch (...) {
    RCLCPP_FATAL(rclcpp::get_logger("robot_simulator"), "Fatal: unknown exception");
  }

  mjr_freeContext(&ctx->con);
  mjv_freeScene(&ctx->scn);
  if (window) glfwDestroyWindow(window);
  glfwTerminate();
  rclcpp::shutdown();
  return 0;
}
