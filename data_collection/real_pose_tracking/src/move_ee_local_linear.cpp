#include <yaml-cpp/yaml.h>

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;
constexpr double kPi = 3.14159265358979323846;

struct Config {
  std::string robot_ip;
  std::string pose_source_field = "O_T_EE";
  double connect_timeout_s = 10.0;
  std::array<double, 3> direction_ee_xyz{1.0, 0.0, 0.0};
  double distance_m = 0.02;
  double speed_mps = 0.01;
  double accel_time_s = 0.5;
};

struct Matrix4 {
  std::array<double, 16> values{};

  double& operator()(int row, int col) {
    return values[static_cast<std::size_t>(row + 4 * col)];
  }

  double operator()(int row, int col) const {
    return values[static_cast<std::size_t>(row + 4 * col)];
  }
};

void HandleSignal(int) {
  g_stop_requested = 1;
}

Config LoadConfig(const std::filesystem::path& config_path) {
  const YAML::Node root = YAML::LoadFile(config_path.string());
  Config config;

  config.robot_ip = root["robot"]["ip"].as<std::string>();
  config.pose_source_field = root["robot"]["pose_source_field"].as<std::string>("O_T_EE");
  config.connect_timeout_s = root["robot"]["connect_timeout_s"].as<double>(10.0);

  const YAML::Node motion = root["motion"];
  if (!motion || !motion.IsMap()) {
    throw std::runtime_error("Missing required 'motion' section in config.");
  }

  const YAML::Node direction = motion["direction_ee_xyz"];
  if (!direction || !direction.IsSequence() || direction.size() != 3) {
    throw std::runtime_error("motion.direction_ee_xyz must be a 3-element sequence.");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    config.direction_ee_xyz[i] = direction[i].as<double>();
  }

  config.distance_m = motion["distance_m"].as<double>(0.02);
  config.speed_mps = motion["speed_mps"].as<double>(0.01);
  config.accel_time_s = motion["accel_time_s"].as<double>(0.5);
  return config;
}

std::unique_ptr<franka::Robot> ConnectRobotWithTimeout(
    const std::string& robot_ip,
    double timeout_s) {
  if (timeout_s <= 0.0) {
    throw std::runtime_error("robot.connect_timeout_s must be positive.");
  }

  std::promise<std::unique_ptr<franka::Robot>> promise;
  std::future<std::unique_ptr<franka::Robot>> future = promise.get_future();

  std::thread worker([robot_ip, promise = std::move(promise)]() mutable {
    try {
      auto robot = std::make_unique<franka::Robot>(robot_ip);
      promise.set_value(std::move(robot));
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  const auto wait_status = future.wait_for(std::chrono::duration<double>(timeout_s));
  if (wait_status == std::future_status::ready) {
    worker.join();
    return future.get();
  }

  worker.detach();
  throw std::runtime_error(
      "Timed out while connecting to Franka robot at " + robot_ip + " after " +
      std::to_string(timeout_s) + " seconds.");
}

Matrix4 TransformFromState(const franka::RobotState& state, const std::string& pose_source_field) {
  Matrix4 transform{};
  const std::array<double, 16>* raw = nullptr;

  if (pose_source_field == "O_T_EE") {
    raw = &state.O_T_EE;
  } else if (pose_source_field == "O_T_EE_c") {
    raw = &state.O_T_EE_c;
  } else {
    throw std::runtime_error("Unsupported pose_source_field: " + pose_source_field);
  }

  transform.values = *raw;
  return transform;
}

double VectorNorm(const std::array<double, 3>& vector) {
  return std::sqrt(
      vector[0] * vector[0] +
      vector[1] * vector[1] +
      vector[2] * vector[2]);
}

std::array<double, 3> NormalizeVector(const std::array<double, 3>& vector) {
  const double norm = VectorNorm(vector);
  if (norm <= 1e-12) {
    throw std::runtime_error("motion.direction_ee_xyz must not be the zero vector.");
  }
  return {vector[0] / norm, vector[1] / norm, vector[2] / norm};
}

std::array<double, 3> RotateLocalDirectionToBase(
    const Matrix4& transform_base_ee,
    const std::array<double, 3>& direction_ee_xyz) {
  std::array<double, 3> direction_base_xyz{};
  for (int row = 0; row < 3; ++row) {
    direction_base_xyz[static_cast<std::size_t>(row)] =
        transform_base_ee(row, 0) * direction_ee_xyz[0] +
        transform_base_ee(row, 1) * direction_ee_xyz[1] +
        transform_base_ee(row, 2) * direction_ee_xyz[2];
  }
  return NormalizeVector(direction_base_xyz);
}

double ComputeTravelDistance(
    double time_s,
    double distance_m,
    double speed_mps,
    double accel_time_s) {
  if (time_s <= 0.0) {
    return 0.0;
  }

  const double effective_accel_time_s = std::max(accel_time_s, 1e-6);
  const double threshold_distance_m = speed_mps * effective_accel_time_s;

  double peak_speed_mps = speed_mps;
  double cruise_time_s = 0.0;
  if (distance_m > threshold_distance_m) {
    cruise_time_s = (distance_m - threshold_distance_m) / speed_mps;
  } else {
    peak_speed_mps = distance_m / effective_accel_time_s;
  }

  const double accel_distance_m = 0.5 * peak_speed_mps * effective_accel_time_s;
  const double total_time_s = 2.0 * effective_accel_time_s + cruise_time_s;
  if (time_s >= total_time_s) {
    return distance_m;
  }

  if (time_s < effective_accel_time_s) {
    return 0.5 * peak_speed_mps *
           (time_s - effective_accel_time_s / kPi * std::sin(kPi * time_s / effective_accel_time_s));
  }

  if (time_s < effective_accel_time_s + cruise_time_s) {
    return accel_distance_m + peak_speed_mps * (time_s - effective_accel_time_s);
  }

  const double decel_time_s = time_s - effective_accel_time_s - cruise_time_s;
  return accel_distance_m +
         peak_speed_mps * cruise_time_s +
         0.5 * peak_speed_mps *
             (decel_time_s +
              effective_accel_time_s / kPi * std::sin(kPi * decel_time_s / effective_accel_time_s));
}

double ComputeTotalMotionTime(double distance_m, double speed_mps, double accel_time_s) {
  const double effective_accel_time_s = std::max(accel_time_s, 1e-6);
  const double threshold_distance_m = speed_mps * effective_accel_time_s;
  if (distance_m > threshold_distance_m) {
    return 2.0 * effective_accel_time_s + (distance_m - threshold_distance_m) / speed_mps;
  }
  return 2.0 * effective_accel_time_s;
}

void ValidateConfig(const Config& config) {
  if (config.distance_m <= 0.0) {
    throw std::runtime_error("motion.distance_m must be positive.");
  }
  if (config.speed_mps <= 0.0) {
    throw std::runtime_error("motion.speed_mps must be positive.");
  }
  if (config.accel_time_s <= 0.0) {
    throw std::runtime_error("motion.accel_time_s must be positive.");
  }
  (void)NormalizeVector(config.direction_ee_xyz);
}

std::string ArrayToString(const std::array<double, 3>& values) {
  std::ostringstream oss;
  oss << "[" << std::fixed << std::setprecision(6)
      << values[0] << ", " << values[1] << ", " << values[2] << "]";
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, HandleSignal);

  std::filesystem::path config_path = "config/move_ee_local_linear.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  try {
    const Config config = LoadConfig(config_path);
    ValidateConfig(config);

    auto robot = ConnectRobotWithTimeout(config.robot_ip, config.connect_timeout_s);
    const franka::RobotState initial_state = robot->readOnce();
    const Matrix4 initial_transform_base_ee = TransformFromState(initial_state, config.pose_source_field);
    const std::array<double, 3> direction_ee_xyz = NormalizeVector(config.direction_ee_xyz);
    const std::array<double, 3> direction_base_xyz =
        RotateLocalDirectionToBase(initial_transform_base_ee, direction_ee_xyz);
    const double total_motion_time_s =
        ComputeTotalMotionTime(config.distance_m, config.speed_mps, config.accel_time_s);

    std::cout << "[INFO] Starting local linear motion.\n";
    std::cout << "[INFO] direction_ee_xyz = " << ArrayToString(direction_ee_xyz) << "\n";
    std::cout << "[INFO] direction_base_xyz = " << ArrayToString(direction_base_xyz) << "\n";
    std::cout << "[INFO] distance_m = " << std::fixed << std::setprecision(6) << config.distance_m << "\n";
    std::cout << "[INFO] speed_mps = " << std::fixed << std::setprecision(6) << config.speed_mps << "\n";
    std::cout << "[INFO] accel_time_s = " << std::fixed << std::setprecision(6) << config.accel_time_s << "\n";
    std::cout << "[INFO] estimated_total_time_s = " << std::fixed << std::setprecision(6) << total_motion_time_s << "\n";
    std::cout << "[INFO] Press Ctrl+C to stop.\n";

    double elapsed_time_s = 0.0;
    robot->control(
        [&](const franka::RobotState&, franka::Duration period) -> franka::CartesianPose {
          elapsed_time_s += period.toSec();

          const double unclamped_distance_m = ComputeTravelDistance(
              elapsed_time_s,
              config.distance_m,
              config.speed_mps,
              config.accel_time_s);
          const double commanded_distance_m =
              g_stop_requested ? std::min(unclamped_distance_m, config.distance_m)
                               : std::min(unclamped_distance_m, config.distance_m);

          std::array<double, 16> target_pose = initial_transform_base_ee.values;

          target_pose[12] = initial_transform_base_ee(0, 3) + direction_base_xyz[0] * commanded_distance_m;
          target_pose[13] = initial_transform_base_ee(1, 3) + direction_base_xyz[1] * commanded_distance_m;
          target_pose[14] = initial_transform_base_ee(2, 3) + direction_base_xyz[2] * commanded_distance_m;

          if (g_stop_requested || elapsed_time_s >= total_motion_time_s) {
            std::cout << "[INFO] Motion finished.\n";
            return franka::MotionFinished(target_pose);
          }

          return target_pose;
        },
        franka::ControllerMode::kJointImpedance);

    return 0;
  } catch (const franka::Exception& exc) {
    std::cerr << "[ERROR] libfranka exception: " << exc.what() << "\n";
    return 1;
  } catch (const std::exception& exc) {
    std::cerr << "[ERROR] " << exc.what() << "\n";
    return 1;
  }
}
