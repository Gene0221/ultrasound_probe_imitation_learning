#include <yaml-cpp/yaml.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

namespace {

using Clock = std::chrono::system_clock;
using SteadyClock = std::chrono::steady_clock;

volatile std::sig_atomic_t g_stop_requested = 0;

struct Config {
  std::string robot_ip;
  std::string pose_source_field;
  double connect_timeout_s = 10.0;
  double target_hz = 30.0;
  std::optional<int> max_samples;
  std::filesystem::path output_root;
  std::string jsonl_file_name = "franka_ee_pose_deltas.jsonl";
  std::string summary_file_name = "summary.json";
  bool write_jsonl = true;
  bool emit_stdout_records = false;
  bool include_absolute_pose = false;
  bool include_pose_source_field = true;
  bool include_robot_ip = true;
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

struct Quaternion {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
};

struct PoseSample {
  int sample_index = 0;
  double host_timestamp_s = 0.0;
  Matrix4 transform_base_ee;
  std::array<double, 3> position_xyz{};
  Quaternion quaternion_xyzw;
  std::string pose_source_field;
};

std::string JsonEscape(const std::string& input) {
  std::ostringstream oss;
  for (char ch : input) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

template <std::size_t N>
std::string ArrayToJson(const std::array<double, N>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < N; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << std::setprecision(15) << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string QuaternionToJson(const Quaternion& q) {
  return ArrayToJson<4>({q.x, q.y, q.z, q.w});
}

std::string Matrix4ToJson(const Matrix4& matrix) {
  std::ostringstream oss;
  oss << "[";
  for (int row = 0; row < 4; ++row) {
    if (row > 0) {
      oss << ", ";
    }
    oss << "[";
    for (int col = 0; col < 4; ++col) {
      if (col > 0) {
        oss << ", ";
      }
      oss << std::setprecision(15) << matrix(row, col);
    }
    oss << "]";
  }
  oss << "]";
  return oss.str();
}

Matrix4 IdentityMatrix4() {
  Matrix4 result;
  result(0, 0) = 1.0;
  result(1, 1) = 1.0;
  result(2, 2) = 1.0;
  result(3, 3) = 1.0;
  return result;
}

Matrix4 Multiply(const Matrix4& a, const Matrix4& b) {
  Matrix4 result{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) {
        sum += a(row, k) * b(k, col);
      }
      result(row, col) = sum;
    }
  }
  return result;
}

Matrix4 InvertTransform(const Matrix4& transform) {
  Matrix4 inverse = IdentityMatrix4();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      inverse(row, col) = transform(col, row);
    }
  }
  for (int row = 0; row < 3; ++row) {
    double value = 0.0;
    for (int col = 0; col < 3; ++col) {
      value -= inverse(row, col) * transform(col, 3);
    }
    inverse(row, 3) = value;
  }
  return inverse;
}

Quaternion NormalizeQuaternion(const Quaternion& q) {
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm == 0.0) {
    throw std::runtime_error("Quaternion norm is zero.");
  }
  return Quaternion{q.x / norm, q.y / norm, q.z / norm, q.w / norm};
}

Quaternion RotationMatrixToQuaternion(const Matrix4& transform) {
  const double trace = transform(0, 0) + transform(1, 1) + transform(2, 2);

  Quaternion q;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (transform(2, 1) - transform(1, 2)) / s;
    q.y = (transform(0, 2) - transform(2, 0)) / s;
    q.z = (transform(1, 0) - transform(0, 1)) / s;
  } else if (transform(0, 0) > transform(1, 1) && transform(0, 0) > transform(2, 2)) {
    const double s = std::sqrt(1.0 + transform(0, 0) - transform(1, 1) - transform(2, 2)) * 2.0;
    q.w = (transform(2, 1) - transform(1, 2)) / s;
    q.x = 0.25 * s;
    q.y = (transform(0, 1) + transform(1, 0)) / s;
    q.z = (transform(0, 2) + transform(2, 0)) / s;
  } else if (transform(1, 1) > transform(2, 2)) {
    const double s = std::sqrt(1.0 + transform(1, 1) - transform(0, 0) - transform(2, 2)) * 2.0;
    q.w = (transform(0, 2) - transform(2, 0)) / s;
    q.x = (transform(0, 1) + transform(1, 0)) / s;
    q.y = 0.25 * s;
    q.z = (transform(1, 2) + transform(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + transform(2, 2) - transform(0, 0) - transform(1, 1)) * 2.0;
    q.w = (transform(1, 0) - transform(0, 1)) / s;
    q.x = (transform(0, 2) + transform(2, 0)) / s;
    q.y = (transform(1, 2) + transform(2, 1)) / s;
    q.z = 0.25 * s;
  }

  return NormalizeQuaternion(q);
}

std::string CurrentIsoTime() {
  const auto now = Clock::now();
  const auto now_time_t = Clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_time_t);
#else
  localtime_r(&now_time_t, &local_tm);
#endif

  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S")
      << "." << std::setw(3) << std::setfill('0') << ms.count();
  return oss.str();
}

double CurrentUnixTimestampSeconds() {
  const auto now = Clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void HandleSignal(int) {
  g_stop_requested = 1;
}

Config LoadConfig(const std::filesystem::path& config_path) {
  const YAML::Node root = YAML::LoadFile(config_path.string());
  Config config;

  config.robot_ip = root["robot"]["ip"].as<std::string>();
  config.pose_source_field = root["robot"]["pose_source_field"].as<std::string>("O_T_EE");
  config.connect_timeout_s = root["robot"]["connect_timeout_s"].as<double>(10.0);
  config.target_hz = root["sampling"]["target_hz"].as<double>(30.0);
  if (root["sampling"]["max_samples"] && !root["sampling"]["max_samples"].IsNull()) {
    config.max_samples = root["sampling"]["max_samples"].as<int>();
  }

  config.output_root = root["output"]["output_root"].as<std::string>("output");
  config.jsonl_file_name = root["output"]["jsonl_file_name"].as<std::string>("franka_ee_pose_deltas.jsonl");
  config.summary_file_name = root["output"]["summary_file_name"].as<std::string>("summary.json");
  config.write_jsonl = root["output"]["write_jsonl"].as<bool>(true);
  config.emit_stdout_records = root["output"]["emit_stdout_records"].as<bool>(false);

  config.include_absolute_pose = root["recording"]["include_absolute_pose"].as<bool>(false);
  config.include_pose_source_field = root["recording"]["include_pose_source_field"].as<bool>(true);
  config.include_robot_ip = root["recording"]["include_robot_ip"].as<bool>(true);

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

PoseSample BuildSample(
    int sample_index,
    double host_timestamp_s,
    const Matrix4& transform_base_ee,
    const std::string& pose_source_field) {
  PoseSample sample;
  sample.sample_index = sample_index;
  sample.host_timestamp_s = host_timestamp_s;
  sample.transform_base_ee = transform_base_ee;
  sample.position_xyz = {transform_base_ee(0, 3), transform_base_ee(1, 3), transform_base_ee(2, 3)};
  sample.quaternion_xyzw = RotationMatrixToQuaternion(transform_base_ee);
  sample.pose_source_field = pose_source_field;
  return sample;
}

void WriteJsonLine(std::ofstream& stream, const std::string& payload) {
  stream << payload << '\n';
  stream.flush();
}

std::string BuildRecordJson(
    const PoseSample& previous_sample,
    const PoseSample& current_sample,
    const Matrix4& delta_transform,
    const std::array<double, 3>& delta_translation,
    const Quaternion& delta_quaternion,
    const Config& config) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"sample_index\": " << current_sample.sample_index << ", ";
  oss << "\"valid\": true, ";
  oss << "\"host_timestamp_s\": " << std::setprecision(15) << current_sample.host_timestamp_s << ", ";
  oss << "\"prev_host_timestamp_s\": " << std::setprecision(15) << previous_sample.host_timestamp_s << ", ";
  oss << "\"curr_host_timestamp_s\": " << std::setprecision(15) << current_sample.host_timestamp_s << ", ";
  oss << "\"delta_transform_prev_to_curr\": " << Matrix4ToJson(delta_transform) << ", ";
  oss << "\"delta_translation_xyz\": " << ArrayToJson<3>(delta_translation) << ", ";
  oss << "\"delta_quaternion_xyzw\": " << QuaternionToJson(delta_quaternion);

  if (config.include_absolute_pose) {
    oss << ", \"prev_position_xyz\": " << ArrayToJson<3>(previous_sample.position_xyz);
    oss << ", \"prev_quaternion_xyzw\": " << QuaternionToJson(previous_sample.quaternion_xyzw);
    oss << ", \"curr_position_xyz\": " << ArrayToJson<3>(current_sample.position_xyz);
    oss << ", \"curr_quaternion_xyzw\": " << QuaternionToJson(current_sample.quaternion_xyzw);
  }

  if (config.include_pose_source_field) {
    oss << ", \"pose_source_field\": \"" << JsonEscape(current_sample.pose_source_field) << "\"";
  }

  if (config.include_robot_ip) {
    oss << ", \"robot_ip\": \"" << JsonEscape(config.robot_ip) << "\"";
  }

  oss << "}";
  return oss.str();
}

std::string BuildSummaryJson(
    const Config& config,
    const std::filesystem::path& output_root,
    const std::filesystem::path& jsonl_path,
    int records_logged,
    int samples_read,
    std::optional<double> first_timestamp_s,
    std::optional<double> last_timestamp_s) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"output_root\": \"" << JsonEscape(output_root.string()) << "\",\n";
  oss << "  \"jsonl_path\": \"" << JsonEscape(jsonl_path.string()) << "\",\n";
  oss << "  \"robot_ip\": \"" << JsonEscape(config.robot_ip) << "\",\n";
  oss << "  \"pose_source_field\": \"" << JsonEscape(config.pose_source_field) << "\",\n";
  oss << "  \"target_hz\": " << std::setprecision(15) << config.target_hz << ",\n";
  oss << "  \"records_logged\": " << records_logged << ",\n";
  oss << "  \"samples_read\": " << samples_read << ",\n";
  oss << "  \"first_host_timestamp_s\": "
      << (first_timestamp_s.has_value() ? std::to_string(*first_timestamp_s) : "null") << ",\n";
  oss << "  \"last_host_timestamp_s\": "
      << (last_timestamp_s.has_value() ? std::to_string(*last_timestamp_s) : "null") << "\n";
  oss << "}\n";
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, HandleSignal);

  std::filesystem::path config_path = "config/default.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  try {
    const Config config = LoadConfig(config_path);
    if (config.target_hz <= 0.0) {
      throw std::runtime_error("sampling.target_hz must be positive.");
    }

    const std::filesystem::path project_root = std::filesystem::absolute(config_path).parent_path().parent_path();
    const std::filesystem::path output_root = std::filesystem::absolute(project_root / config.output_root);
    const std::filesystem::path jsonl_path = output_root / config.jsonl_file_name;
    const std::filesystem::path summary_path = output_root / config.summary_file_name;

    std::filesystem::create_directories(output_root);

    std::optional<std::ofstream> jsonl_stream;
    if (config.write_jsonl) {
      jsonl_stream.emplace(jsonl_path, std::ios::out | std::ios::trunc);
      if (!jsonl_stream->is_open()) {
        throw std::runtime_error("Cannot open JSONL output file: " + jsonl_path.string());
      }
    }

    auto robot = ConnectRobotWithTimeout(config.robot_ip, config.connect_timeout_s);

    const double dt = 1.0 / config.target_hz;
    int sample_index = 0;
    int records_logged = 0;
    std::optional<PoseSample> previous_sample;
    std::optional<double> first_timestamp_s;
    std::optional<double> last_timestamp_s;

    while (!g_stop_requested) {
      const auto loop_start = SteadyClock::now();
      const double host_timestamp_s = CurrentUnixTimestampSeconds();
      const franka::RobotState state = robot->readOnce();
      const Matrix4 transform_base_ee = TransformFromState(state, config.pose_source_field);

      ++sample_index;
      PoseSample current_sample = BuildSample(
          sample_index,
          host_timestamp_s,
          transform_base_ee,
          config.pose_source_field);

      if (!first_timestamp_s.has_value()) {
        first_timestamp_s = current_sample.host_timestamp_s;
      }
      last_timestamp_s = current_sample.host_timestamp_s;

      if (previous_sample.has_value()) {
        const Matrix4 delta_transform = Multiply(
            InvertTransform(previous_sample->transform_base_ee),
            current_sample.transform_base_ee);
        const std::array<double, 3> delta_translation = {
            delta_transform(0, 3),
            delta_transform(1, 3),
            delta_transform(2, 3),
        };
        const Quaternion delta_quaternion = RotationMatrixToQuaternion(delta_transform);

        const std::string record_json = BuildRecordJson(
            *previous_sample,
            current_sample,
            delta_transform,
            delta_translation,
            delta_quaternion,
            config);

        if (jsonl_stream.has_value()) {
          WriteJsonLine(*jsonl_stream, record_json);
          ++records_logged;
        }
      }

      previous_sample = current_sample;

      if (config.max_samples.has_value() && sample_index >= *config.max_samples) {
        break;
      }

      const auto elapsed = std::chrono::duration<double>(SteadyClock::now() - loop_start).count();
      const double sleep_s = std::max(0.0, dt - elapsed);
      std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }

    std::ofstream summary_stream(summary_path, std::ios::out | std::ios::trunc);
    summary_stream << BuildSummaryJson(
        config,
        output_root,
        jsonl_path,
        records_logged,
        sample_index,
        first_timestamp_s,
        last_timestamp_s);
    summary_stream.close();
    return 0;
  } catch (const franka::Exception& exc) {
    std::cerr << "[ERROR] libfranka exception: " << exc.what() << "\n";
    return 1;
  } catch (const std::exception& exc) {
    std::cerr << "[ERROR] " << exc.what() << "\n";
    return 1;
  }
}
