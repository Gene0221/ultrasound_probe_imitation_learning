#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

std::atomic_bool g_stop_requested{false};

void SignalHandler(int) {
  g_stop_requested.store(true);
}

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Quaternion {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
};

struct Pose {
  Vec3 p;
  Quaternion q;
};

struct Matrix4 {
  std::array<double, 16> data{};

  double& operator()(int row, int col) {
    return data[static_cast<std::size_t>(row + 4 * col)];
  }

  double operator()(int row, int col) const {
    return data[static_cast<std::size_t>(row + 4 * col)];
  }
};

struct Action {
  Vec3 translation;
  Quaternion rotation;
};

struct TrajectorySample {
  double time_s = 0.0;
  Pose pose;
};

struct PolicyChunk {
  long long seq = -1;
  double action_dt_s = 0.03;
  double speed_scale = 0.4;
  int execute_steps = 1;
  bool force_safety_ok = true;
  bool has_fz = false;
  double fz_N = 0.0;
  std::vector<Action> actions;
  std::chrono::steady_clock::time_point received_at = std::chrono::steady_clock::now();
};

struct SharedPolicyState {
  std::mutex mutex;
  std::optional<PolicyChunk> latest;
  std::optional<double> latest_fz_N;
  std::optional<double> calibration_initial_fz_N;
  bool python_ready = false;
  long long latest_force_request_id = -1;
};

struct Options {
  std::string robot_ip;
  std::string host = "127.0.0.1";
  int port = 50555;
  int action_horizon = 20;
  double receive_timeout_s = 0.15;
  double max_step_translation = 0.003;
  double max_step_rotation = 0.05;
  double max_translation_speed = 0.2;
  double max_translation_acceleration = 0.07;
  double max_rotation_speed = 0.35;
  double max_rotation_acceleration = 0.5;
  double ramp_time_s = 3.0;
  bool filter_enabled = true;
  double filter_cutoff_hz = 1.0;
  bool orientation_filter_enabled = true;
  double orientation_filter_cutoff_hz = 1.0;
  bool calibration_enabled = false;
  int calibration_interval_inferences = 3;
  double calibration_force_tolerance = 0.5;
  double calibration_z_gain = 0.0002;
  double calibration_z_sign = 1.0;
  double calibration_max_z_step = 0.0005;
  double calibration_max_total_z = 0.01;
  double calibration_orientation_tolerance = 0.01;
  int calibration_force_settle_cycles = 3;
  double calibration_force_sample_hz = 30.0;
};

enum class ControllerMode {
  kWaitingForPolicy,
  kRunning,
  kCalibrating,
  kHoldTimeout,
  kHoldForceLimit,
  kHoldStopRequested,
};

enum class CalibrationPhase {
  kNone,
  kOrientation,
  kForce,
};

const char* ModeName(ControllerMode mode) {
  switch (mode) {
    case ControllerMode::kWaitingForPolicy:
      return "WAIT_FOR_POLICY";
    case ControllerMode::kRunning:
      return "RUNNING";
    case ControllerMode::kHoldTimeout:
      return "HOLD_TIMEOUT";
    case ControllerMode::kHoldForceLimit:
      return "HOLD_FORCE_LIMIT";
    case ControllerMode::kHoldStopRequested:
      return "HOLD_STOP_REQUESTED";
    case ControllerMode::kCalibrating:
      return "CALIBRATING";
  }
  return "UNKNOWN";
}

double Clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(value, hi));
}

double Smoothstep(double value) {
  const double x = Clamp(value, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

double QuinticTimeScaling(double value) {
  const double x = Clamp(value, 0.0, 1.0);
  const double x2 = x * x;
  const double x3 = x2 * x;
  return 10.0 * x3 - 15.0 * x3 * x + 6.0 * x3 * x2;
}

Vec3 operator+(const Vec3& a, const Vec3& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3& v, double s) {
  return Vec3{v.x * s, v.y * s, v.z * s};
}

Vec3 operator/(const Vec3& v, double s) {
  return Vec3{v.x / s, v.y / s, v.z / s};
}

double Norm(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 LimitVectorNorm(const Vec3& v, double max_norm) {
  const double length = Norm(v);
  if (length <= max_norm || length < 1e-12) {
    return v;
  }
  return v * (max_norm / length);
}

Quaternion Normalize(const Quaternion& q) {
  const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n < 1e-12) {
    throw std::runtime_error("Quaternion norm is zero.");
  }
  return Quaternion{q.x / n, q.y / n, q.z / n, q.w / n};
}

Quaternion Conjugate(const Quaternion& q) {
  return Quaternion{-q.x, -q.y, -q.z, q.w};
}

Quaternion Multiply(const Quaternion& a, const Quaternion& b) {
  return Normalize(Quaternion{
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  });
}

Vec3 RotateVector(const Quaternion& q, const Vec3& v) {
  const Quaternion unit = Normalize(q);
  const Vec3 u{unit.x, unit.y, unit.z};
  const double s = unit.w;
  const Vec3 cross_uv{
      u.y * v.z - u.z * v.y,
      u.z * v.x - u.x * v.z,
      u.x * v.y - u.y * v.x,
  };
  const Vec3 cross_u_cross{
      u.y * cross_uv.z - u.z * cross_uv.y,
      u.z * cross_uv.x - u.x * cross_uv.z,
      u.x * cross_uv.y - u.y * cross_uv.x,
  };
  return v + cross_uv * (2.0 * s) + cross_u_cross * 2.0;
}

double Dot(const Quaternion& a, const Quaternion& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

Quaternion Slerp(Quaternion a, Quaternion b, double t) {
  a = Normalize(a);
  b = Normalize(b);
  double cos_theta = Dot(a, b);
  if (cos_theta < 0.0) {
    b = Quaternion{-b.x, -b.y, -b.z, -b.w};
    cos_theta = -cos_theta;
  }
  if (cos_theta > 0.9995) {
    return Normalize(Quaternion{
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z),
        a.w + t * (b.w - a.w),
    });
  }
  const double theta = std::acos(Clamp(cos_theta, -1.0, 1.0));
  const double sin_theta = std::sin(theta);
  const double w1 = std::sin((1.0 - t) * theta) / sin_theta;
  const double w2 = std::sin(t * theta) / sin_theta;
  return Normalize(Quaternion{
      a.x * w1 + b.x * w2,
      a.y * w1 + b.y * w2,
      a.z * w1 + b.z * w2,
      a.w * w1 + b.w * w2,
  });
}

double QuaternionAngle(const Quaternion& q) {
  const Quaternion unit = Normalize(q);
  return 2.0 * std::acos(Clamp(std::fabs(unit.w), -1.0, 1.0));
}

Vec3 QuaternionToRotationVector(Quaternion q) {
  q = Normalize(q);
  if (q.w < 0.0) {
    q = Quaternion{-q.x, -q.y, -q.z, -q.w};
  }
  const double vector_norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  if (vector_norm < 1e-12) {
    return Vec3{0.0, 0.0, 0.0};
  }
  const double angle = 2.0 * std::atan2(vector_norm, Clamp(q.w, -1.0, 1.0));
  const double scale = angle / vector_norm;
  return Vec3{q.x * scale, q.y * scale, q.z * scale};
}

Quaternion RotationVectorToQuaternion(const Vec3& rotvec) {
  const double angle = Norm(rotvec);
  if (angle < 1e-12) {
    return Quaternion{0.0, 0.0, 0.0, 1.0};
  }
  const Vec3 axis = rotvec / angle;
  const double half = 0.5 * angle;
  const double s = std::sin(half);
  return Normalize(Quaternion{axis.x * s, axis.y * s, axis.z * s, std::cos(half)});
}

Matrix4 Identity() {
  Matrix4 m;
  m(0, 0) = 1.0;
  m(1, 1) = 1.0;
  m(2, 2) = 1.0;
  m(3, 3) = 1.0;
  return m;
}

Matrix4 PoseToMatrix(const Pose& pose) {
  const Quaternion q = Normalize(pose.q);
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;

  Matrix4 m = Identity();
  m(0, 0) = 1.0 - 2.0 * (yy + zz);
  m(0, 1) = 2.0 * (xy - wz);
  m(0, 2) = 2.0 * (xz + wy);
  m(1, 0) = 2.0 * (xy + wz);
  m(1, 1) = 1.0 - 2.0 * (xx + zz);
  m(1, 2) = 2.0 * (yz - wx);
  m(2, 0) = 2.0 * (xz - wy);
  m(2, 1) = 2.0 * (yz + wx);
  m(2, 2) = 1.0 - 2.0 * (xx + yy);
  m(0, 3) = pose.p.x;
  m(1, 3) = pose.p.y;
  m(2, 3) = pose.p.z;
  return m;
}

Matrix4 Multiply(const Matrix4& a, const Matrix4& b) {
  Matrix4 out{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      double value = 0.0;
      for (int k = 0; k < 4; ++k) {
        value += a(r, k) * b(k, c);
      }
      out(r, c) = value;
    }
  }
  return out;
}

Quaternion MatrixToQuaternion(const Matrix4& m) {
  const double trace = m(0, 0) + m(1, 1) + m(2, 2);
  Quaternion q;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (m(2, 1) - m(1, 2)) / s;
    q.y = (m(0, 2) - m(2, 0)) / s;
    q.z = (m(1, 0) - m(0, 1)) / s;
  } else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) {
    const double s = std::sqrt(1.0 + m(0, 0) - m(1, 1) - m(2, 2)) * 2.0;
    q.w = (m(2, 1) - m(1, 2)) / s;
    q.x = 0.25 * s;
    q.y = (m(0, 1) + m(1, 0)) / s;
    q.z = (m(0, 2) + m(2, 0)) / s;
  } else if (m(1, 1) > m(2, 2)) {
    const double s = std::sqrt(1.0 + m(1, 1) - m(0, 0) - m(2, 2)) * 2.0;
    q.w = (m(0, 2) - m(2, 0)) / s;
    q.x = (m(0, 1) + m(1, 0)) / s;
    q.y = 0.25 * s;
    q.z = (m(1, 2) + m(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + m(2, 2) - m(0, 0) - m(1, 1)) * 2.0;
    q.w = (m(1, 0) - m(0, 1)) / s;
    q.x = (m(0, 2) + m(2, 0)) / s;
    q.y = (m(1, 2) + m(2, 1)) / s;
    q.z = 0.25 * s;
  }
  return Normalize(q);
}

Pose MatrixToPose(const Matrix4& m) {
  return Pose{Vec3{m(0, 3), m(1, 3), m(2, 3)}, MatrixToQuaternion(m)};
}

std::array<double, 16> MatrixToArray(const Matrix4& m) {
  return m.data;
}

Matrix4 ArrayToMatrix(const std::array<double, 16>& values) {
  Matrix4 m;
  m.data = values;
  return m;
}

Matrix4 ActionToMatrix(const Action& action) {
  return PoseToMatrix(Pose{action.translation, action.rotation});
}

double LowpassAlpha(double dt_s, double cutoff_hz) {
  const double pi = std::acos(-1.0);
  const double tau = 1.0 / (2.0 * pi * cutoff_hz);
  return dt_s / (tau + dt_s);
}

Pose LowpassPose(const Pose& previous, const Pose& raw, double dt_s, const Options& opt) {
  Pose out = raw;
  if (opt.filter_enabled) {
    const double alpha = LowpassAlpha(dt_s, opt.filter_cutoff_hz);
    out.p = previous.p + (raw.p - previous.p) * alpha;
  }
  if (opt.orientation_filter_enabled) {
    const double alpha = LowpassAlpha(dt_s, opt.orientation_filter_cutoff_hz);
    const Quaternion delta = Multiply(raw.q, Conjugate(previous.q));
    const Vec3 rotvec = QuaternionToRotationVector(delta);
    out.q = Multiply(RotationVectorToQuaternion(rotvec * alpha), previous.q);
  }
  return out;
}

TrajectorySample Interpolate(const std::vector<TrajectorySample>& samples, double t) {
  if (samples.empty()) {
    throw std::runtime_error("No trajectory samples available.");
  }
  if (t <= samples.front().time_s) {
    return samples.front();
  }
  if (t >= samples.back().time_s) {
    return samples.back();
  }

  auto upper = std::upper_bound(samples.begin(), samples.end(), t, [](double value, const TrajectorySample& sample) {
    return value < sample.time_s;
  });
  const auto& b = *upper;
  const auto& a = *(upper - 1);
  const double s = (t - a.time_s) / (b.time_s - a.time_s);
  const double alpha = QuinticTimeScaling(s);
  return TrajectorySample{
      t,
      Pose{
          a.pose.p * (1.0 - alpha) + b.pose.p * alpha,
          Slerp(a.pose.q, b.pose.q, alpha),
      },
  };
}

std::vector<TrajectorySample> BuildTrajectory(const Pose& start_pose, const PolicyChunk& chunk, const Options& opt) {
  const double action_dt = chunk.action_dt_s / std::max(chunk.speed_scale, 1e-6);
  const int execute_steps = std::max(1, std::min(chunk.execute_steps, static_cast<int>(chunk.actions.size())));
  std::vector<TrajectorySample> samples;
  samples.reserve(static_cast<std::size_t>(execute_steps + 1));
  samples.push_back(TrajectorySample{0.0, start_pose});

  Matrix4 cumulative = PoseToMatrix(start_pose);
  Pose filtered_previous = start_pose;
  for (int i = 0; i < execute_steps; ++i) {
    Action action = chunk.actions[static_cast<std::size_t>(i)];
    action.translation = LimitVectorNorm(action.translation, opt.max_step_translation);
    const Vec3 rotvec = LimitVectorNorm(QuaternionToRotationVector(action.rotation), opt.max_step_rotation);
    action.rotation = RotationVectorToQuaternion(rotvec);
    cumulative = Multiply(cumulative, ActionToMatrix(action));
    const Pose raw_pose = MatrixToPose(cumulative);
    const Pose filtered = LowpassPose(filtered_previous, raw_pose, action_dt, opt);
    samples.push_back(TrajectorySample{(i + 1) * action_dt, filtered});
    filtered_previous = filtered;
  }
  return samples;
}

std::vector<double> ParseNumberArray(const std::string& line, const std::string& key) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = line.find(quoted_key);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("Missing JSON key: " + key);
  }
  const std::size_t start = line.find('[', key_pos);
  if (start == std::string::npos) {
    throw std::runtime_error("Missing array for JSON key: " + key);
  }
  int depth = 0;
  std::size_t end = std::string::npos;
  for (std::size_t i = start; i < line.size(); ++i) {
    if (line[i] == '[') {
      ++depth;
    } else if (line[i] == ']') {
      --depth;
      if (depth == 0) {
        end = i;
        break;
      }
    }
  }
  if (end == std::string::npos) {
    throw std::runtime_error("Unclosed array for JSON key: " + key);
  }

  std::vector<double> numbers;
  const std::string text = line.substr(start, end - start + 1);
  const char* ptr = text.c_str();
  char* parsed_end = nullptr;
  while (*ptr != '\0') {
    if ((*ptr >= '0' && *ptr <= '9') || *ptr == '-' || *ptr == '+' || *ptr == '.') {
      const double value = std::strtod(ptr, &parsed_end);
      if (parsed_end != ptr) {
        numbers.push_back(value);
        ptr = parsed_end;
        continue;
      }
    }
    ++ptr;
  }
  return numbers;
}

double ParseNumberField(const std::string& line, const std::string& key, double default_value) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = line.find(quoted_key);
  if (key_pos == std::string::npos) {
    return default_value;
  }
  const std::size_t colon = line.find(':', key_pos);
  if (colon == std::string::npos) {
    return default_value;
  }
  const char* ptr = line.c_str() + colon + 1;
  char* parsed_end = nullptr;
  const double value = std::strtod(ptr, &parsed_end);
  return parsed_end == ptr ? default_value : value;
}

bool ParseBoolField(const std::string& line, const std::string& key, bool default_value) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = line.find(quoted_key);
  if (key_pos == std::string::npos) {
    return default_value;
  }
  const std::size_t colon = line.find(':', key_pos);
  if (colon == std::string::npos) {
    return default_value;
  }
  const std::size_t value_pos = line.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos) {
    return default_value;
  }
  if (line.compare(value_pos, 4, "true") == 0) {
    return true;
  }
  if (line.compare(value_pos, 5, "false") == 0) {
    return false;
  }
  return default_value;
}

PolicyChunk ParsePolicyChunk(const std::string& line, const Options& opt) {
  PolicyChunk chunk;
  chunk.seq = static_cast<long long>(ParseNumberField(line, "seq", -1.0));
  chunk.action_dt_s = ParseNumberField(line, "action_dt_s", 0.03);
  chunk.speed_scale = ParseNumberField(line, "speed_scale", 0.4);
  chunk.execute_steps = static_cast<int>(ParseNumberField(line, "execute_steps", 1.0));
  chunk.force_safety_ok = ParseBoolField(line, "force_safety_ok", true);
  double fz = ParseNumberField(line, "calibration_Fz_N", std::numeric_limits<double>::quiet_NaN());
  if (!std::isfinite(fz)) {
    fz = ParseNumberField(line, "Fz_N", std::numeric_limits<double>::quiet_NaN());
  }
  if (std::isfinite(fz)) {
    chunk.has_fz = true;
    chunk.fz_N = fz;
  }
  chunk.received_at = std::chrono::steady_clock::now();

  const std::vector<double> values = ParseNumberArray(line, "actions");
  if (values.size() % 7 != 0) {
    throw std::runtime_error("actions must contain a multiple of 7 numeric values.");
  }
  const std::size_t action_count = values.size() / 7;
  if (action_count == 0 || action_count > static_cast<std::size_t>(opt.action_horizon)) {
    throw std::runtime_error("actions count is outside configured horizon.");
  }
  chunk.actions.reserve(action_count);
  for (std::size_t i = 0; i < action_count; ++i) {
    const std::size_t j = i * 7;
    chunk.actions.push_back(Action{
        Vec3{values[j], values[j + 1], values[j + 2]},
        Normalize(Quaternion{values[j + 3], values[j + 4], values[j + 5], values[j + 6]}),
    });
  }
  return chunk;
}

void SetConservativeCollisionBehavior(franka::Robot& robot) {
  robot.setCollisionBehavior(
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
}

#ifndef _WIN32
class LineServer {
 public:
  LineServer(std::string host, int port) : host_(std::move(host)), port_(port) {}

  void Run(const Options& opt, SharedPolicyState& state) {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
      throw std::runtime_error("Failed to create TCP socket.");
    }
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &address.sin_addr) != 1) {
      ::close(server_fd);
      throw std::runtime_error("Invalid --host IPv4 address: " + host_);
    }
    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
      ::close(server_fd);
      throw std::runtime_error("Failed to bind TCP socket.");
    }
    if (::listen(server_fd, 1) < 0) {
      ::close(server_fd);
      throw std::runtime_error("Failed to listen on TCP socket.");
    }

    std::cout << "[INFO] Waiting for Python policy stream on " << host_ << ":" << port_ << "\n";
    while (!g_stop_requested.load()) {
      sockaddr_in client_address{};
      socklen_t client_len = sizeof(client_address);
      const int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_address), &client_len);
      if (client_fd < 0) {
        if (!g_stop_requested.load()) {
          std::cerr << "[WARN] accept failed: " << std::strerror(errno) << "\n";
        }
        continue;
      }
      std::cout << "[INFO] Python policy client connected.\n";
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        client_fd_ = client_fd;
      }
      ReadClient(client_fd, opt, state);
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        client_fd_ = -1;
      }
      ::close(client_fd);
      std::cout << "[WARN] Python policy client disconnected.\n";
    }
    ::close(server_fd);
  }

  bool SendCommand(const std::string& command, long long request_id) {
    const std::string message = "{\"command\":\"" + command + "\",\"request_id\":" +
                                std::to_string(request_id) + "}\n";
    std::lock_guard<std::mutex> lock(client_mutex_);
    if (client_fd_ < 0) {
      return false;
    }
    std::size_t sent = 0;
    while (sent < message.size()) {
      const ssize_t count = ::send(client_fd_, message.data() + sent, message.size() - sent, 0);
      if (count <= 0) {
        return false;
      }
      sent += static_cast<std::size_t>(count);
    }
    return true;
  }

 private:
  void ReadClient(int client_fd, const Options& opt, SharedPolicyState& state) {
    std::string buffer;
    std::array<char, 4096> chunk{};
    bool first_chunk_logged = false;
    while (!g_stop_requested.load()) {
      const ssize_t count = ::recv(client_fd, chunk.data(), chunk.size(), 0);
      if (count <= 0) {
        return;
      }
      buffer.append(chunk.data(), static_cast<std::size_t>(count));
      std::size_t newline = std::string::npos;
      while ((newline = buffer.find('\n')) != std::string::npos) {
        const std::string line = buffer.substr(0, newline);
        buffer.erase(0, newline + 1);
        if (line.empty()) {
          continue;
        }
        try {
          if (line.find("\"mode\":\"ready\"") != std::string::npos) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.python_ready = true;
            std::cout << "[INFO] Python inference service is ready for C++ requests.\n";
            continue;
          }
          if (line.find("\"mode\":\"force_sample\"") != std::string::npos) {
            double fz = ParseNumberField(line, "calibration_Fz_N", std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(fz)) {
              fz = ParseNumberField(line, "Fz_N", std::numeric_limits<double>::quiet_NaN());
            }
            if (!std::isfinite(fz)) {
              throw std::runtime_error("calibration message missing force.Fz_N");
            }
            const long long request_id = static_cast<long long>(ParseNumberField(line, "request_id", -1.0));
            std::lock_guard<std::mutex> lock(state.mutex);
            state.latest_fz_N = fz;
            state.latest_force_request_id = request_id;
            const double initial_fz =
                ParseNumberField(line, "calibration_initial_force_N", std::numeric_limits<double>::quiet_NaN());
            if (std::isfinite(initial_fz)) {
              state.calibration_initial_fz_N = initial_fz;
            }
            continue;
          }
          PolicyChunk parsed = ParsePolicyChunk(line, opt);
          const long long seq = parsed.seq;
          const std::size_t action_count = parsed.actions.size();
          const bool force_ok = parsed.force_safety_ok;
          std::lock_guard<std::mutex> lock(state.mutex);
          if (parsed.has_fz) {
            state.latest_fz_N = parsed.fz_N;
          }
          const double initial_fz =
              ParseNumberField(line, "calibration_initial_force_N", std::numeric_limits<double>::quiet_NaN());
          if (std::isfinite(initial_fz)) {
            state.calibration_initial_fz_N = initial_fz;
          }
          state.latest = std::move(parsed);
          if (!first_chunk_logged) {
            std::cout << "[INFO] First policy chunk received: seq=" << seq
                      << " actions=" << action_count
                      << " force_ok=" << (force_ok ? "true" : "false") << "\n";
            first_chunk_logged = true;
          }
        } catch (const std::exception& e) {
          std::cerr << "[WARN] Dropped invalid policy chunk: " << e.what() << "\n";
        }
      }
    }
  }

  std::string host_;
  int port_;
  std::mutex client_mutex_;
  int client_fd_ = -1;
};
#endif

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --robot-ip <ip> [options]\n\n"
      << "Options:\n"
      << "  --host <ip>                         Default: 127.0.0.1\n"
      << "  --port <port>                       Default: 50555\n"
      << "  --receive-timeout-ms <ms>           Default: 150\n"
      << "  --action-horizon <n>                Default: 20\n"
      << "  --max-step-translation <m>          Default: 0.003\n"
      << "  --max-step-rotation <rad>           Default: 0.05\n"
      << "  --max-translation-speed <m/s>       Default: 0.2\n"
      << "  --max-translation-acceleration <m/s^2> Default: 0.07\n"
      << "  --max-rotation-speed <rad/s>        Default: 0.35\n"
      << "  --max-rotation-acceleration <rad/s^2> Default: 0.5\n"
      << "  --ramp-time <s>                     Default: 3.0\n"
      << "  --disable-filter\n"
      << "  --filter-cutoff-hz <hz>             Default: 1.0\n"
      << "  --disable-orientation-filter\n"
      << "  --orientation-filter-cutoff-hz <hz> Default: 1.0\n"
      << "  --enable-calibration\n"
      << "  --calibration-interval-inferences <n> Default: 3\n"
      << "  --calibration-force-tolerance <N>   Default: 0.5\n"
      << "  --calibration-z-gain <m/N>          Default: 0.0002\n"
      << "  --calibration-z-sign <sign>         Default: 1.0\n"
      << "  --calibration-max-z-step <m>        Default: 0.0005\n"
      << "  --calibration-max-total-z <m>       Default: 0.01\n"
      << "  --calibration-orientation-tolerance <rad> Default: 0.01\n"
      << "  --calibration-force-settle-cycles <n> Default: 3\n"
      << "  --calibration-force-sample-hz <hz>   Default: 30\n";
}

Options ParseArgs(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--robot-ip") {
      opt.robot_ip = require_value(arg);
    } else if (arg == "--host") {
      opt.host = require_value(arg);
    } else if (arg == "--port") {
      opt.port = std::stoi(require_value(arg));
    } else if (arg == "--receive-timeout-ms") {
      opt.receive_timeout_s = std::stod(require_value(arg)) / 1000.0;
    } else if (arg == "--action-horizon") {
      opt.action_horizon = std::stoi(require_value(arg));
    } else if (arg == "--max-step-translation") {
      opt.max_step_translation = std::stod(require_value(arg));
    } else if (arg == "--max-step-rotation") {
      opt.max_step_rotation = std::stod(require_value(arg));
    } else if (arg == "--max-translation-speed") {
      opt.max_translation_speed = std::stod(require_value(arg));
    } else if (arg == "--max-translation-acceleration") {
      opt.max_translation_acceleration = std::stod(require_value(arg));
    } else if (arg == "--max-rotation-speed") {
      opt.max_rotation_speed = std::stod(require_value(arg));
    } else if (arg == "--max-rotation-acceleration") {
      opt.max_rotation_acceleration = std::stod(require_value(arg));
    } else if (arg == "--ramp-time") {
      opt.ramp_time_s = std::stod(require_value(arg));
    } else if (arg == "--disable-filter") {
      opt.filter_enabled = false;
    } else if (arg == "--filter-cutoff-hz") {
      opt.filter_cutoff_hz = std::stod(require_value(arg));
    } else if (arg == "--disable-orientation-filter") {
      opt.orientation_filter_enabled = false;
    } else if (arg == "--orientation-filter-cutoff-hz") {
      opt.orientation_filter_cutoff_hz = std::stod(require_value(arg));
    } else if (arg == "--enable-calibration") {
      opt.calibration_enabled = true;
    } else if (arg == "--calibration-interval-inferences") {
      opt.calibration_interval_inferences = std::stoi(require_value(arg));
    } else if (arg == "--calibration-force-tolerance") {
      opt.calibration_force_tolerance = std::stod(require_value(arg));
    } else if (arg == "--calibration-z-gain") {
      opt.calibration_z_gain = std::stod(require_value(arg));
    } else if (arg == "--calibration-z-sign") {
      opt.calibration_z_sign = std::stod(require_value(arg));
    } else if (arg == "--calibration-max-z-step") {
      opt.calibration_max_z_step = std::stod(require_value(arg));
    } else if (arg == "--calibration-max-total-z") {
      opt.calibration_max_total_z = std::stod(require_value(arg));
    } else if (arg == "--calibration-orientation-tolerance") {
      opt.calibration_orientation_tolerance = std::stod(require_value(arg));
    } else if (arg == "--calibration-force-settle-cycles") {
      opt.calibration_force_settle_cycles = std::stoi(require_value(arg));
    } else if (arg == "--calibration-force-sample-hz") {
      opt.calibration_force_sample_hz = std::stod(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (opt.robot_ip.empty()) {
    throw std::runtime_error("--robot-ip is required.");
  }
  if (opt.port <= 0 || opt.port > 65535) {
    throw std::runtime_error("--port must be in [1, 65535].");
  }
  if (opt.action_horizon <= 0) {
    throw std::runtime_error("--action-horizon must be positive.");
  }
  if (opt.receive_timeout_s <= 0.0) {
    throw std::runtime_error("--receive-timeout-ms must be positive.");
  }
  if (opt.filter_cutoff_hz <= 0.0 || opt.orientation_filter_cutoff_hz <= 0.0) {
    throw std::runtime_error("Filter cutoff values must be positive.");
  }
  if (opt.calibration_interval_inferences <= 0 || opt.calibration_force_settle_cycles <= 0 ||
      opt.calibration_force_sample_hz <= 0.0) {
    throw std::runtime_error("Calibration interval and settle cycles must be positive.");
  }
  if (opt.calibration_force_tolerance <= 0.0 || opt.calibration_z_gain < 0.0 ||
      opt.calibration_max_z_step < 0.0 || opt.calibration_max_total_z < 0.0 ||
      opt.calibration_orientation_tolerance < 0.0) {
    throw std::runtime_error("Calibration numeric limits must be non-negative, and force tolerance must be positive.");
  }
  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, SignalHandler);

  try {
    const Options opt = ParseArgs(argc, argv);
#ifdef _WIN32
    throw std::runtime_error("rolling_policy_controller uses POSIX sockets and is intended for the Linux Franka control computer.");
#else
    auto shared_state = std::make_shared<SharedPolicyState>();
    auto server = std::make_shared<LineServer>(opt.host, opt.port);
    std::thread receiver([&, server, shared_state]() {
      try {
        server->Run(opt, *shared_state);
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Receiver stopped: " << e.what() << "\n";
        g_stop_requested.store(true);
      }
    });
    receiver.detach();

    std::cout << "[INFO] Connecting to Franka at " << opt.robot_ip << "\n";
    franka::Robot robot(opt.robot_ip);
    robot.automaticErrorRecovery();
    SetConservativeCollisionBehavior(robot);

    const franka::RobotState initial_state = robot.readOnce();
    Pose commanded_pose = MatrixToPose(ArrayToMatrix(initial_state.O_T_EE_c));
    const Pose initial_measured_pose = MatrixToPose(ArrayToMatrix(initial_state.O_T_EE));
    const Quaternion initial_orientation = initial_measured_pose.q;
    Vec3 commanded_velocity{0.0, 0.0, 0.0};
    Vec3 commanded_angular_velocity{0.0, 0.0, 0.0};
    std::vector<TrajectorySample> active_trajectory{TrajectorySample{0.0, commanded_pose}};
    long long active_seq = -1;
    long long pending_infer_request_id = -1;
    long long next_request_id = 0;
    long long pending_force_request_id = -1;
    long long last_applied_force_request_id = -1;
    int completed_inferences = 0;
    ControllerMode current_mode = ControllerMode::kWaitingForPolicy;
    double trajectory_elapsed_s = 0.0;
    double control_elapsed_s = 0.0;
    std::optional<double> initial_fz_N;
    CalibrationPhase calibration_phase = CalibrationPhase::kNone;
    int calibration_settled_cycles = 0;
    double calibration_total_z = 0.0;
    Pose calibration_target = commanded_pose;
    auto last_force_request_time = std::chrono::steady_clock::now();

    if (opt.calibration_enabled) {
      std::cout << "[INFO] Calibration enabled. Initial EE orientation captured: quaternion_xyzw=["
                << initial_orientation.x << ", " << initial_orientation.y << ", "
                << initial_orientation.z << ", " << initial_orientation.w << "]\n";
    } else {
      std::cout << "[INFO] Calibration disabled.\n";
    }

    std::cout << "[INFO] Realtime rolling controller started. Press Ctrl+C to stop.\n";
    robot.control([&](const franka::RobotState&, franka::Duration period) -> franka::CartesianPose {
      const double dt = period.toSec();
      control_elapsed_s += dt;
      trajectory_elapsed_s += dt;

      std::optional<PolicyChunk> latest;
      std::optional<double> latest_fz_N;
      std::optional<double> calibration_initial_fz_N;
      bool python_ready = false;
      long long latest_force_request_id = -1;
      {
        std::lock_guard<std::mutex> lock(shared_state->mutex);
        latest = shared_state->latest;
        latest_fz_N = shared_state->latest_fz_N;
        calibration_initial_fz_N = shared_state->calibration_initial_fz_N;
        python_ready = shared_state->python_ready;
        latest_force_request_id = shared_state->latest_force_request_id;
      }

      if (opt.calibration_enabled && !initial_fz_N.has_value() && calibration_initial_fz_N.has_value()) {
        initial_fz_N = *calibration_initial_fz_N;
        std::cout << "[INFO] Calibration force reference received by controller: Fz="
                  << *initial_fz_N << " N\n";
      }

      ControllerMode next_mode = ControllerMode::kWaitingForPolicy;
      const bool active_chunk_running =
          active_seq >= 0 &&
          !active_trajectory.empty() &&
          trajectory_elapsed_s <= active_trajectory.back().time_s;
      if (g_stop_requested.load()) {
        next_mode = ControllerMode::kHoldStopRequested;
      } else if (calibration_phase != CalibrationPhase::kNone) {
        next_mode = ControllerMode::kCalibrating;
      } else if (active_chunk_running) {
        next_mode = ControllerMode::kRunning;
      } else if (active_seq >= 0) {
        ++completed_inferences;
        std::cout << "[INFO] Policy chunk completed: seq=" << active_seq
                  << " completed_inferences=" << completed_inferences << "\n";
        active_seq = -1;
        active_trajectory = std::vector<TrajectorySample>{TrajectorySample{0.0, commanded_pose}};
        trajectory_elapsed_s = 0.0;
        if (opt.calibration_enabled && completed_inferences >= opt.calibration_interval_inferences) {
          calibration_phase = CalibrationPhase::kOrientation;
          calibration_target = commanded_pose;
          calibration_target.q = initial_orientation;
          calibration_settled_cycles = 0;
          calibration_total_z = 0.0;
          std::cout << "[INFO] Starting calibration: orientation phase at fixed position.\n";
          next_mode = ControllerMode::kCalibrating;
        }
      }
      if (calibration_phase == CalibrationPhase::kNone && active_seq < 0 && pending_infer_request_id < 0 && python_ready) {
        const long long request_id = next_request_id++;
        if (server->SendCommand("infer", request_id)) {
          pending_infer_request_id = request_id;
          std::cout << "[INFO] Requested inference: request_id=" << request_id << "\n";
        }
      }
      if (calibration_phase == CalibrationPhase::kNone && pending_infer_request_id >= 0 && latest.has_value() &&
          latest->seq == pending_infer_request_id) {
        pending_infer_request_id = -1;
        if (!latest->force_safety_ok) {
          next_mode = ControllerMode::kHoldForceLimit;
        } else {
          active_trajectory = BuildTrajectory(commanded_pose, *latest, opt);
          active_seq = latest->seq;
          trajectory_elapsed_s = 0.0;
          next_mode = ControllerMode::kRunning;
          std::cout << "[INFO] Activated requested policy chunk: seq=" << active_seq
                    << " duration_s=" << active_trajectory.back().time_s << "\n";
        }
      }
      const bool hold_position = next_mode != ControllerMode::kRunning && next_mode != ControllerMode::kCalibrating;
      if (next_mode != current_mode) {
        std::cout << "[INFO] Controller state: " << ModeName(current_mode)
                  << " -> " << ModeName(next_mode) << "\n";
        current_mode = next_mode;
      }

      Pose target_pose = commanded_pose;
      if (calibration_phase == CalibrationPhase::kOrientation) {
        target_pose = calibration_target;
        const Quaternion orientation_error_q = Multiply(initial_orientation, Conjugate(commanded_pose.q));
        const bool orientation_in_range =
            QuaternionAngle(orientation_error_q) <= opt.calibration_orientation_tolerance;
        if (orientation_in_range) {
          calibration_settled_cycles += 1;
        } else {
          calibration_settled_cycles = 0;
        }
        if (calibration_settled_cycles >= opt.calibration_force_settle_cycles) {
          calibration_phase = CalibrationPhase::kForce;
          calibration_settled_cycles = 0;
          calibration_target.p = commanded_pose.p;
          std::cout << "[INFO] Calibration orientation complete. Starting EE-z force phase.\n";
        }
      } else if (calibration_phase == CalibrationPhase::kForce) {
        target_pose = calibration_target;
        const auto now = std::chrono::steady_clock::now();
        const double sample_period_s = 1.0 / std::max(opt.calibration_force_sample_hz, 1e-6);
        if (pending_force_request_id < 0 &&
            std::chrono::duration<double>(now - last_force_request_time).count() >= sample_period_s && python_ready) {
          const long long request_id = next_request_id++;
          if (server->SendCommand("force_sample", request_id)) {
            pending_force_request_id = request_id;
            last_force_request_time = now;
          }
        }
        if (pending_force_request_id >= 0 && latest_force_request_id == pending_force_request_id &&
            latest_fz_N.has_value() && initial_fz_N.has_value() &&
            pending_force_request_id != last_applied_force_request_id) {
          const double force_error = *latest_fz_N - *initial_fz_N;
          last_applied_force_request_id = pending_force_request_id;
          pending_force_request_id = -1;
          if (std::fabs(force_error) <= opt.calibration_force_tolerance) {
            ++calibration_settled_cycles;
          } else {
            calibration_settled_cycles = 0;
            double z_step = opt.calibration_z_sign * (-force_error) * opt.calibration_z_gain;
            z_step = Clamp(z_step, -opt.calibration_max_z_step, opt.calibration_max_z_step);
            const double clamped_total = Clamp(calibration_total_z + z_step,
                                               -opt.calibration_max_total_z,
                                               opt.calibration_max_total_z);
            z_step = clamped_total - calibration_total_z;
            calibration_total_z = clamped_total;
            calibration_target.p = calibration_target.p + RotateVector(initial_orientation, Vec3{0.0, 0.0, 1.0}) * z_step;
          }
          if (calibration_settled_cycles >= opt.calibration_force_settle_cycles) {
            calibration_phase = CalibrationPhase::kNone;
            completed_inferences = 0;
            calibration_settled_cycles = 0;
            std::cout << "[INFO] Calibration complete: total_z_correction=" << calibration_total_z << " m\n";
          }
        }
      } else if (!hold_position && !active_trajectory.empty()) {
        target_pose = Interpolate(active_trajectory, trajectory_elapsed_s).pose;
        if (trajectory_elapsed_s > active_trajectory.back().time_s) {
          target_pose = commanded_pose;
        }
      }

      const double ramp_factor = opt.ramp_time_s > 1e-9 ? Smoothstep(control_elapsed_s / opt.ramp_time_s) : 1.0;
      const double max_translation_speed = opt.max_translation_speed * ramp_factor;
      const double max_rotation_speed = opt.max_rotation_speed * ramp_factor;

      const Vec3 translation_error = target_pose.p - commanded_pose.p;
      const double translation_error_norm = Norm(translation_error);
      const double stopping_speed = std::sqrt(2.0 * opt.max_translation_acceleration * translation_error_norm);
      const double desired_speed_limit = std::min(max_translation_speed, stopping_speed);
      const Vec3 desired_velocity = LimitVectorNorm(translation_error / std::max(dt, 1e-9), desired_speed_limit);
      const Vec3 velocity_delta = LimitVectorNorm(desired_velocity - commanded_velocity, opt.max_translation_acceleration * dt);
      commanded_velocity = LimitVectorNorm(commanded_velocity + velocity_delta, max_translation_speed);

      if (translation_error_norm < 1e-7 && Norm(commanded_velocity) < 1e-5) {
        commanded_pose.p = target_pose.p;
        commanded_velocity = Vec3{0.0, 0.0, 0.0};
      } else {
        commanded_pose.p = commanded_pose.p + commanded_velocity * dt;
      }

      const Quaternion rotation_error_q = Multiply(target_pose.q, Conjugate(commanded_pose.q));
      const Vec3 rotation_error = QuaternionToRotationVector(rotation_error_q);
      const double rotation_error_norm = Norm(rotation_error);
      const double stopping_angular_speed = std::sqrt(2.0 * opt.max_rotation_acceleration * rotation_error_norm);
      const double desired_angular_speed_limit = std::min(max_rotation_speed, stopping_angular_speed);
      const Vec3 desired_angular_velocity = LimitVectorNorm(rotation_error / std::max(dt, 1e-9), desired_angular_speed_limit);
      const Vec3 angular_velocity_delta = LimitVectorNorm(
          desired_angular_velocity - commanded_angular_velocity,
          opt.max_rotation_acceleration * dt);
      commanded_angular_velocity = LimitVectorNorm(commanded_angular_velocity + angular_velocity_delta, max_rotation_speed);

      if (rotation_error_norm < 1e-7 && Norm(commanded_angular_velocity) < 1e-5) {
        commanded_pose.q = Normalize(target_pose.q);
        commanded_angular_velocity = Vec3{0.0, 0.0, 0.0};
      } else {
        commanded_pose.q = Multiply(RotationVectorToQuaternion(commanded_angular_velocity * dt), commanded_pose.q);
      }

      const Matrix4 command_matrix = PoseToMatrix(commanded_pose);
      if (g_stop_requested.load() && Norm(commanded_velocity) < 1e-5 && Norm(commanded_angular_velocity) < 1e-5) {
        return franka::MotionFinished(franka::CartesianPose(MatrixToArray(command_matrix)));
      }
      return franka::CartesianPose(MatrixToArray(command_matrix));
    });

    g_stop_requested.store(true);
    std::cout << "[INFO] Realtime rolling controller stopped.\n";
    return 0;
#endif
  } catch (const franka::Exception& e) {
    std::cerr << "[FRANKA ERROR] " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
