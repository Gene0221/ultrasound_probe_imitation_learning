#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

struct TrajectorySample {
  double time_s = 0.0;
  Pose pose;
  double target_fz = 0.0;
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

struct Options {
  std::string robot_ip;
  std::string trajectory_path;
  std::string mode = "relative";
  double speed_scale = 1.0;
  double max_translation_speed = 0.03;
  double max_translation_acceleration = 0.01;
  double max_rotation_speed = 0.35;
  double max_rotation_acceleration = 0.1;
  double ramp_time_s = 2.0;
  bool hold_at_end = false;
  bool enable_force_correction = false;
  double force_gain = 0.0001;
  double max_force_correction = 0.002;
  double force_sign = 1.0;
  double bspline_smoothing_factor = 0.0016;
};

double Clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(value, hi));
}

double Smoothstep(double value) {
  const double x = Clamp(value, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

std::array<double, 4> CubicBsplineBasis(double value) {
  const double u = Clamp(value, 0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  return std::array<double, 4>{
      (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
      (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
      (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
      u3 / 6.0,
  };
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

double Dot(const Quaternion& a, const Quaternion& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

double QuaternionAngle(const Quaternion& a, const Quaternion& b) {
  const double cos_theta = std::fabs(Dot(Normalize(a), Normalize(b)));
  return 2.0 * std::acos(Clamp(cos_theta, -1.0, 1.0));
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

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> cells;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    cells.push_back(cell);
  }
  return cells;
}

std::string Trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::string UnquoteCsvCell(const std::string& text) {
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

double ParseDouble(const std::string& text, const std::string& field_name) {
  const std::string trimmed = UnquoteCsvCell(Trim(text));
  try {
    std::size_t parsed = 0;
    const double value = std::stod(trimmed, &parsed);
    if (parsed != trimmed.size()) {
      throw std::runtime_error("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("Failed to parse " + field_name + " from '" + text + "'.");
  }
}

std::vector<TrajectorySample> LoadTrajectoryCsv(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot open trajectory file: " + path);
  }

  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("Trajectory file is empty: " + path);
  }

  std::vector<TrajectorySample> samples;
  int line_no = 1;
  while (std::getline(input, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::vector<std::string> c = SplitCsvLine(line);
    if (c.size() < 8) {
      throw std::runtime_error("Line " + std::to_string(line_no) + " has fewer than 8 columns.");
    }
    TrajectorySample sample;
    sample.time_s = ParseDouble(c[0], "time_s");
    sample.pose.p = Vec3{ParseDouble(c[1], "x"), ParseDouble(c[2], "y"), ParseDouble(c[3], "z")};
    sample.pose.q = Normalize(Quaternion{ParseDouble(c[4], "qx"), ParseDouble(c[5], "qy"), ParseDouble(c[6], "qz"), ParseDouble(c[7], "qw")});
    sample.target_fz = c.size() >= 9 ? ParseDouble(c[8], "target_fz") : 0.0;
    samples.push_back(sample);
  }

  if (samples.size() < 2) {
    throw std::runtime_error("At least two trajectory samples are required.");
  }
  for (std::size_t i = 1; i < samples.size(); ++i) {
    if (samples[i].time_s <= samples[i - 1].time_s) {
      throw std::runtime_error("Trajectory time_s must be strictly increasing.");
    }
  }
  return samples;
}

TrajectorySample SmoothSample(
    const TrajectorySample& previous,
    const TrajectorySample& current,
    const TrajectorySample& next,
    double factor) {
  const double denom = 1.0 + 2.0 * factor;

  TrajectorySample out = current;
  out.pose.p = (current.pose.p + (previous.pose.p + next.pose.p) * factor) / denom;
  out.target_fz = (current.target_fz + factor * (previous.target_fz + next.target_fz)) / denom;

  Quaternion q_prev = previous.pose.q;
  Quaternion q_curr = current.pose.q;
  Quaternion q_next = next.pose.q;
  if (Dot(q_curr, q_prev) < 0.0) {
    q_prev = Quaternion{-q_prev.x, -q_prev.y, -q_prev.z, -q_prev.w};
  }
  if (Dot(q_curr, q_next) < 0.0) {
    q_next = Quaternion{-q_next.x, -q_next.y, -q_next.z, -q_next.w};
  }
  out.pose.q = Normalize(Quaternion{
      (q_curr.x + factor * (q_prev.x + q_next.x)) / denom,
      (q_curr.y + factor * (q_prev.y + q_next.y)) / denom,
      (q_curr.z + factor * (q_prev.z + q_next.z)) / denom,
      (q_curr.w + factor * (q_prev.w + q_next.w)) / denom,
  });
  return out;
}

std::vector<TrajectorySample> SmoothTrajectorySamples(
    const std::vector<TrajectorySample>& samples,
    double smoothing_factor) {
  if (smoothing_factor <= 0.0 || samples.size() < 3) {
    return samples;
  }

  std::vector<TrajectorySample> smoothed = samples;
  for (std::size_t i = 1; i + 1 < samples.size(); ++i) {
    smoothed[i] = SmoothSample(samples[i - 1], samples[i], samples[i + 1], smoothing_factor);
  }
  return smoothed;
}

TrajectorySample Interpolate(const std::vector<TrajectorySample>& samples, double t) {
  if (t <= samples.front().time_s) {
    return samples.front();
  }
  if (t >= samples.back().time_s) {
    return samples.back();
  }

  auto upper = std::upper_bound(samples.begin(), samples.end(), t, [](double value, const TrajectorySample& sample) {
    return value < sample.time_s;
  });
  const std::size_t right = static_cast<std::size_t>(upper - samples.begin());
  const std::size_t left = right - 1;
  const auto& a = samples[left];
  const auto& b = samples[right];
  const double u = (t - a.time_s) / (b.time_s - a.time_s);
  const auto basis = CubicBsplineBasis(u);

  auto sample_at = [&](long long index) -> const TrajectorySample& {
    const long long last = static_cast<long long>(samples.size() - 1);
    const long long clamped = std::max(0LL, std::min(index, last));
    return samples[static_cast<std::size_t>(clamped)];
  };

  const std::array<const TrajectorySample*, 4> control{
      &sample_at(static_cast<long long>(left) - 1),
      &sample_at(static_cast<long long>(left)),
      &sample_at(static_cast<long long>(left) + 1),
      &sample_at(static_cast<long long>(left) + 2),
  };

  TrajectorySample out;
  out.time_s = t;
  out.pose.p = Vec3{0.0, 0.0, 0.0};
  out.pose.q = Quaternion{0.0, 0.0, 0.0, 0.0};
  out.target_fz = 0.0;

  const Quaternion reference_q = control[1]->pose.q;
  for (std::size_t i = 0; i < control.size(); ++i) {
    Quaternion q = control[i]->pose.q;
    if (Dot(reference_q, q) < 0.0) {
      q = Quaternion{-q.x, -q.y, -q.z, -q.w};
    }
    out.pose.p = out.pose.p + control[i]->pose.p * basis[i];
    out.pose.q.x += q.x * basis[i];
    out.pose.q.y += q.y * basis[i];
    out.pose.q.z += q.z * basis[i];
    out.pose.q.w += q.w * basis[i];
    out.target_fz += control[i]->target_fz * basis[i];
  }
  out.pose.q = Normalize(out.pose.q);
  return out;
}

Matrix4 ArrayToMatrix(const std::array<double, 16>& values) {
  Matrix4 m;
  m.data = values;
  return m;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --robot-ip <ip> --trajectory <csv> [options]\n\n"
      << "Options:\n"
      << "  --mode <relative|absolute>        Default: relative\n"
      << "  --speed-scale <value>             Default: 1.0\n"
      << "  --max-translation-speed <m/s>     Default: 0.03\n"
      << "  --max-translation-acceleration <m/s^2>  Default: 0.01\n"
      << "  --max-rotation-speed <rad/s>      Default: 0.35\n"
      << "  --max-rotation-acceleration <rad/s^2>  Default: 0.1\n"
      << "  --ramp-time <s>                   Default: 2.0\n"
      << "  --hold-at-end                     Keep commanding the final pose instead of exiting\n"
      << "  --enable-force-correction         Disabled by default\n"
      << "  --force-gain <m/N>                Default: 0.0001\n"
      << "  --max-force-correction <m>        Default: 0.002\n"
      << "  --force-sign <1|-1>               Default: 1\n"
      << "  --bspline-smoothing-factor <value> Default: 0.0016\n";
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
    } else if (arg == "--trajectory") {
      opt.trajectory_path = require_value(arg);
    } else if (arg == "--mode") {
      opt.mode = require_value(arg);
    } else if (arg == "--speed-scale") {
      opt.speed_scale = ParseDouble(require_value(arg), arg);
    } else if (arg == "--max-translation-speed") {
      opt.max_translation_speed = ParseDouble(require_value(arg), arg);
    } else if (arg == "--max-translation-acceleration") {
      opt.max_translation_acceleration = ParseDouble(require_value(arg), arg);
    } else if (arg == "--max-rotation-speed") {
      opt.max_rotation_speed = ParseDouble(require_value(arg), arg);
    } else if (arg == "--max-rotation-acceleration") {
      opt.max_rotation_acceleration = ParseDouble(require_value(arg), arg);
    } else if (arg == "--ramp-time") {
      opt.ramp_time_s = ParseDouble(require_value(arg), arg);
    } else if (arg == "--hold-at-end") {
      opt.hold_at_end = true;
    } else if (arg == "--enable-force-correction") {
      opt.enable_force_correction = true;
    } else if (arg == "--force-gain") {
      opt.force_gain = ParseDouble(require_value(arg), arg);
    } else if (arg == "--max-force-correction") {
      opt.max_force_correction = ParseDouble(require_value(arg), arg);
    } else if (arg == "--force-sign") {
      opt.force_sign = ParseDouble(require_value(arg), arg);
    } else if (arg == "--bspline-smoothing-factor") {
      opt.bspline_smoothing_factor = ParseDouble(require_value(arg), arg);
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
  if (opt.trajectory_path.empty()) {
    throw std::runtime_error("--trajectory is required.");
  }
  if (opt.mode != "relative" && opt.mode != "absolute") {
    throw std::runtime_error("--mode must be relative or absolute.");
  }
  if (opt.speed_scale <= 0.0 || opt.speed_scale > 1.0) {
    throw std::runtime_error("--speed-scale must be in (0, 1].");
  }
  if (opt.max_translation_speed <= 0.0) {
    throw std::runtime_error("--max-translation-speed must be positive.");
  }
  if (opt.max_translation_acceleration <= 0.0) {
    throw std::runtime_error("--max-translation-acceleration must be positive.");
  }
  if (opt.max_rotation_speed <= 0.0) {
    throw std::runtime_error("--max-rotation-speed must be positive.");
  }
  if (opt.max_rotation_acceleration <= 0.0) {
    throw std::runtime_error("--max-rotation-acceleration must be positive.");
  }
  if (opt.ramp_time_s < 0.0) {
    throw std::runtime_error("--ramp-time must be non-negative.");
  }
  if (opt.bspline_smoothing_factor < 0.0) {
    throw std::runtime_error("--bspline-smoothing-factor must be non-negative.");
  }
  return opt;
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

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, SignalHandler);

  try {
    const Options opt = ParseArgs(argc, argv);
    const auto raw_trajectory = LoadTrajectoryCsv(opt.trajectory_path);
    const auto trajectory = SmoothTrajectorySamples(raw_trajectory, opt.bspline_smoothing_factor);

    std::cout << "[INFO] Loaded " << trajectory.size() << " trajectory samples from " << opt.trajectory_path << "\n";
    std::cout << "[INFO] Interpolation: local cubic B-spline approximation\n";
    std::cout << "[INFO] B-spline smoothing factor: " << opt.bspline_smoothing_factor << "\n";
    std::cout << "[INFO] Connecting to Franka at " << opt.robot_ip << "\n";

    franka::Robot robot(opt.robot_ip);
    robot.automaticErrorRecovery();
    SetConservativeCollisionBehavior(robot);

    const franka::RobotState initial_state = robot.readOnce();
    const Matrix4 start_pose_matrix = ArrayToMatrix(initial_state.O_T_EE_c);
    Pose commanded_pose = MatrixToPose(start_pose_matrix);
    Vec3 commanded_velocity{0.0, 0.0, 0.0};
    Vec3 commanded_angular_velocity{0.0, 0.0, 0.0};

    const double start_time_s = trajectory.front().time_s;
    const double end_time_s = trajectory.back().time_s;
    double elapsed_s = 0.0;
    double control_elapsed_s = 0.0;
    double finish_settle_elapsed_s = 0.0;
    std::cout << "[INFO] Replay mode: " << opt.mode << "\n";
    std::cout << "[INFO] Duration after speed scaling: " << (end_time_s - start_time_s) / opt.speed_scale << " s\n";
    std::cout << "[INFO] Startup ramp time: " << opt.ramp_time_s << " s\n";
    std::cout << "[INFO] Press Ctrl+C to stop.\n";

    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose {
      const double dt = period.toSec();
      control_elapsed_s += dt;
      elapsed_s += dt * opt.speed_scale;

      const double trajectory_time_s = std::min(start_time_s + elapsed_s, end_time_s);
      const TrajectorySample sample = Interpolate(trajectory, trajectory_time_s);

      Matrix4 target_matrix = PoseToMatrix(sample.pose);
      if (opt.mode == "relative") {
        target_matrix = Multiply(start_pose_matrix, target_matrix);
      }

      Pose target_pose = MatrixToPose(target_matrix);
      const bool stop_requested = g_stop_requested.load();
      if (stop_requested) {
        target_pose = commanded_pose;
      } else if (opt.enable_force_correction) {
        const double measured_fz = opt.force_sign * state.O_F_ext_hat_K[2];
        const double force_error = sample.target_fz - measured_fz;
        const double correction = Clamp(opt.force_gain * force_error, -opt.max_force_correction, opt.max_force_correction);
        const Vec3 tool_z{target_matrix(0, 2), target_matrix(1, 2), target_matrix(2, 2)};
        target_pose.p = target_pose.p + tool_z * correction;
      }

      const double ramp_factor = opt.ramp_time_s > 1e-9 ? Smoothstep(control_elapsed_s / opt.ramp_time_s) : 1.0;
      const double max_translation_speed = opt.max_translation_speed * opt.speed_scale * ramp_factor;
      const double max_translation_acceleration = opt.max_translation_acceleration * opt.speed_scale;
      const double max_rotation_speed = opt.max_rotation_speed * opt.speed_scale * ramp_factor;
      const double max_rotation_acceleration = opt.max_rotation_acceleration * opt.speed_scale;

      const Vec3 translation_error = target_pose.p - commanded_pose.p;
      const double translation_error_norm = Norm(translation_error);
      const double stopping_speed = std::sqrt(2.0 * max_translation_acceleration * translation_error_norm);
      const double desired_speed_limit = std::min(max_translation_speed, stopping_speed);
      const Vec3 desired_velocity = LimitVectorNorm(
          translation_error / std::max(dt, 1e-9),
          desired_speed_limit);
      const Vec3 velocity_delta = LimitVectorNorm(
          desired_velocity - commanded_velocity,
          max_translation_acceleration * dt);
      commanded_velocity = LimitVectorNorm(commanded_velocity + velocity_delta, max_translation_speed);

      const Vec3 translation_step = commanded_velocity * dt;
      if (translation_error_norm < 1e-7 && Norm(commanded_velocity) < 1e-5) {
        commanded_pose.p = target_pose.p;
        commanded_velocity = Vec3{0.0, 0.0, 0.0};
      } else {
        commanded_pose.p = commanded_pose.p + translation_step;
      }
      Quaternion rotation_error_q = Multiply(target_pose.q, Conjugate(commanded_pose.q));
      Vec3 rotation_error = QuaternionToRotationVector(rotation_error_q);
      const double rotation_error_norm = Norm(rotation_error);
      const double stopping_angular_speed = std::sqrt(2.0 * max_rotation_acceleration * rotation_error_norm);
      const double desired_angular_speed_limit = std::min(max_rotation_speed, stopping_angular_speed);
      const Vec3 desired_angular_velocity = LimitVectorNorm(
          rotation_error / std::max(dt, 1e-9),
          desired_angular_speed_limit);
      const Vec3 angular_velocity_delta = LimitVectorNorm(
          desired_angular_velocity - commanded_angular_velocity,
          max_rotation_acceleration * dt);
      commanded_angular_velocity = LimitVectorNorm(
          commanded_angular_velocity + angular_velocity_delta,
          max_rotation_speed);

      if (rotation_error_norm < 1e-7 && Norm(commanded_angular_velocity) < 1e-5) {
        commanded_pose.q = Normalize(target_pose.q);
        commanded_angular_velocity = Vec3{0.0, 0.0, 0.0};
      } else {
        const Quaternion rotation_step = RotationVectorToQuaternion(commanded_angular_velocity * dt);
        commanded_pose.q = Multiply(rotation_step, commanded_pose.q);
      }

      const Matrix4 command_matrix = PoseToMatrix(commanded_pose);

      const bool trajectory_time_done = trajectory_time_s >= end_time_s;
      const bool command_reached =
          Norm(target_pose.p - commanded_pose.p) < 1e-5 &&
          QuaternionAngle(target_pose.q, commanded_pose.q) < 1e-4;
      const bool command_stopped = Norm(commanded_velocity) < 1e-5;
      const bool rotation_stopped = Norm(commanded_angular_velocity) < 1e-5;

      if (trajectory_time_done && command_reached && command_stopped && rotation_stopped) {
        finish_settle_elapsed_s += dt;
      } else {
        finish_settle_elapsed_s = 0.0;
      }

      if (stop_requested && command_stopped && rotation_stopped) {
        return franka::MotionFinished(franka::CartesianPose(MatrixToArray(command_matrix)));
      }

      if (finish_settle_elapsed_s > 2.0) {
        if (opt.hold_at_end) {
          return franka::CartesianPose(MatrixToArray(command_matrix));
        }
        return franka::MotionFinished(franka::CartesianPose(MatrixToArray(command_matrix)));
      }

      return franka::CartesianPose(MatrixToArray(command_matrix));
    });

    std::cout << "[INFO] Replay controller stopped.\n";
    return 0;
  } catch (const franka::Exception& e) {
    std::cerr << "[FRANKA ERROR] " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}






