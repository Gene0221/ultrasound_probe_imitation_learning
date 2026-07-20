#include <franka/cartesian_pose.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
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
  double max_rotation_speed = 0.35;
  bool enable_force_correction = false;
  double force_gain = 0.0001;
  double max_force_correction = 0.002;
  double force_sign = 1.0;
};

double Clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(value, hi));
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

double Norm(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 LimitStep(const Vec3& current, const Vec3& target, double max_step) {
  const Vec3 delta = target - current;
  const double length = Norm(delta);
  if (length <= max_step || length < 1e-12) {
    return target;
  }
  return current + delta * (max_step / length);
}

Quaternion Normalize(const Quaternion& q) {
  const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n < 1e-12) {
    throw std::runtime_error("Quaternion norm is zero.");
  }
  return Quaternion{q.x / n, q.y / n, q.z / n, q.w / n};
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

double QuaternionAngle(const Quaternion& a, const Quaternion& b) {
  const double cos_theta = std::fabs(Dot(Normalize(a), Normalize(b)));
  return 2.0 * std::acos(Clamp(cos_theta, -1.0, 1.0));
}

Quaternion LimitRotationStep(const Quaternion& current, const Quaternion& target, double max_angle_step) {
  const double angle = QuaternionAngle(current, target);
  if (angle <= max_angle_step || angle < 1e-12) {
    return Normalize(target);
  }
  return Slerp(current, target, max_angle_step / angle);
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

double ParseDouble(const std::string& text, const std::string& field_name) {
  try {
    std::size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size()) {
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
  const auto& b = *upper;
  const auto& a = *(upper - 1);
  const double alpha = (t - a.time_s) / (b.time_s - a.time_s);

  TrajectorySample out;
  out.time_s = t;
  out.pose.p = a.pose.p * (1.0 - alpha) + b.pose.p * alpha;
  out.pose.q = Slerp(a.pose.q, b.pose.q, alpha);
  out.target_fz = a.target_fz * (1.0 - alpha) + b.target_fz * alpha;
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
      << "  --max-rotation-speed <rad/s>      Default: 0.35\n"
      << "  --enable-force-correction         Disabled by default\n"
      << "  --force-gain <m/N>                Default: 0.0001\n"
      << "  --max-force-correction <m>        Default: 0.002\n"
      << "  --force-sign <1|-1>               Default: 1\n";
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
    } else if (arg == "--max-rotation-speed") {
      opt.max_rotation_speed = ParseDouble(require_value(arg), arg);
    } else if (arg == "--enable-force-correction") {
      opt.enable_force_correction = true;
    } else if (arg == "--force-gain") {
      opt.force_gain = ParseDouble(require_value(arg), arg);
    } else if (arg == "--max-force-correction") {
      opt.max_force_correction = ParseDouble(require_value(arg), arg);
    } else if (arg == "--force-sign") {
      opt.force_sign = ParseDouble(require_value(arg), arg);
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
    const auto trajectory = LoadTrajectoryCsv(opt.trajectory_path);

    std::cout << "[INFO] Loaded " << trajectory.size() << " trajectory samples from " << opt.trajectory_path << "\n";
    std::cout << "[INFO] Connecting to Franka at " << opt.robot_ip << "\n";

    franka::Robot robot(opt.robot_ip);
    robot.automaticErrorRecovery();
    SetConservativeCollisionBehavior(robot);

    const franka::RobotState initial_state = robot.readOnce();
    const Matrix4 start_pose_matrix = ArrayToMatrix(initial_state.O_T_EE);
    Pose commanded_pose = MatrixToPose(start_pose_matrix);

    const double start_time_s = trajectory.front().time_s;
    const double end_time_s = trajectory.back().time_s;
    double elapsed_s = 0.0;
    std::cout << "[INFO] Replay mode: " << opt.mode << "\n";
    std::cout << "[INFO] Duration after speed scaling: " << (end_time_s - start_time_s) / opt.speed_scale << " s\n";
    std::cout << "[INFO] Press Ctrl+C to stop.\n";

    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose {
      const double dt = period.toSec();
      elapsed_s += dt * opt.speed_scale;

      const double trajectory_time_s = std::min(start_time_s + elapsed_s, end_time_s);
      const TrajectorySample sample = Interpolate(trajectory, trajectory_time_s);

      Matrix4 target_matrix = PoseToMatrix(sample.pose);
      if (opt.mode == "relative") {
        target_matrix = Multiply(start_pose_matrix, target_matrix);
      }

      Pose target_pose = MatrixToPose(target_matrix);

      if (opt.enable_force_correction) {
        const double measured_fz = opt.force_sign * state.O_F_ext_hat_K[2];
        const double force_error = sample.target_fz - measured_fz;
        const double correction = Clamp(opt.force_gain * force_error, -opt.max_force_correction, opt.max_force_correction);
        const Vec3 tool_z{target_matrix(0, 2), target_matrix(1, 2), target_matrix(2, 2)};
        target_pose.p = target_pose.p + tool_z * correction;
      }

      const double max_translation_step = opt.max_translation_speed * opt.speed_scale * dt;
      const double max_rotation_step = opt.max_rotation_speed * opt.speed_scale * dt;
      commanded_pose.p = LimitStep(commanded_pose.p, target_pose.p, max_translation_step);
      commanded_pose.q = LimitRotationStep(commanded_pose.q, target_pose.q, max_rotation_step);

      const Matrix4 command_matrix = PoseToMatrix(commanded_pose);

      const bool trajectory_time_done = trajectory_time_s >= end_time_s;
      const bool command_reached =
          Norm(target_pose.p - commanded_pose.p) < 1e-5 &&
          QuaternionAngle(target_pose.q, commanded_pose.q) < 1e-4;

      if (g_stop_requested.load() || (trajectory_time_done && command_reached)) {
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


