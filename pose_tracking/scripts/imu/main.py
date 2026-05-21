import time
import json
import numpy as np
import sys
from pathlib import Path

from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.imu.imu_reader import IMUReader
from utils.imu.attitude_viewer import AttitudeViewer
from utils.imu.madgwick_filter import MadgwickFilter

BIAS_PATH = PROJECT_ROOT / "data" / "imu" / "calibration" / "imu_bias.npy"
LOG_PATH = PROJECT_ROOT / "data" / "imu" / "logs" / "imu_quaternion_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# 加载 IMU Bias
# =========================================================

bias = np.load(
    BIAS_PATH,
    allow_pickle=True
).item()

accel_bias = bias["accel_bias"]
gyro_bias  = bias["gyro_bias"]

# =========================================================
# 初始化 IMU
# =========================================================

imu = IMUReader()

madgwick = MadgwickFilter(
    beta=0.03
)

# =========================================================
# 可视化
# =========================================================

viewer = AttitudeViewer()
viewer.start()

# =========================================================
# JSONL 日志文件
# =========================================================

json_file = open(
    LOG_PATH,
    "w",
    encoding="utf-8"
)

# =========================================================
# 初始姿态参考
# 用于：
# “当前静止姿态”作为零位
# =========================================================

initial_q = None

print("=" * 60)
print(" D435i IMU Quaternion Tracking Started")
print(" 当前静止姿态将作为世界坐标零位")
print("=" * 60)

try:

    while True:

        # =================================================
        # 获取 IMU 数据
        # =================================================

        imu_result = imu.get_data()

        if imu_result is None:
            continue

        accel, gyro, dt = imu_result

        if dt is None:
            continue

        # =================================================
        # Bias 修正
        # =================================================

        accel_corrected = accel - accel_bias
        gyro_corrected  = gyro  - gyro_bias

        # =================================================
        # Madgwick 更新
        # =================================================

        madgwick.update(
            gyro_corrected,
            accel_corrected,
            dt
        )

        # =================================================
        # 当前四元数
        # q = [w, x, y, z]
        # =================================================

        q = madgwick.q.copy()

        # =================================================
        # 初始化参考姿态
        # =================================================

        if initial_q is None:

            initial_q = q.copy()

            print("\nInitial Pose Captured")
            print("当前姿态已设为世界坐标零位\n")

        # =================================================
        # scipy Rotation
        # scipy格式:
        # [x, y, z, w]
        # =================================================

        r0 = R.from_quat([
            initial_q[1],
            initial_q[2],
            initial_q[3],
            initial_q[0]
        ])

        r1 = R.from_quat([
            q[1],
            q[2],
            q[3],
            q[0]
        ])

        # =================================================
        # 相对姿态
        # q_relative = q0^-1 × q
        # =================================================

        relative_rot = r0.inv() * r1

        relative_q = relative_rot.as_quat()

        # =================================================
        # 转回:
        # [w, x, y, z]
        # =================================================

        viewer_q = np.array([
            relative_q[3],
            relative_q[0],
            relative_q[1],
            relative_q[2]
        ])

        # =================================================
        # 如果某轴方向相反
        # 在这里修正
        # =================================================

        # Roll方向反了时启用
        # viewer_q[1] *= -1

        # Pitch方向反了时启用
        # viewer_q[2] *= -1

        # Yaw方向反了时启用
        # viewer_q[3] *= -1

        # =================================================
        # Quaternion 分量
        # =================================================

        qw = float(viewer_q[0])
        qx = float(viewer_q[1])
        qy = float(viewer_q[2])
        qz = float(viewer_q[3])

        # =================================================
        # 主机时间戳
        # =================================================

        timestamp = time.time()

        # =================================================
        # JSON记录
        # =================================================

        record = {

            "timestamp": timestamp,

            "quaternion": {

                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz
            }
        }

        json_file.write(
            json.dumps(record) + "\n"
        )

        json_file.flush()

        # =================================================
        # OpenGL 可视化
        # =================================================

        viewer.update_quaternion(
            viewer_q
        )

        # =================================================
        # 输出
        # =================================================

        print(
            f"t={timestamp:.3f} | "
            f"q=[{qw:+.5f}, "
            f"{qx:+.5f}, "
            f"{qy:+.5f}, "
            f"{qz:+.5f}] | "
            f"dt={dt*1000:.1f}ms"
        )

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    imu.stop()

    viewer.stop()

    json_file.close()

    print("\nSaved:")
    print(str(LOG_PATH))
