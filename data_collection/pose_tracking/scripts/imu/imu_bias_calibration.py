import numpy as np
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.imu.imu_reader import IMUReader

CALIBRATION_DIR = PROJECT_ROOT / "data" / "imu" / "calibration"
BIAS_PATH = CALIBRATION_DIR / "imu_bias.npy"

# 采样数量，5000 条在 200Hz 下约 25 秒
NUM_SAMPLES = 5000

# 确保输出目录存在
os.makedirs(CALIBRATION_DIR, exist_ok=True)

imu = IMUReader()

gyro_list = []
accel_list = []

print("=" * 45)
print("  IMU Bias 标定")
print("  请将 D435i 静置在水平面上，勿触碰")
print(f"  采集 {NUM_SAMPLES} 条数据后自动结束")
print("=" * 45)

try:

    while len(gyro_list) < NUM_SAMPLES:

        # ── 适配修改后的 get_data，返回三元组 ──
        result = imu.get_data()

        if result is None:
            continue

        accel, gyro, dt = result

        # 首帧 dt 为 None 时跳过
        if dt is None:
            continue

        accel_list.append(accel.copy())
        gyro_list.append(gyro.copy())

        if len(gyro_list) % 100 == 0:
            print(f"  采集进度: {len(gyro_list)}/{NUM_SAMPLES}")

finally:

    imu.stop()

if len(accel_list) < NUM_SAMPLES:
    print(f"[警告] 仅采集到 {len(accel_list)} 条，标定可能不准确")

accel_arr = np.array(accel_list)  # shape (N, 3)
gyro_arr = np.array(gyro_list)  # shape (N, 3)

# ── 计算均值（即静止时各轴读数，含重力） ──
accel_bias = np.mean(accel_arr, axis=0)
gyro_bias = np.mean(gyro_arr, axis=0)

# ── 计算标准差（衡量噪声水平） ──
accel_std = np.std(accel_arr, axis=0)
gyro_std = np.std(gyro_arr, axis=0)

bias = {
    "accel_bias": accel_bias,  # 静止均值（含重力），main.py 中会自动分离零偏
    "gyro_bias": gyro_bias,  # 静止陀螺仪均值（理论应接近 [0,0,0]）
}

np.save(BIAS_PATH, bias)

print("\n" + "=" * 45)
print("  标定完成，结果已保存至 data/imu/calibration/imu_bias.npy")
print("=" * 45)
print(f"  Accel 均值 (m/s²): {np.round(accel_bias, 4)}")
print(f"  Accel 标准差:      {np.round(accel_std, 6)}")
print(f"  重力模长:          {np.linalg.norm(accel_bias):.4f} m/s²"
      f"  (理论值 9.8067)")
print(f"  Gyro  均值 (rad/s):{np.round(gyro_bias, 6)}")
print(f"  Gyro  标准差:      {np.round(gyro_std, 6)}")

# 质量检验
gravity_err = abs(np.linalg.norm(accel_bias) - 9.8067)
if gravity_err < 0.12:
    print("\n  ✅ 重力模长正常，标定质量良好")
else:
    print(f"\n  ⚠️  重力模长偏差 {gravity_err:.3f} m/s²，"
          "请检查设备是否完全静止或重新标定")
