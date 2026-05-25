import pyrealsense2 as rs
import numpy as np


class IMUReader:

    def __init__(self):

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 指定频率，accel 200Hz，gyro 400Hz
        self.config.enable_stream(
            rs.stream.accel, rs.format.motion_xyz32f, 250)
        self.config.enable_stream(
            rs.stream.gyro, rs.format.motion_xyz32f, 400)

        self.pipeline.start(self.config)
        self._is_started = True

        # 上一帧硬件时间戳（ms），用于计算真实 dt
        self._last_ts_ms: float = None

    def get_data(self):
        """
        返回 (accel_data, gyro_data, dt)
          accel_data : np.array([x, y, z])  单位 m/s²（含重力）
          gyro_data  : np.array([x, y, z])  单位 rad/s  ← 修正：不再乘 180/π
          dt         : float 秒，由硬件时间戳差值计算，首帧返回 None
        """
        frames = self.pipeline.wait_for_frames()

        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if not accel_frame or not gyro_frame:
            return None

        # ── 硬件时间戳（ms） ──────────────────────────
        ts_ms = gyro_frame.get_timestamp()

        if self._last_ts_ms is None:
            dt = None
        else:
            dt = (ts_ms - self._last_ts_ms) / 1000.0  # 转换为秒
            # 异常帧保护：dt 不合理时跳过（< 0.5ms 或 > 100ms）
            if dt < 0.0005 or dt > 0.1:
                self._last_ts_ms = ts_ms
                return None

        self._last_ts_ms = ts_ms

        # ── 加速度计（m/s²，保留重力分量，不做处理） ──
        accel = accel_frame.as_motion_frame().get_motion_data()
        accel_data = np.array([accel.x, accel.y, accel.z], dtype=np.float64)

        # ── 陀螺仪（rad/s）
        gyro = gyro_frame.as_motion_frame().get_motion_data()
        gyro_data = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float64)

        return accel_data, gyro_data, dt

    def stop(self):
        if self._is_started:
            self.pipeline.stop()
            self._is_started = False