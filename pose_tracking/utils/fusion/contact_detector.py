"""
contact_detector.py
探头接触状态检测模块
========================================================
输入：IMU 数据（加速度计）+ RGB 帧
输出：ContactState 枚举（未接触/轻接触/压紧/滑动）

检测依据：
  1. IMU 加速度突变     → 接触瞬间冲击
  2. IMU 加速度方差     → 静止=压紧，抖动=滑动
  3. 光流均值           → 图像运动量（接触后趋近于零）
  4. 图像清晰度         → 贴近皮肤时过曝/模糊

"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict


# ══════════════════════════════════════════════
#  接触状态枚举
# ══════════════════════════════════════════════

class ContactState(Enum):
    UNKNOWN = 0  # 初始化中，数据不足
    FREE = 1  # 未接触（悬空）
    TOUCHING = 2  # 轻接触
    PRESSED = 3  # 压紧接触
    SLIDING = 4  # 接触中滑动


# ══════════════════════════════════════════════
#  检测结果数据类
# ══════════════════════════════════════════════

@dataclass
class ContactInfo:
    state: ContactState
    accel_norm: float  # 当前加速度模长 m/s²
    accel_std: float  # 加速度短窗口标准差（抖动指标）
    accel_delta: float  # 加速度突变量（冲击指标）
    optical_flow: float  # 光流均值（图像运动量 px/frame）
    blur_score: float  # 图像清晰度（Laplacian方差）
    confidence: float  # 综合置信度 0~1


# ══════════════════════════════════════════════
#  各子检测器
# ══════════════════════════════════════════════

class ImuContactDetector:
    """
    基于 IMU 加速度计的接触检测。
    维护短时窗口，分析：
      - 突变（冲击）：探头碰到皮肤的瞬间
      - 方差（稳定性）：压紧时稳定，滑动时抖动
      - 模长偏离重力：有外力时模长偏离 9.807
    """

    def __init__(self,
                 window_size: int = 20,
                 impact_threshold: float = 1.5,
                 std_pressed: float = 0.08,
                 std_sliding: float = 0.25):
        """
        window_size     : 滑动窗口帧数（250Hz 下约 80ms）
        impact_threshold: 加速度突变阈值 m/s²，超过视为冲击
        std_pressed     : 方差低于此值视为压紧
        std_sliding     : 方差高于此值视为滑动
        """
        self.window = deque(maxlen=window_size)
        self.impact_th = impact_threshold
        self.std_pressed = std_pressed
        self.std_sliding = std_sliding
        self._prev_norm = None

    def update(self, accel: np.ndarray
               ) -> Tuple[float, float, float]:
        """
        返回 (accel_norm, accel_std, accel_delta)
        """
        norm = float(np.linalg.norm(accel))
        self.window.append(norm)

        # 突变量（与上一帧的差）
        delta = abs(norm - self._prev_norm) if self._prev_norm is not None else 0.0
        self._prev_norm = norm

        # 短窗口标准差
        std = float(np.std(self.window)) if len(self.window) >= 5 else 0.0

        return norm, std, delta


class OpticalFlowDetector:
    """
    稀疏光流（Lucas-Kanade）检测图像运动量。
    接触压紧后图像静止，光流均值接近 0。
    滑动时光流方向一致且均值较大。
    """

    def __init__(self,
                 max_corners: int = 80,
                 flow_free_th: float = 3.0,
                 flow_contact_th: float = 0.8):
        """
        max_corners    : 追踪的最大特征点数
        flow_free_th   : 光流均值超过此值视为自由运动
        flow_contact_th: 光流均值低于此值视为静止（接触）
        """
        self.max_corners = max_corners
        self.flow_free_th = flow_free_th
        self.flow_contact_th = flow_contact_th

        self._prev_gray = None
        self._prev_pts = None

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

    def update(self, frame: np.ndarray) -> float:
        """
        输入 BGR 帧，返回光流均值（px/frame）。
        首帧返回 -1（数据不足）。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 10:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, **self.feature_params)
            return -1.0

        # 计算光流
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_pts, None,
            **self.lk_params
        )

        if next_pts is None or status is None:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, **self.feature_params)
            return -1.0

        # 只保留追踪成功的点
        good_prev = self._prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) < 5:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, **self.feature_params)
            return -1.0

        # 各点位移模长均值
        displacements = np.linalg.norm(
            good_next - good_prev, axis=1)
        flow_mean = float(np.mean(displacements))

        # 每隔一段时间重新检测特征点
        self._prev_gray = gray
        if len(good_next) < self.max_corners // 2:
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, **self.feature_params)
        else:
            self._prev_pts = good_next.reshape(-1, 1, 2)

        return flow_mean


class BlurDetector:
    """
    图像清晰度检测（Laplacian 方差）。
    探头贴近皮肤时图像模糊（焦距外）或过曝，清晰度急剧下降。
    """

    def __init__(self,
                 blur_threshold: float = 40.0,
                 window_size: int = 10):
        self.blur_th = blur_threshold
        self.window = deque(maxlen=window_size)

    def update(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.window.append(score)
        # 返回短窗口均值（更稳定）
        return float(np.mean(self.window))


# ══════════════════════════════════════════════
#  融合检测器（主模块）
# ══════════════════════════════════════════════

class ContactDetector:
    """
    融合 IMU + 光流 + 图像清晰度的接触状态检测器。

    使用方式：
        detector = ContactDetector()

        # 主循环中
        state, info = detector.update(accel, frame)

        # 只用 IMU（无 RGB 帧时）
        state, info = detector.update(accel, frame=None)
    """

    def __init__(self,
                 # IMU 参数
                 imu_window: int = 20,
                 impact_threshold: float = 1.5,
                 std_pressed: float = 0.08,
                 std_sliding: float = 0.25,
                 # 光流参数
                 flow_free_th: float = 3.0,
                 flow_contact_th: float = 0.8,
                 # 清晰度参数
                 blur_threshold: float = 40.0,
                 # 状态平滑（防止状态抖动）
                 smooth_window: int = 8):

        self.imu_det = ImuContactDetector(
            imu_window, impact_threshold, std_pressed, std_sliding)
        self.flow_det = OpticalFlowDetector(
            flow_free_th=flow_free_th,
            flow_contact_th=flow_contact_th)
        self.blur_det = BlurDetector(blur_threshold)

        # 状态平滑窗口（投票）
        self.state_window = deque(maxlen=smooth_window)
        self._frame_count = 0

    def update(self,
               accel: np.ndarray,
               frame: Optional[np.ndarray] = None
               ) -> Tuple[ContactState, ContactInfo]:
        """
        accel : np.array([x, y, z])  m/s²（已去偏置，保留重力）
        frame : BGR 图像（可为 None，此时只用 IMU）
        """
        self._frame_count += 1

        # ── 1. IMU 特征提取 ───────────────────
        accel_norm, accel_std, accel_delta = \
            self.imu_det.update(accel)

        # ── 2. 光流特征提取 ───────────────────
        flow_mean = self.flow_det.update(frame) \
            if frame is not None else -1.0

        # ── 3. 清晰度特征提取 ─────────────────
        blur_score = self.blur_det.update(frame) \
            if frame is not None else -1.0

        # ── 4. 数据不足时返回 UNKNOWN ─────────
        if self._frame_count < 10:
            info = ContactInfo(
                state=ContactState.UNKNOWN,
                accel_norm=accel_norm,
                accel_std=accel_std,
                accel_delta=accel_delta,
                optical_flow=flow_mean,
                blur_score=blur_score,
                confidence=0.0
            )
            return ContactState.UNKNOWN, info

        # ── 5. 状态判断逻辑 ───────────────────
        state, confidence = self._classify(
            accel_norm, accel_std, accel_delta,
            flow_mean, blur_score)

        # ── 6. 平滑（多数投票） ───────────────
        self.state_window.append(state)
        smoothed = self._majority_vote()

        info = ContactInfo(
            state=smoothed,
            accel_norm=accel_norm,
            accel_std=accel_std,
            accel_delta=accel_delta,
            optical_flow=flow_mean,
            blur_score=blur_score,
            confidence=confidence
        )
        return smoothed, info

    def _classify(self,
                  accel_norm: float,
                  accel_std: float,
                  accel_delta: float,
                  flow_mean: float,
                  blur_score: float
                  ) -> Tuple[ContactState, float]:
        """
        规则融合分类器。
        返回 (ContactState, confidence)
        """
        imu_det = self.imu_det
        flow_det = self.flow_det

        # ── IMU 信号 ──────────────────────────
        # 重力偏离量：接触时探头受到皮肤支持力，模长偏离 9.807
        gravity_deviation = abs(accel_norm - 9.80665)

        imu_is_stable = accel_std < imu_det.std_pressed  # 加速度稳定
        imu_is_moving = accel_std > imu_det.std_sliding  # 加速度抖动
        imu_has_impact = accel_delta > imu_det.impact_th  # 冲击突变
        imu_has_force = gravity_deviation > 0.5  # 受到外力

        # ── 光流信号 ──────────────────────────
        has_flow = flow_mean >= 0  # 有有效光流
        flow_still = has_flow and flow_mean < flow_det.flow_contact_th
        flow_free = has_flow and flow_mean > flow_det.flow_free_th

        # ── 清晰度信号 ────────────────────────
        has_blur = blur_score >= 0
        is_blurry = has_blur and blur_score < self.blur_det.blur_th

        # ══════════════════════════════════════
        #  分类规则（优先级由高到低）
        # ══════════════════════════════════════

        # SLIDING：接触中但有相对滑动
        # 条件：加速度方差大 且 (光流不静止 或 受到外力)
        if imu_is_moving and (not flow_still) and imu_has_force:
            conf = min(1.0, accel_std / imu_det.std_sliding)
            return ContactState.SLIDING, conf

        # PRESSED：压紧接触
        # 条件：加速度稳定 且 图像静止（或模糊） 且 有外力迹象
        if imu_is_stable and (flow_still or is_blurry) and imu_has_force:
            conf_imu = 1.0 - accel_std / imu_det.std_pressed
            conf_flow = (1.0 - flow_mean / flow_det.flow_contact_th
                         if flow_still else 0.5)
            conf = min(1.0, (conf_imu + conf_flow) / 2)
            return ContactState.PRESSED, conf

        # TOUCHING：轻接触
        # 条件：有冲击 或 (受力 且 加速度较稳定)
        if imu_has_impact or (imu_has_force and not imu_is_moving):
            conf = min(1.0, max(
                accel_delta / imu_det.impact_th,
                gravity_deviation / 1.0
            ))
            return ContactState.TOUCHING, conf

        # FREE：未接触
        # 条件：无外力 且 (光流自由 或 图像清晰)
        if not imu_has_force and (flow_free or (has_blur and not is_blurry)):
            conf_imu = max(0.0, 1.0 - gravity_deviation / 0.5)
            conf_flow = min(1.0, flow_mean / flow_det.flow_free_th) \
                if flow_free else 0.3
            conf = min(1.0, (conf_imu + conf_flow) / 2)
            return ContactState.FREE, conf

        # 默认：数据模糊，维持当前窗口多数状态
        return ContactState.TOUCHING, 0.3

    def _majority_vote(self) -> ContactState:
        """对状态窗口做多数投票，减少状态抖动"""
        if not self.state_window:
            return ContactState.UNKNOWN
        counts: Dict[ContactState, int] = {}
        for s in self.state_window:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)


# ══════════════════════════════════════════════
#  可视化叠加工具
# ══════════════════════════════════════════════

STATE_COLORS = {
    ContactState.UNKNOWN: (128, 128, 128),
    ContactState.FREE: (0, 200, 0),
    ContactState.TOUCHING: (0, 200, 255),
    ContactState.PRESSED: (0, 80, 255),
    ContactState.SLIDING: (0, 165, 255),
}

STATE_LABELS = {
    ContactState.UNKNOWN: "UNKNOWN",
    ContactState.FREE: "FREE",
    ContactState.TOUCHING: "TOUCHING",
    ContactState.PRESSED: "PRESSED",
    ContactState.SLIDING: "SLIDING",
}


def draw_contact_overlay(frame: np.ndarray,
                         info: ContactInfo) -> np.ndarray:
    """
    在 BGR 帧上叠加接触状态信息，返回新帧。
    """
    vis = frame.copy()
    state = info.state
    color = STATE_COLORS[state]
    label = STATE_LABELS[state]

    # 顶部状态栏
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (20, 20, 20), -1)
    cv2.putText(vis, f"Contact: {label}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    # 置信度条
    bar_w = int(info.confidence * 200)
    cv2.rectangle(vis, (vis.shape[1] - 210, 10),
                  (vis.shape[1] - 10, 40), (60, 60, 60), -1)
    cv2.rectangle(vis, (vis.shape[1] - 210, 10),
                  (vis.shape[1] - 210 + bar_w, 40), color, -1)
    cv2.putText(vis, f"{info.confidence:.0%}",
                (vis.shape[1] - 200, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 底部指标
    metrics = [
        f"Accel: {info.accel_norm:.2f} m/s2  std={info.accel_std:.3f}",
        f"Flow:  {info.optical_flow:.2f} px   blur={info.blur_score:.0f}",
    ]
    y = vis.shape[0] - 45
    cv2.rectangle(vis, (0, y - 5),
                  (vis.shape[1], vis.shape[0]), (20, 20, 20), -1)
    for line in metrics:
        cv2.putText(vis, line, (8, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22

    return vis