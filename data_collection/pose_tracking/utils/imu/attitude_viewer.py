"""
visualization/attitude_viewer.py
3D 姿态可视化：独立线程 OpenGL 窗口，实时显示 Roll/Pitch/Yaw
依赖：pip install PyOpenGL PyOpenGL_accelerate pygame
"""

import threading
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as R


class AttitudeViewer:
    """
    独立线程运行的 OpenGL 姿态可视化窗口。
    主程序调用 update(roll, pitch, yaw) 更新姿态，不阻塞主线程。

    使用方式：
        viewer = AttitudeViewer()
        viewer.start()
        viewer.update(roll, pitch, yaw)   # 在主循环中调用
        viewer.stop()
    """

    def __init__(self, width=600, height=600,
                 title="D435i Attitude Viewer"):
        self.width   = width
        self.height  = height
        self.title   = title

        # 四元数 [w, x, y, z]
        self._q = np.array([
            1.0,
            0.0,
            0.0,
            0.0
        ])
        self._lock  = threading.Lock()

        self._running = False
        self._thread  = threading.Thread(
            target=self._run, daemon=True)

    # ── 外部接口 ──────────────────────────────

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)

    def update_quaternion(self, q):
        """
        主线程调用：
        更新四元数姿态

        q = [w, x, y, z]
        """

        with self._lock:
            self._q = q.copy()

    def get_quaternion(self):

        with self._lock:
            return self._q.copy()

    # ── OpenGL 初始化 ─────────────────────────

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(3.5, 2.5, 3.5,   # 相机位置
                  0.0, 0.0, 0.0,   # 看向原点
                  0.0, 1.0, 0.0)   # 上方向

    # ── 绘制立方体 ────────────────────────────

    def _draw_cube(self):
        """绘制带面颜色的半透明立方体，模拟探头形状"""
        w, h, d = 1.2, 0.3, 0.6   # 宽/高/深，模拟扁平探头形状

        vertices = np.array([
            [-w, -h, -d], [ w, -h, -d],
            [ w,  h, -d], [-w,  h, -d],
            [-w, -h,  d], [ w, -h,  d],
            [ w,  h,  d], [-w,  h,  d],
        ]) * 0.5

        # 六个面及颜色（RGBA）
        faces = [
            ([4,5,6,7], (0.2, 0.6, 1.0, 0.7)),   # 前面  蓝
            ([0,1,2,3], (0.2, 0.4, 0.8, 0.7)),   # 后面  深蓝
            ([0,4,7,3], (0.2, 0.8, 0.4, 0.7)),   # 左面  绿
            ([1,5,6,2], (0.2, 0.7, 0.3, 0.7)),   # 右面  深绿
            ([3,2,6,7], (0.9, 0.7, 0.1, 0.7)),   # 上面  黄（镜头侧）
            ([0,1,5,4], (0.6, 0.6, 0.6, 0.7)),   # 下面  灰
        ]

        glBegin(GL_QUADS)
        for idx_list, color in faces:
            glColor4f(*color)
            for i in idx_list:
                glVertex3fv(vertices[i])
        glEnd()

        # 边框
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        glColor4f(1.0, 1.0, 1.0, 0.9)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        for a, b in edges:
            glVertex3fv(vertices[a])
            glVertex3fv(vertices[b])
        glEnd()

        # 镜头圆圈（上面中心）
        glColor4f(0.1, 0.1, 0.1, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        r_lens = 0.12
        y_top  = h * 0.5 + 0.001
        for i in range(32):
            angle = 2 * np.pi * i / 32
            glVertex3f(r_lens * np.cos(angle),
                       y_top,
                       r_lens * np.sin(angle))
        glEnd()

    # ── 绘制坐标轴 ────────────────────────────

    def _draw_axes(self, length=1.0):
        """绘制 X(红) Y(绿) Z(蓝) 坐标轴及标签"""
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0,0,0); glVertex3f(length, 0, 0)
        glColor3f(0, 1, 0); glVertex3f(0,0,0); glVertex3f(0, length, 0)
        glColor3f(0, 0, 1); glVertex3f(0,0,0); glVertex3f(0, 0, length)
        glEnd()

    # ── 绘制参考地平线网格 ────────────────────

    def _draw_grid(self, size=2, step=0.5):
        glColor4f(0.4, 0.4, 0.4, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        x = -size
        while x <= size + 0.001:
            glVertex3f(x, -0.8, -size)
            glVertex3f(x, -0.8,  size)
            glVertex3f(-size, -0.8, x)
            glVertex3f( size, -0.8, x)
            x += step
        glEnd()

    # ── 绘制 HUD 文字 ─────────────────────────

    def _draw_hud(self, surface, roll, pitch, yaw, font):
        """在 pygame surface 上叠加文字信息"""
        lines = [
            (f"Roll  : {roll:+7.2f}°",  (220, 100, 100)),
            (f"Pitch : {pitch:+7.2f}°", (100, 220, 100)),
            (f"Yaw   : {yaw:+7.2f}°",   (100, 150, 255)),
        ]
        y = 12
        for text, color in lines:
            img = font.render(text, True, color)
            surface.blit(img, (12, y))
            y += 28

        note = font.render(
            "Yaw drifts without magnetometer", True, (160, 160, 160))
        surface.blit(note, (12, self.height - 28))

    # ── 主渲染循环 ────────────────────────────

    def _run(self):
        pygame.init()
        screen = pygame.display.set_mode(
            (self.width, self.height),
            DOUBLEBUF | OPENGL)
        pygame.display.set_caption(self.title)

        self._init_gl()

        font = pygame.font.Font(None, 26)   # 使用 pygame 内置字体，避开 SysFont Win32 bug
        clock = pygame.time.Clock()

        while self._running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return

            q = self.get_quaternion()

            # ── 清空缓冲 ──────────────────────
            glClearColor(0.08, 0.08, 0.12, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # ── 固定视角下绘制参考网格 ─────────
            self._draw_grid()

            # ── 应用四元数姿态 ──────────────────

            glPushMatrix()

            q = self.get_quaternion()

            # scipy quaternion:
            # [x, y, z, w]

            quat = [
                q[1],
                -q[3],
                q[2],
                q[0]
            ]

            rot_imu = R.from_quat(quat)

            # =====================================
            # 固定安装旋转
            # 让镜头朝下
            # =====================================

            mount_rot = R.from_euler(
                'x',
                180,
                degrees=True
            )

            # 最终姿态
            rot = rot_imu * mount_rot

            mat = rot.as_matrix()
            r = R.from_quat(quat)

            roll, pitch, yaw = r.as_euler(
                'xyz',
                degrees=True
            )

            # OpenGL 列优先
            gl_mat = np.eye(4, dtype=np.float32)
            gl_mat[:3, :3] = mat.T

            glMultMatrixf(gl_mat.flatten())

            self._draw_axes(length=0.9)
            self._draw_cube()

            glPopMatrix()

            # ── HUD 文字叠加（2D overlay）─────
            # 临时切换到 2D 正交投影绘制文字
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glDisable(GL_DEPTH_TEST)

            # 用 pygame surface 渲染文字再上传为纹理
            text_surf = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA)
            text_surf.fill((0, 0, 0, 0))
            self._draw_hud(text_surf, roll, pitch, yaw, font)

            text_data = pygame.image.tostring(text_surf, "RGBA", True)
            glRasterPos2i(0, 0)
            glDrawPixels(self.width, self.height,
                         GL_RGBA, GL_UNSIGNED_BYTE, text_data)

            glEnable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            pygame.display.flip()
            clock.tick(60)   # 60 fps 渲染上限

        pygame.quit()