import numpy as np
from scipy.spatial.transform import Rotation as R

class MadgwickFilter:

    def __init__(self, beta=0.003):

        self.beta = beta

        # quaternion
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro, accel, dt):

        q1, q2, q3, q4 = self.q

        gx, gy, gz = gyro
        ax, ay, az = accel

        # normalize accel
        norm_acc = np.linalg.norm(accel)

        if norm_acc < 1e-6:
            return self.q

        ax /= norm_acc
        ay /= norm_acc
        az /= norm_acc

        # objective function
        f = np.array([
            2*(q2*q4 - q1*q3) - ax,
            2*(q1*q2 + q3*q4) - ay,
            2*(0.5 - q2*q2 - q3*q3) - az
        ])

        # jacobian
        J = np.array([
            [-2*q3,  2*q4, -2*q1, 2*q2],
            [ 2*q2,  2*q1,  2*q4, 2*q3],
            [ 0,    -4*q2, -4*q3, 0]
        ])

        step = J.T @ f
        step_norm = np.linalg.norm(step)

        if step_norm > 1e-6:
            step = step / step_norm
        else:
            step = np.zeros(4)

        # gyro quaternion
        gyro_q = np.array([
            0.0,
            gx,
            gy,
            gz
        ])

        # quaternion derivative
        q_dot = (0.5 * self.quaternion_multiply(
            self.q,
            np.array([0, gx, gy, gz])
        ) - self.beta * step)

        # integrate
        self.q += q_dot * dt

        # normalize quaternion
        q_norm = np.linalg.norm(self.q)
        if q_norm > 1e-6:
            self.q /= q_norm

        return self.q

    def quaternion_multiply(self, q, r):

        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_euler(self):

        r = R.from_quat([
            self.q[1],
            self.q[2],
            self.q[3],
            self.q[0]
        ])

        roll, pitch, yaw = r.as_euler(
            'xyz',
            degrees=True
        )

        return roll, pitch, yaw