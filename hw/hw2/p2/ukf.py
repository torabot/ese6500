import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as Rot
from quaternion import Quaternion

def quat_from_rotvec(rotvec):
    angle = np.linalg.norm(rotvec)

    if angle < 1e-12:
        return Quaternion(1.0, np.zeros(3))

    axis = rotvec / angle
    return Quaternion(np.cos(angle / 2.0), axis * np.sin(angle / 2.0))


def process_model(q, omega, dt):

    rotvec = omega * dt
    dq = quat_from_rotvec(rotvec)

    q_next = q.__mul__(dq)
    q_next.normalize()

    omega_next = omega.copy()

    return np.hstack([q_next.q, omega_next])


def compute_sigma_points(q_mean, omega_mean, Sigma, Q_proc):
    n = Sigma.shape[0]
    X = np.zeros((2 * n, 7))

    S = scipy.linalg.sqrtm(Sigma + Q_proc)

    for i in range(n):
        col = np.sqrt(n) * S[:, i]

        for sign, idx in [(+1.0, i), (-1.0, i + n)]:
            Wi = sign * col
            q_W = quat_from_rotvec(Wi[:3])
            omega_W = Wi[3:]

            q_sigma = q_mean.__mul__(q_W)
            q_sigma.normalize()

            omega_sigma = omega_mean + omega_W

            X[idx, :] = np.hstack([q_sigma.q, omega_sigma])

    return X


def quaternion_mean_and_covariance(X, q_bar, tol=1e-6, max_iters=100):
    num_sigma = X.shape[0]
    n = num_sigma // 2

    for _ in range(max_iters):
        E = np.zeros((3, num_sigma))

        for i in range(num_sigma):
            q_i = Quaternion(X[i, 0], X[i, 1:4])
            e_i = q_i * q_bar.inv()

            if e_i.q[0] < 0:
                e_i.q = -e_i.q

            E[:, i] = np.asarray(e_i.axis_angle()).reshape(3,)

        e_mean_vec = np.mean(E, axis=1)
        eps = np.linalg.norm(e_mean_vec)

        if eps < tol:
            break

        e_mean = Quaternion()
        e_mean.from_axis_angle(e_mean_vec)

        q_bar = e_mean * q_bar
        q_bar.normalize()

    E = np.zeros((3, num_sigma))

    for i in range(num_sigma):
        q_i = Quaternion(X[i, 0], X[i, 1:4])
        e_i = q_i * q_bar.inv()

        if e_i.q[0] < 0:
            e_i.q = -e_i.q

        E[:, i] = np.asarray(e_i.axis_angle()).reshape(3,)

    omega_bar = np.mean(X[:, 4:7], axis=0)

    Wprime = np.zeros((6, num_sigma))
    for i in range(num_sigma):
        d_omega_i = X[i, 4:7] - omega_bar
        Wprime[:, i] = np.hstack([E[:, i], d_omega_i])

    Sigma_bar = (Wprime @ Wprime.T) / (2 * n)
    mu_bar = np.hstack([q_bar.q, omega_bar])

    return mu_bar, Sigma_bar, E


def predict_step(mu_km1, Sigma_km1, Q_proc, dt):
    q_km1 = Quaternion(mu_km1[0], mu_km1[1:4])
    q_km1.normalize()
    omega_km1 = mu_km1[4:7]

    X = compute_sigma_points(q_km1, omega_km1, Sigma_km1, Q_proc)

    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        q_i = Quaternion(X[i, 0], X[i, 1:4])
        q_i.normalize()
        omega_i = X[i, 4:7]

        Y[i, :] = process_model(q_i, omega_i, dt)

    mu_pred, Sigma_pred, _ = quaternion_mean_and_covariance(Y, q_km1)

    return mu_pred, Sigma_pred, Y


def measurement_model(q_bar, omega):
    g = Quaternion(0, np.array([0, 0, -9.81]))
    g_prime = q_bar.inv().__mul__(g).__mul__(q_bar)
    return np.hstack([g_prime.q[1:], omega])


def obs_step(Y, mu_pred, Q_meas):
    num_sigma = Y.shape[0]
    n = num_sigma // 2

    q_bar_t = Quaternion(mu_pred[0], mu_pred[1:4])
    q_bar_t.normalize()
    omega_k = mu_pred[4:7]

    Z = np.zeros((num_sigma, 6))
    e_rot = np.zeros((3, num_sigma))
    Wprime = np.zeros((6, num_sigma))

    for i in range(num_sigma):
        q_i = Quaternion(Y[i, 0], Y[i, 1:4])
        q_i.normalize()
        omega_i = Y[i, 4:7]

        Z[i, :] = measurement_model(q_i, omega_i)

        e_i = q_i.__mul__(q_bar_t.inv())
        if e_i.q[0] < 0:
            e_i.q = -e_i.q
        e_rot[:,i] = e_i.axis_angle().reshape(3,)
        Wprime[:, i] = np.hstack([e_rot[:,i], omega_i - omega_k])

    y_hat = np.mean(Z, axis=0)
    w = 1 / (2*n)

    Sigma_yy = np.zeros((6, 6))
    for i in range(num_sigma):
        gmy = (Z[i] - y_hat).reshape(6, 1)
        Sigma_yy += gmy @ gmy.T
    Sigma_yy *= w
    Sigma_yy += Q_meas

    Sigma_xy = np.zeros((6, 6))
    for i in range(num_sigma):
        xmmu = Wprime[:, i].reshape(6, 1)
        gmy = (Z[i] - y_hat).reshape(6, 1)
        Sigma_xy += xmmu @ gmy.T
    Sigma_xy *= w

    return y_hat, Sigma_xy, Sigma_yy,Z


def belief(mu_pred, Sigma_pred, Sigma_xy, Sigma_yy, y, y_hat):
    innovation = y - y_hat
    K = Sigma_xy @ np.linalg.inv(Sigma_yy)
    dx = K @ innovation
    dq = quat_from_rotvec(dx[:3])
    q_bar = Quaternion(mu_pred[0], mu_pred[1:4])
    q_bar.normalize()
    q_bel = q_bar.__mul__(dq)
    q_bel.normalize()

    mu_bel = np.hstack([q_bel.q, mu_pred[4:] + dx[3:]])
    Sigma_bel = Sigma_pred - K @ Sigma_yy @ K.T

    return mu_bel, Sigma_bel

def ukf(euler_angle_init, omega_init, Sigma_init, accel_data, gyro_data, Q_proc=np.eye(6), Q_meas=np.eye(6), dt = 0.01):
    q_tmp = Rot.from_euler('xyz', euler_angle_init, degrees=False).as_quat()
    q_km1 = Quaternion(q_tmp[3], q_tmp[:3])

    q_km1.q = q_km1.q / np.linalg.norm(q_km1.q)
    mu_km1 = np.hstack([q_km1.q, omega_init])
    Sigma_km1  = Sigma_init

    y = np.hstack((accel_data, gyro_data))

    mu_pred, Sigma_pred, Y = predict_step(mu_km1, Sigma_km1, Q_proc, dt)
    y_hat, Sigma_xy, Sigma_yy, Z = obs_step(Y, mu_pred, Q_meas)
    mu_bel, Sigma_bel = belief(mu_pred, Sigma_pred, Sigma_xy, Sigma_yy, y, y_hat)

    return mu_bel, Sigma_bel