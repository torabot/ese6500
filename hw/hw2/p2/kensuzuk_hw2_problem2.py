import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scipy.linalg
import matplotlib.pyplot as plt
import ukf

# Load data
data_num = 1
imu = np.load(f"./imu/imuBiased{data_num}.npy", allow_pickle=True).item()
accel = imu['accel']
gyro = imu['gyro']
imu_ts = np.asarray(imu['ts'])
T = imu_ts.shape[0]

# Comment out line below for GradeScope submission
vicon = np.load(f"./vicon/viconRot{data_num}.npy", allow_pickle=True).item()
vicon_ts = np.asarray(vicon['ts'])

rot_mats = Rot.from_matrix(vicon['rots'])
euler_angles = rot_mats.as_euler('xyz', degrees=False)
roll_vicon = euler_angles[:,0]
pitch_vicon = euler_angles[:,1]
yaw_vicon = euler_angles[:,2]

# Calibrate accelerometer
imu_overlap_mask = (imu_ts >= vicon_ts[0]) & (imu_ts <= vicon_ts[-1])
imu_static_mask = ((imu_ts < 1296636791.2) | (imu_ts > 1296636828.7)) & imu_overlap_mask
g = np.array([0, 0, -9.81])
g_body = rot_mats.apply(g, inverse=True)
g_body_interp = np.column_stack([
    np.interp(imu_ts[imu_static_mask], vicon_ts, g_body[:, axis])
    for axis in range(3)
])

bias_samples_accel = accel[imu_static_mask] + g_body_interp
bias_accel = np.mean(bias_samples_accel, axis=0)
unbiased_accel = accel[imu_overlap_mask] - bias_accel

roll_imu = np.atan2(unbiased_accel[:, 1], unbiased_accel[:, 2])
pitch_imu = -np.atan2(unbiased_accel[:, 0], np.sqrt(unbiased_accel[:, 1]**2 + unbiased_accel[:, 2]**2))


# Calibrate gyroscope
dt_vicon = np.diff(vicon_ts)
delta_rot = rot_mats[:-1].inv() * rot_mats[1:]
vicon_angular_velocity = delta_rot.as_rotvec() / dt_vicon[:, None]
vicon_angular_velocity_ts = vicon_ts[:-1]
vicon_angular_velocity_interp = np.column_stack([
    np.interp(imu_ts[imu_overlap_mask], vicon_angular_velocity_ts, vicon_angular_velocity[:, axis])
    for axis in range(3)
])

gyro_overlap = gyro[imu_overlap_mask]
bias_samples_gyro = gyro_overlap - vicon_angular_velocity_interp
bias_gyro = np.mean(bias_samples_gyro, axis=0)
unbiased_gyro = gyro_overlap - bias_gyro

def process_model(q_kp1, omega, dt):
    angle = np.linalg.norm(omega) * dt
    axis = omega / np.linalg.norm(omega)
    q_tmp = np.hstack([np.cos(angle / 2.0), axis * np.sin(angle / 2.0)])
    dq = Quaternion(q_tmp[0], q_tmp[1:4])
    q_kp1.q = q_kp1.__mul__(dq)
    q_kp1.q = q_kp1.q / np.linalg.norm(q_kp1.q)

    omega_kp1 = omega.copy()
    return np.concatenate(q_kp1.q, omega_kp1)

def compute_sigma_pts(q_mean, omega_mean, Sigma, Q_proc):
    n = Sigma.shape[0]
    X = np.zeros((2*n, 7))
    S = scipy.linalg.sqrtm(Sigma + Q_proc)

    for i in range(n):
        W = np.sqrt(n) * np.real(S[:, i])

        for sign, idx in [(1.0, i), (-1.0, i+n)]:
            Wi = sign * W
            q_tmp = Rot.from_rotvec(Wi[:3]).as_quat()
            q_W = Quaternion(q_tmp[3], q_tmp[:3])
            q_sigma = q_mean * q_W
            omega_sigma = omega_mean + Wi[3:]
            X[idx, :] = np.hstack([q_sigma.q, omega_sigma])

    return X

def gradient_descent(mu_km1, X):
    n = X.shape[1]
    eps = np.inf
    q_bar = Quaternion(mu_km1[0], mu_km1[1:4])
    while eps > 1e-6:
        E = np.zeros((3, 2*n))

        for i in range(2*n):
            q_i = Quaternion(X[i, 0], X[i, 1:4])

            e_i = q_i.__mul_(q_bar.inv())

            if e_i.q[0] < 0:
                e_i.q = -e_i.q

            E[:, i] = e_i.axis_angle().reshape(3,)

        e_vec = np.mean(E, axis=1)
        e_mean = Quaternion()
        e_mean.from_axis_angle(e_vec)
        q_bar = e_mean.__mul__(q_bar)
        eps = np.linalg.norm(e_vec)
            
    Wprime = np.zeros((6, 2*n))
    omega_bar = np.mean(X[:, 4:7], axis=0)
    for i in range(2*n):
        d_omega_i = X[i, 4:7] - omega_bar
        Wprime[:, i] = np.hstack([E[:, i], d_omega_i])

    mu_bar = np.hstack([q_bar.q, omega_bar])
    Sigma_bar = (Wprime @ Wprime.T) / (2*n)
    return mu_bar, Sigma_bar

def measurement_update()

def ukf(euler_angle_init, omega_init, Sigma_init, accel_data, gyro_data, dt = 0.01, R = 1.0, Q = 0.5, seed=42069):

    q_tmp = Rot.from_euler('xyz', euler_angle_init, degrees=False).as_quat()
    q_km1 = Quaternion(q_tmp[3], q_tmp[:3])

    q_km1.q = q_km1.q / np.linalg.norm(q_km1.q)
    x_bar_init = np.hstack([q_km1.q, omega_init])

    omega_km1 = omega_init
    omega_W = 0.

    n = Sigma_init.shape[1]
    mu_km1 = x_bar_init
    Sigma_km1  = Sigma_init

    # Compute sigma points
    X = compute_sigma_pts(q_km1, omega_km1, omega_W, Sigma_km1, R, dt)


    mu_bar, Sigma_bar = gradient_descent(mu_km1, X)


    X_tmp = compute_sigma_pts(Quaternion(mu_bar[0], mu_bar[1:4]), mu_bar[4:], omega_W, Sigma_bar, R, dt)

    Y = np.zeros_like(X)
    for i in range(n):
        q_tmp = X_tmp[i]
        q_tmp = Quaternion(q_tmp[0], q_tmp[1:4])
        Y[i,:] = process_model()
        q_tmp = X_tmp[i+n]
        q_tmp = Quaternion(q_tmp[0], q_tmp[1:4])
        Y[i+n,:] = process_model(X_tmp[i+n])


"""
Generate figures
"""

plt.figure()
plt.plot(vicon_ts, roll_vicon, label='roll_vicon', marker='.', markersize=0.7)
plt.plot(vicon_ts, pitch_vicon, label='pitch_vicon', marker='.', markersize=0.7)
plt.plot(vicon_ts, yaw_vicon, label='yaw_vicon', marker='.', markersize=0.7)
plt.xlabel("Timestamp")
plt.ylabel("Orientation (rad)")
title = "Vicon Orientation"
plt.title(title)
plt.gcf().canvas.manager.set_window_title(title)
plt.legend()

# plt.figure()
# plt.plot(imu_ts, accel[:,0], label='x', marker='.', markersize=0.7)
# plt.plot(imu_ts, accel[:,1], label='y', marker='.', markersize=0.7)
# plt.plot(imu_ts, accel[:,2], label='z', marker='.', markersize=0.7)
# plt.xlabel("Timestamp")
# plt.ylabel("Acceleration (m/s^2)")
# title = "Raw Accelerometer Data"
# plt.title(title)
# plt.gcf().canvas.manager.set_window_title(title)
# plt.legend()

# plt.figure()
# plt.plot(imu_ts[imu_overlap_mask], unbiased_accel[:, 0], label='x', marker='.', markersize=0.7)
# plt.plot(imu_ts[imu_overlap_mask], unbiased_accel[:, 1], label='y', marker='.', markersize=0.7)
# plt.plot(imu_ts[imu_overlap_mask], unbiased_accel[:, 2], label='z', marker='.', markersize=0.7)
# plt.xlabel("Timestamp")
# plt.ylabel("Acceleration (m/s^2)")
# title = "Unbiased Accelerometer Data"
# plt.title(title)
# plt.gcf().canvas.manager.set_window_title(title)
# plt.legend()

# plt.figure()
# plt.plot(imu_ts, gyro[:,0], label='rx', marker='.', markersize=0.7)
# plt.plot(imu_ts, gyro[:,1], label='ry', marker='.', markersize=0.7)
# plt.plot(imu_ts, gyro[:,2], label='rz', marker='.', markersize=0.7)
# plt.xlabel("Timestamp")
# plt.ylabel("Angular Velocity (rad/s)")
# title = "Raw Gyroscope Data"
# plt.title(title)
# plt.gcf().canvas.manager.set_window_title(title)
# plt.legend()

# plt.figure()
# plt.plot(imu_ts[imu_overlap_mask], unbiased_gyro[:,0], label='rx', marker='.', markersize=0.7)
# plt.plot(imu_ts[imu_overlap_mask], unbiased_gyro[:,1], label='ry', marker='.', markersize=0.7)
# plt.plot(imu_ts[imu_overlap_mask], unbiased_gyro[:,2], label='rz', marker='.', markersize=0.7)
# plt.xlabel("Timestamp")
# plt.ylabel("Angular Velocity (rad/s)")
# title = "Unbiased Gyroscope Data"
# plt.title(title)
# plt.gcf().canvas.manager.set_window_title(title)
# plt.legend()

plt.figure()
plt.plot(imu_ts[imu_overlap_mask], roll_imu, label='roll', marker='.', markersize=0.7)
plt.plot(imu_ts[imu_overlap_mask], pitch_imu, label='pitch', marker='.', markersize=0.7)
plt.xlabel("Timestamp")
plt.ylabel("Orientation (rad)")
title = "Orientation from Accel/Gyroscope Data"
plt.title(title)
plt.gcf().canvas.manager.set_window_title(title)
plt.legend()


plt.show()


"""
first timestamp to 1296636791.2
1296636828.7 to end
"""
