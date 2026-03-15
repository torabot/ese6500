import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from quaternion import Quaternion
import estimate_rot
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


"""
Calibration
"""

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

# print(f"Size of roll_imu: {roll_imu.shape}")
# print(f"Size of pitch_imu: {pitch_imu.shape}")
# print(f"Size of T : {T}")
# print(f"Size of vicon_ts : {vicon_ts.shape}")


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

print(f"bias_accel: {bias_accel}")
print(f"bias_gyro: {bias_gyro}")


"""
Run UKF
"""
roll_ukf, pitch_ukf, yaw_ukf = estimate_rot.estimate_rot(data_num)


"""
Generate figures
"""

# plt.figure()
# plt.plot(vicon_ts, roll_vicon, label='roll_vicon', marker='.', markersize=0.7)
# plt.plot(vicon_ts, pitch_vicon, label='pitch_vicon', marker='.', markersize=0.7)
# plt.plot(vicon_ts, yaw_vicon, label='yaw_vicon', marker='.', markersize=0.7)
# plt.xlabel("Timestamp")
# plt.ylabel("Orientation (rad)")
# title = "Vicon Orientation"
# plt.title(title)
# plt.gcf().canvas.manager.set_window_title(title)
# plt.legend()

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

# plt.figure()
# plt.plot(imu_ts[imu_overlap_mask], roll_imu, label='roll', marker='.', markersize=0.7)
# plt.plot(imu_ts[imu_overlap_mask], pitch_imu, label='pitch', marker='.', markersize=0.7)
# plt.xlabel("Timestamp")
# plt.ylabel("Orientation (rad)")
# title = "Orientation from Accel/Gyroscope Data"
# plt.title(title)
# plt.gcf().canvas.manager.set_window_title(title)
# plt.legend()

plt.figure()
plt.plot(vicon_ts, roll_vicon, label='roll_vicon', marker='.', markersize=0.7)
plt.plot(imu_ts, roll_ukf, label='roll_ukf', marker='.', markersize=0.7)
plt.show()
