# import math
import numpy as np
from quaternion import Quaternion
from scipy.spatial.transform import Rotation as Rot
import ukf

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = np.load(f"imu/imuBiased{data_num}.npy", allow_pickle=True).item()
    vicon = np.load(f"vicon/viconRot{data_num}.npy", allow_pickle=True).item()
    accel = imu['accel']
    gyro = imu['gyro']
    T = np.shape(imu['ts'])[0]

    # your code goes here

    """
    Unbias Sensor Data
    """

    bias_accel = np.array([-47.78809278, -45.58779243, 45.67206614])
    bias_gyro = np.array([6.01701315, 6.33421172, 5.07885884])

    unbiased_accel = accel - bias_accel
    unbiased_gyro = gyro - bias_gyro

    """
    Find Initial Roll/Pitch Values from Accelerometer
    """
    roll_imu = np.atan2(unbiased_accel[:, 1], unbiased_accel[:, 2])
    pitch_imu = -np.atan2(unbiased_accel[:, 0], np.sqrt(unbiased_accel[:, 1]**2 + unbiased_accel[:, 2]**2))

    """
    Run UKF
    """

    euler_init = np.array([roll_imu[0], pitch_imu[0], 0.0])
    q_init_tmp = Rot.from_euler('xyz', euler_init, degrees=False).as_quat()
    q_init = Quaternion(q_init_tmp[3], q_init_tmp[:3])
    q_init.normalize()

    mu_bel = np.hstack([q_init.q, np.zeros(3)])
    Sigma_bel = np.eye(6)

    euler_angle_hist = np.zeros((T, 3))
    euler_angle_hist[0, :] = euler_init

    for k in range(T):
        euler_angles_init = Quaternion(mu_bel[0], mu_bel[1:4]).euler_angles()
        mu_bel, Sigma_bel = ukf.ukf(euler_angles_init, mu_bel[4:], Sigma_bel, unbiased_accel[k, :], unbiased_gyro[k, :], dt=0.01)
        euler_angle_hist[k, :] = Quaternion(mu_bel[0], mu_bel[1:4]).euler_angles()

    roll = euler_angle_hist[:,0]
    pitch = euler_angle_hist[:,1]
    yaw = euler_angle_hist[:,2]

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw
