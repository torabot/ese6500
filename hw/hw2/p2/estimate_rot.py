import math
import numpy as np
from quaternion import Quaternion

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

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw