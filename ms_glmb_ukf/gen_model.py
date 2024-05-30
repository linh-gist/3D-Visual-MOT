import numpy as np
from scipy.linalg import block_diag
import h5py
import scipy.io as sio


def lognormal_with_mean_cov(mean, sigma):
    # if mean=1, it return the same value of lognormal_with_mean_one(sigma)
    # mean, sigma are value of lognormal distribution
    temp = np.log(sigma ** 2 / mean ** 2 + 1)
    std_dev = np.sqrt(temp)
    mean = np.log(mean) - temp / 2
    return mean, std_dev


def lognormal_with_mean_one(percen):
    percen_v = percen ** 2
    std_dev = np.sqrt(np.log(percen_v + 1))
    mean = - std_dev ** 2 / 2
    return mean, std_dev


class model:
    def __init__(self, dataset="CMC1"):
        # basic parameters
        self.N_sensors = 4  # number of sensors
        self.x_dim = 9  # dimension of state vector
        self.z_dim = 4  # Assume all sensors have the same dimension of observation vector
        self.xv_dim = 5  # dimension of process noise
        self.zv_dim = 4  # dimension of observation noise
        self.XMAX = [2.03, 6.3]  # 2.03 5.77 6.3
        self.YMAX = [0.00, 3.41]  # [0.05 3.41];
        self.ZMAX = [0, 3]  # 5.77

        self.mode_type = ["Upright", "Fallen"]  # modes

        # param for ellipsoid plotting
        self.ellipsoid_n = 10

        # camera positions, image size and room dimensions
        self.sensor_pos = np.zeros((4, 3))
        self.sensor_pos[0] = [0.21, 3.11, 2.24]
        self.sensor_pos[1] = [7.17, 3.34, 2.16]
        self.sensor_pos[2] = [7.55, 0.47, 2.16]
        self.sensor_pos[3] = [0.21, 1.26, 2.20]
        self.imagesize = [1920, 1024]
        self.room_dim = [7.67, 3.41, 2.7]

        # load camera parameters
        self.cam_mat = np.zeros((self.N_sensors, 3, 4))
        self.cam_mat[0] = h5py.File("./cmc/cam1_cam_mat.mat", mode='r').get("cam1_cam_mat")[()].T
        self.cam_mat[1] = h5py.File("./cmc/cam2_cam_mat.mat", mode='r').get("cam2_cam_mat")[()].T
        self.cam_mat[2] = h5py.File("./cmc/cam3_cam_mat.mat", mode='r').get("cam3_cam_mat")[()].T
        self.cam_mat[3] = h5py.File("./cmc/cam4_cam_mat.mat", mode='r').get("cam4_cam_mat")[()].T

        # dynamical model parameters (CV model)
        T = 1  # sampling period
        A0 = np.array([[1, T],
                       [0, 1]])  # transition matrix
        self.F = block_diag(*[np.kron(np.eye(3, dtype='f8'), A0), np.eye(3, dtype='f8')])
        n_mu0, n_std_dev0 = lognormal_with_mean_one(0.06)  # input is std dev of multiplicative lognormal noise.
        n_mu1, n_std_dev1 = lognormal_with_mean_one(0.02)

        if dataset in ["CMC1", "CMC2", "CMC3", "WILDTRACK"]:
            sigma_v, sigma_radius, sigma_heig = 0.035, n_std_dev0, n_std_dev1
            B0 = sigma_v * np.array([[(T ** 2) / 2], [T]])
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.eye(3, dtype='f8'), B0), B1])
            self.Q = np.dot(B, B.T)[np.newaxis, :, :]  # process noise covariance
            self.r_birth = np.array([0.004])  # prob of birth
            self.mode = np.log(np.array([1.0]))
            self.w_birth = np.log(np.array([1.0]))  # weight of Gaussians - must be column_vector
            self.scale = 1
            n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.1)
            self.m_birth = np.array([2.3, 0.0, 1.2, 0, 0.825, 0, (np.log(0.3)) + n_mu_hold,
                                     (np.log(0.3)) + n_mu_hold, (np.log(0.84)) + n_mu_hold])[:, np.newaxis]
            B_birth = np.diagflat([[0.25, 0.1, 0.25, 0.1, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_birth = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]  # cov of Gaussians
            # Markov transition matrix for mode 0 is standing 1 is fall, # probability of survival
            self.mode_trans_matrix = np.log(np.array([[0.99, 0.01], [0.99, 0.01]]))
            self.n_mu = np.array([[n_mu0], [n_mu1]])

            # Adaptive birth parameters
            self.tau_ru = 0.9
            self.num_det = 2
            self.rB_max = 0.001  # cap birth probability
            self.rB_min = 1e-5

        if dataset in ["CMC4", "CMC5"]:
            Q = []
            # transition for standing to standing
            self.n_mu = np.zeros((2, 3))
            self.n_mu[:, 0] = [n_mu0, n_mu1]
            sigma_vz, sigma_vxy, sigma_radius, sigma_heig = 0.035, 0.035, n_std_dev0, n_std_dev1
            B0 = np.array([[T ** 2 / 2], [T]])
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.diag([sigma_vxy, sigma_vxy, sigma_vz]), B0), B1])
            Q.append(np.dot(B, B.T))  # process noise covariance

            # transition for falling to falling
            n_mu0, sigma_radius = lognormal_with_mean_one(0.4)
            n_mu1, sigma_heig = lognormal_with_mean_one(0.2)
            self.n_mu[:, 1] = [n_mu0, n_mu1]
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.diag([sigma_vxy, sigma_vxy, sigma_vz]), B0), B1])
            Q.append(np.dot(B, B.T))  # process noise covariance

            # transition from standing to fallen (vice versa)
            sigma_vz = 0.07
            sigma_vxy = 0.07
            n_mu0, n_std_dev = lognormal_with_mean_one(0.1)
            self.n_mu[:, 2] = [n_mu0, 0]
            sigma_radius = n_std_dev
            sigma_heig = n_std_dev
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.diag([sigma_vxy, sigma_vxy, sigma_vz]), B0), B1])
            Q.append(np.dot(B, B.T))  # process noise covariance
            self.Q = np.array(Q)

            self.r_birth = np.array([0.001])  # prob of birth
            self.mode = np.log(np.array([0.6, 0.4]))
            self.w_birth = np.log(np.array([1.0, 1.0]))  # weight of Gaussians - must be column_vector
            n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.2)
            m_birth1 = np.array([2.3, 0, 1.2, 0, 0.825, 0, (np.log(0.3)) + n_mu_hold, (np.log(0.3)) + n_mu_hold,
                                 (np.log(0.84)) + n_mu_hold])
            m_birth2 = np.array([2.3, 0, 1.2, 0, 0.825 / 2, 0, (np.log(0.84)) + n_mu_hold, (np.log(0.84)) + n_mu_hold,
                                 (np.log(0.3)) + n_mu_hold])
            self.m_birth = np.array([m_birth1, m_birth2]).T
            B_birth = np.diagflat([[0.25, 0.1, 0.25, 0.1, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_birth = np.array([np.dot(B_birth, B_birth.T), np.dot(B_birth, B_birth.T)]).T
            # Markov transition matrix for mode 0 is standing 1 is fall, # probability of survival
            self.mode_trans_matrix = np.log(np.array([[0.6, 0.4], [0.4, 0.6]]))

            # Adaptive birth parameters
            self.tau_ru = 0.9
            self.num_det = 2
            self.rB_max = 0.001  # cap birth probability
            self.rB_min = 1e-5

        if dataset == "CMC5":  # different calib params due to different set of recording
            self.cam_mat[2] = sio.loadmat('./cmc/cam3_cam_mat__.mat')["cam3_cam_mat__"]
            self.cam_mat[3] = sio.loadmat('./cmc/cam4_cam_mat__.mat')["cam4_cam_mat__"]

        if dataset in ["WILDTRACK"]:  # change some parameters
            from wildtrack.wildtrack import compute_wildtrack_cam_mat
            self.sensor_pos, self.cam_mat = compute_wildtrack_cam_mat("./wildtrack")
            self.XMAX = [-3.0, 9.0]
            self.YMAX = [-8.975, 26.9999479167]
            self.N_sensors = 7

            sigma_v, sigma_radius, sigma_heig = 0.15, n_std_dev0, n_std_dev1
            B0 = sigma_v * np.array([[(T ** 2) / 2], [T]])
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.eye(3, dtype='f8'), B0), B1])
            self.Q = np.dot(B, B.T)[np.newaxis, :, :]  # process noise covariance
            n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.1)
            B_birth = np.diagflat([[0.5, 0.55, 0.5, 0.55, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_birth = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]  # cov of Gaussians

            # Adaptive birth parameters
            self.tau_ru = 0.9
            self.num_det = 1
            self.rB_max = 0.06  # cap birth probability
            self.rB_min = 1e-25

        # survival/death parameters
        self.P_S = 0.999999999
        self.Q_S = 1 - self.P_S

        # measurement parameters
        self.meas_n_mu = np.zeros((2, 2))
        # mode 0 (standing)
        self.meas_n_mu[0, 0], meas_n_std_dev0 = lognormal_with_mean_one(0.1)
        self.meas_n_mu[1, 0], meas_n_std_dev1 = lognormal_with_mean_one(0.05)
        D0 = np.diag([20, 20, meas_n_std_dev0, meas_n_std_dev1])
        # mode 1 (fallen)
        self.meas_n_mu[0, 1], meas_n_std_dev0 = lognormal_with_mean_one(0.05)
        self.meas_n_mu[1, 1], meas_n_std_dev1 = lognormal_with_mean_one(0.1)
        D1 = np.diag([20, 20, meas_n_std_dev0, meas_n_std_dev1])
        self.R = np.array([np.dot(D0, D0.T), np.dot(D1, D1.T)])  # observation noise covariance

        # detection probabilities
        self.P_D = 0.97  # probability of detection in measurements
        self.Q_D = 1 - self.P_D  # probability of missed detection in measurements

        lambda_c = 5
        self.lambda_c = np.tile(lambda_c, (self.N_sensors, 1))  # poisson average rate of uniform clutter (per scan)
        self.lambda_c = np.log(self.lambda_c)
        self.range_c = np.array([[1, 1920], [1, 1024], [1, 1920], [1, 1024]], dtype='f8')  # uniform clutter region
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])  # uniform clutter density
        range_temp = self.range_c[:, 1] - self.range_c[:, 0]
        range_temp[2: 4] = np.log(range_temp[2: 4])
        self.pdf_c = np.tile(1 / np.prod(range_temp), (self.N_sensors, 1))
        self.pdf_c = np.log(self.pdf_c)


if __name__ == '__main__':
    import pprint

    model_params = model()
    pprint.pprint(vars(model_params))
