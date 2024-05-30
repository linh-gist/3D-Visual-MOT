from copy import deepcopy
import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import block_diag
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from gen_meas import homtrans


def ut(m, P, alpha, kappa):
    n_x = len(m)
    lambda_ = alpha ** 2 * (n_x + kappa) - n_x
    Psqrtm = cholesky((n_x + lambda_) * P).T
    temp = np.zeros((n_x, 2 * n_x + 1))
    temp[:, 1:n_x + 1], temp[:, n_x + 1:2 * n_x + 1] = -Psqrtm, Psqrtm
    X = np.tile(m, (1, 2 * n_x + 1)) + temp
    w = 0.5 * np.ones((2 * n_x + 1, 1))
    w[0] = lambda_
    w = w / (n_x + lambda_)
    return X, w


def gen_msobservation_fn(model, Xorg, W, q):
    if len(Xorg) == 0:
        return []
    X = np.copy(Xorg)
    X[[6, 7, 8], :] = np.exp((X[[6, 7, 8], :]))
    bbs_noiseless = np.zeros((4, X.shape[1]))
    for i in range(X.shape[1]):
        ellipsoid_c = np.array([X[0, i], X[2, i], X[4, i]])  # xc, yc, zc
        rx, ry, hh = X[6, i], X[7, i], X[8, i]  # half length radius (x, y, z)
        # Quadric general equation 0 = Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J
        # Q = [A D/2 E/2 G/2;
        #      D/2 B F/2 H/2;
        #      E/2 F/2 C I/2;
        #      G/2 H/2 I/2 J];
        A, B, C = 1 / (rx ** 2), 1 / (ry ** 2), 1 / (hh ** 2)  # calculations for A, B, C
        D, E, F = 0, 0, 0  # calculations for D, E, F, no rotation (axis-aligned) means D, E, F = 0
        # calculations for G, H, I, J
        PSD = np.diag([A, B, C])
        eig_vals, right_eig = np.linalg.eig(PSD)  # [V,D] = eig(A), right eigenvectors, so that A*V = V*D
        temp_ellip_c = np.dot(right_eig.T, ellipsoid_c)
        ggs = (-2 * np.multiply(temp_ellip_c, eig_vals))
        desired = np.dot(ggs, right_eig.T)
        G, H, I = desired[0], desired[1], desired[2]
        J_temp = np.sum(np.divide(np.power(ggs.T, 2), 4 * eig_vals))  # or sum(eig_vals_vec.*(-temp_ellip_c).^2)
        J = -1 + (J_temp)
        Q = np.array([[A, D / 2, E / 2, G / 2],
                      [D / 2, B, F / 2, H / 2],
                      [E / 2, F / 2, C, I / 2],
                      [G / 2, H / 2, I / 2, J]])  # 4x4 matrix
        C_t = np.dot(np.dot(model.cam_mat[q], np.linalg.inv(Q)), model.cam_mat[q].T)
        CI = np.linalg.inv(C_t)  # 3x3 matrix
        C_strip = CI[0:2, 0:2]  # [C(1,1:2);C(2,1:2)]; % 2x2 matrix
        eig_vals, right_eig = np.linalg.eig(C_strip)
        x_and_y_vec = 2 * CI[0:2, 2]  # np.array([[2. * CI[0, 2], 2. * CI[1, 2]]])  # extrack D and E
        x_and_y_vec_transformed = np.dot(x_and_y_vec, right_eig)
        h_temp = np.divide(x_and_y_vec_transformed, eig_vals) / 2
        h_temp_squared = np.multiply(eig_vals, np.power(h_temp, 2))
        h = -1 * h_temp
        ellipse_c = np.dot(right_eig, h)
        offset = -np.sum(h_temp_squared) + CI[2, 2]
        if (-offset / eig_vals[0] > 0) and (-offset / eig_vals[1] > 0):
            uu = right_eig[:, 0] * np.sqrt(-offset / eig_vals[0])
            vv = right_eig[:, 1] * np.sqrt(-offset / eig_vals[1])
            e = np.sqrt(np.multiply(uu, uu) + np.multiply(vv, vv))
            bbox = np.vstack((ellipse_c - e, ellipse_c + e)).T
            tl0 = np.amin(bbox[0, :])  # top_left0
            tl1 = np.amin(bbox[1, :])  # top_left1
            br0 = np.amax(bbox[0, :])  # bottm_right0
            br1 = np.amax(bbox[1, :])  # bottm_right1
            bbs_noiseless[:, i] = np.array([tl0, tl1, np.log((br0 - tl0)), np.log((br1 - tl1))])
        else:
            # top_left = [1 1];
            bottm_right = [model.imagesize[0], model.imagesize[1]]
            bbs_noiseless[:, i] = np.array([1, 1, np.log((bottm_right[0] - 1)), np.log((bottm_right[1] - 1))])

    return bbs_noiseless + W  # bounding measurement


def gen_msobservation_fn_v2(model, Xorg, W, meas_n_mu_mode, q):
    if len(Xorg) == 0:
        return []
    X = np.copy(Xorg)
    X[[6, 7, 8], :] = np.exp((X[[6, 7, 8], :]))
    vet2 = np.ones((4, 6))
    bbs_noiseless = np.zeros((4, X.shape[1]))
    for i in range(X.shape[1]):
        temp = X[:, i]
        temp = temp[[0, 2, 4, 6, 7, 8]]
        vet2[0:3, 0] = [temp[0] + temp[3], temp[1], temp[2]]  # - right
        vet2[0:3, 1] = [temp[0] - temp[3], temp[1], temp[2]]  # - left
        vet2[0:3, 2] = [temp[0], temp[1] + temp[4], temp[2]]  # | right
        vet2[0:3, 3] = [temp[0], temp[1] - temp[4], temp[2]]  # | left
        vet2[0:3, 4] = [temp[0], temp[1], temp[2] + temp[5]]
        vet2[0:3, 5] = [temp[0], temp[1], temp[2] - temp[5]]

        temp_c = np.dot(model.cam_mat[q], vet2)
        vertices = temp_c[[0, 1], :] / temp_c[2, :]
        x_2 = max(vertices[0, :])
        x_1 = min(vertices[0, :])
        y_2 = max(vertices[1, :])
        y_1 = min(vertices[1, :])
        bbs_noiseless[:, i] = [x_1, y_1, np.log(x_2 - x_1) + meas_n_mu_mode[0], np.log(y_2 - y_1) + meas_n_mu_mode[1]]
    return bbs_noiseless + W


def ukf_update_per_sensor(z, m, P, s, mode, model, alpha=1, kappa=2, beta=2):
    m_hold = m[[0, 2, 4]]
    ch1 = model.XMAX[0] < m_hold[0] < model.XMAX[1]
    ch2 = model.YMAX[0] < m_hold[1] < model.YMAX[1]
    ch3 = model.ZMAX[0] < m_hold[2] < model.ZMAX[1]
    if not (ch1 and ch2 and ch3):
        return np.log(np.spacing(0)), m, P
    m_ut, P_ut = np.append(m, np.zeros((model.z_dim, 1)), axis=0), block_diag(*[P, model.R[mode]])
    X_ukf, u = ut(m_ut, P_ut, alpha, kappa)
    Z_pred = gen_msobservation_fn_v2(model, X_ukf[0:model.x_dim, :], X_ukf[model.x_dim:model.x_dim + model.z_dim, :],
                                     model.meas_n_mu[:, 0], s)
    eta = np.dot(Z_pred, u)
    S_temp = Z_pred - np.tile(eta, (1, len(u)))
    u[0] = u[0] + (1 - alpha ** 2 + beta)
    S = np.linalg.multi_dot([S_temp, np.diagflat(u), S_temp.T])
    Vs = cholesky(S)
    det_S = np.prod(np.diag(Vs)) ** 2
    inv_sqrt_S = np.linalg.inv(Vs)
    iS = np.dot(inv_sqrt_S, inv_sqrt_S.T)
    G_temp = X_ukf[0:model.x_dim, :] - np.tile(m, (1, len(u)))
    G = np.dot(np.dot(G_temp, np.diagflat(u)), S_temp.T)
    K = np.dot(G, iS)
    z_eta = z - eta
    qz_temp = -0.5 * (z.shape[0] * np.log(2 * np.pi) + np.log(det_S) + np.dot(np.dot(z_eta.T, iS), z_eta))
    # qz_temp = np.exp(qz_temp)
    m_temp = m + np.dot(K, z_eta)
    P_temp = P - np.dot(np.dot(G, iS), G.T)

    return qz_temp, m_temp, P_temp


def cleanup(w, m, P, elim_threshold=1e-5, merge_threshold=4, l_max=10):
    if len(w) <= 1:
        return w, m, P
    # Gaussian prune, remove components that have weight lower than a threshold
    w = np.exp(w)
    idx = np.nonzero(w > elim_threshold)[0]
    w = w[idx]
    m = m[:, idx]
    P = P[:, :, idx]

    # Merging Gaussian mixture components using Mahalanobis distance
    x_dim, max_cmpt = m.shape[0], m.shape[1]
    w_new, x_new, P_new = np.empty(max_cmpt), np.empty((x_dim, max_cmpt)), np.empty((x_dim, x_dim, max_cmpt))
    I = np.arange(len(w))
    idx = 0
    while len(I):
        j = np.argmax(w)
        Ij = np.empty(0, dtype=int)
        iPt = np.linalg.inv(P[:, :, j])
        for i in I:
            xi_xj = m[:, i] - m[:, j]
            val = np.dot(np.dot(xi_xj.T, iPt), xi_xj)
            if val <= merge_threshold:
                Ij = np.append(Ij, i)
        w_new_t = sum(w[Ij])
        x_new_t = np.sum(m[:, Ij] * w[Ij], axis=1)
        P_new_t = np.sum(P[:, :, Ij] * w[Ij], axis=2)
        x_new_t = x_new_t / w_new_t
        P_new_t = P_new_t / w_new_t
        w_new[idx] = w_new_t
        x_new[:, idx] = x_new_t
        P_new[:, :, idx] = P_new_t
        I = np.setdiff1d(I, Ij)
        w[Ij] = -1
        idx += 1
    w = w_new[:idx]
    m = x_new[:, :idx]
    P = P_new[:, :, :idx]

    # Gaussian cap, limit on number of Gaussians in each track
    if len(w) > l_max:
        idx = np.argsort(-w)
        w_new = w[idx[:l_max]]
        w = w_new * (sum(w) / sum(w_new))
        m = m[:, idx[:l_max]]
        P = P[:, :, idx[:l_max]]

    return np.log(w), m, P


def norm_feat01(x):
    min_x = np.amin(x)
    x = (x - min_x) / (np.amax(x) - min_x)
    return x / np.linalg.norm(x)


class Target:
    # track table for GLMB (cell array of structs for individual tracks)
    # (1) r: existence probability
    # (2) Gaussian Mixture w (weight), m (mean), P (covariance matrix)
    # (3) Label: birth time & index of target at birth time step
    # (4) gatemeas: indexes gating measurement (using  Chi-squared distribution)
    # (5) model: [0.6, 0.4] probability of being ["Upright", "Fallen"]
    # (6) feat: array of re-identification feature of Target in all sensors
    # (7) ah: Measurement history association
    def __init__(self, m, P, prob_birth, label, feat, model, birth_time, use_feat=True):
        self.m = m  # [x dx y dy z dz log_wx, log_wy, log_z]
        self.P = P
        self.w = np.copy(model.w_birth)  # weights of Gaussians for birth track
        self.r = prob_birth
        self.P_S = model.P_S
        self.l = label  # track label
        self.gatemeas = [[] for i in range(model.N_sensors)]
        self.mode = np.copy(model.mode)

        # wg, mg, Pg, ..., store temporary Gaussian mixtures while predicting and updating
        max_cpmt, x_dim = 100, 9
        self.wg = np.zeros(max_cpmt, dtype='f8')
        self.mg = np.zeros((x_dim, max_cpmt), dtype='f8')
        self.Pg = np.zeros((x_dim, x_dim, max_cpmt), dtype='f8')
        self.gm_len = np.ones(len(model.mode), dtype=int)  # number of Gaussian mixtures (block of GM for each mode)

        self.use_feat = use_feat  # determine whether to use re-id feature
        if use_feat:
            self.alpha_feat = 0.9
            self.re_alpha_feat = 0.2
            self.feat_flag = (np.sum(feat, axis=1) == 0)  # checking whether initial feature is missed
            average_feat = np.sum(feat, axis=0) / sum(1 - self.feat_flag)
            average_feat = norm_feat01(average_feat)
            self.feat = np.zeros((model.N_sensors, len(average_feat)))
            for s in range(model.N_sensors):
                if sum(feat[s, :]) == 0:
                    # if missed, obtain average feature from other detected measurement
                    self.feat[s, :] = average_feat
                else:
                    self.feat[s, :] = norm_feat01(feat[s, :])
        self.last_active = birth_time
        self.birth_time = birth_time
        self.ah = []

    def predict(self, model):
        offset = np.zeros((model.x_dim, 1))
        mode_len = len(self.mode)
        mode_predict = np.zeros(mode_len)
        gm_tmp_idx = 0
        for n_mode in range(mode_len):
            mode_predict_temp = np.zeros(mode_len)
            gm_s = 0  # Start of GM index
            gm_e = 0  # End of GM index
            for c_mode in range(mode_len):
                if c_mode == n_mode:  # transition to same mode
                    if c_mode == 0:  # transition from Upright to Upright
                        offset[[6, 7, 8], 0] = np.array([model.n_mu[0, 0], model.n_mu[0, 0], model.n_mu[1, 0]])
                        idx_noise = 0
                    if c_mode == 1:  # transition from Fallen to Fallen
                        offset[[6, 7, 8], 0] = np.array([model.n_mu[0, 1], model.n_mu[0, 1], model.n_mu[1, 1]])
                        idx_noise = 1
                else:  # transition to different mode
                    offset[[6, 7, 8], 0] = np.array([model.n_mu[0, 2], model.n_mu[0, 2], model.n_mu[0, 2]])
                    idx_noise = 2
                gm_e += self.gm_len[c_mode]
                m_per_mode = self.m + offset
                for l in range(gm_s, gm_e):
                    self.wg[gm_tmp_idx] = self.w[l]
                    # kalman filter prediction for a single component
                    self.mg[:, gm_tmp_idx] = np.dot(model.F, m_per_mode[:, l])
                    self.Pg[:, :, gm_tmp_idx] = model.Q[idx_noise] + np.dot(np.dot(model.F, self.P[:, :, l]), model.F.T)
                    gm_tmp_idx += 1
                gm_s = gm_e
                mode_predict_temp[c_mode] = model.mode_trans_matrix[c_mode, n_mode] + self.mode[c_mode]
            start_norm = gm_tmp_idx - sum(self.gm_len)
            self.wg[start_norm: gm_tmp_idx] -= logsumexp(self.wg[start_norm: gm_tmp_idx])
            mode_predict[n_mode] = logsumexp(mode_predict_temp)
        self.mode = mode_predict
        self.gm_len = np.ones(len(self.mode), dtype=int) * sum(self.gm_len)
        self.w = self.wg[:gm_tmp_idx]
        self.m = self.mg[:, :gm_tmp_idx]
        self.P = self.Pg[:, :, :gm_tmp_idx]

    def ukf_likelihood_per_sensor(self, z, s, model):
        z_bbox, z_feat = z[:4], norm_feat01(z[5:])
        for_cost = np.zeros(len(self.mode))
        gm_s = 0
        gm_e = 0
        for mode in range(len(self.mode)):
            ratio = np.exp(z_bbox[3] - z_bbox[2])
            if mode == 0:
                prob = 1 * (ratio - 1)
            if mode == 1:
                prob = -1 * (ratio - 1)
            q_mode = prob + self.mode[mode]
            q_z = np.zeros(self.gm_len[mode])
            gm_e += self.gm_len[mode]
            for idxp in range(gm_s, gm_e):
                m, P = self.m[:, idxp][:, np.newaxis], self.P[:, :, idxp]
                qz_temp, _, _ = ukf_update_per_sensor(z_bbox[:, np.newaxis], m, P, s, mode, model)
                q_z[idxp - gm_s] = qz_temp
            w_temp = q_z + self.w[gm_s:gm_e]
            for_cost[mode] = q_mode + logsumexp(w_temp)
            gm_s = gm_e
        pm_temp = 0
        if self.use_feat:
            dim = len(self.feat[s, :])
            x_z = self.feat[s, :] - z_feat
            xz_square = sum(x_z * x_z)
            xsum_temp = np.sum(self.feat[s, :] * self.feat[s, :]) - sum(self.feat[s, :])
            no_change = np.log(4 * dim - xz_square) - (np.log(11 * dim / 3 - xsum_temp))
            change = np.log(xz_square) - (np.log(dim / 3 + xsum_temp))
            pm_temp = logsumexp([np.log(0.9) + no_change, np.log(0.1) + change])
        return logsumexp(for_cost) + pm_temp

    def ukf_update(self, z, nestmeasidxs, model, time_step):
        for_cost = np.zeros(len(self.mode))
        tt = deepcopy(self)
        tt.ah.append(np.append(time_step, nestmeasidxs))
        nestmeasidxs = nestmeasidxs - 1  # restore original measurement index 0-|Z|
        gm_s = 0
        gm_e = 0
        for mode in range(len(tt.mode)):
            q_mode = np.zeros(model.N_sensors)
            q_z = np.zeros((model.N_sensors, tt.gm_len[mode]))
            gm_e += self.gm_len[mode]
            for q, idx in enumerate(nestmeasidxs):
                if idx < 0:
                    continue
                z_bbox = z[q][:4, :]
                ratio = np.exp(z_bbox[3, idx] - z_bbox[2, idx])
                if mode == 0:
                    prob = 1 * (ratio - 1)
                if mode == 1:
                    prob = -1 * (ratio - 1)
                q_mode[q] = prob + tt.mode[mode]
                for idxp in range(gm_s, gm_e):
                    m, P = tt.m[:, idxp:idxp + 1], tt.P[:, :, idxp]
                    qt, mt, Pt = ukf_update_per_sensor(z_bbox[:, idx:idx + 1], m, P, q, mode, model)
                    q_z[q, idxp - gm_s] = qt
                    tt.m[:, idxp:idxp + 1] = mt
                    tt.P[:, :, idxp] = Pt
            w_temp = np.sum(q_z, axis=0) + tt.w[gm_s:gm_e]
            tt.w[gm_s:gm_e] = sum(q_mode) + w_temp
            for_cost[mode] = sum(q_mode) + logsumexp(w_temp)
            gm_s = gm_e
        gm_s = 0
        gm_e = 0
        for mode in range(len(tt.mode)):
            gm_e += self.gm_len[mode]
            tt.w[gm_s:gm_e] = tt.w[gm_s:gm_e] - logsumexp(for_cost)
            tt.mode[mode] = logsumexp(tt.w[gm_s:gm_e])
            tt.w[gm_s:gm_e] = tt.w[gm_s:gm_e] - tt.mode[mode]
            gm_s = gm_e
        # Update re-identification feature information
        tt.last_active = time_step
        pm_temp = 0
        for q, idx in enumerate(nestmeasidxs):
            if idx < 0 or not self.use_feat:
                continue
            z_feat = norm_feat01(z[q][5:, idx])
            dim = len(self.feat[q, :])
            x_z = z_feat - self.feat[q, :]
            xz_square = sum(x_z * x_z)
            xsum_temp = np.sum(self.feat[q, :] * self.feat[q, :]) - sum(self.feat[q, :])
            # NO CHANGE f(z|x) = sum_i (4 - (x_i - z_i)^2)  => g(z | x, varrho=0) = f(z|x) / \int (f(z|x)dz)
            no_change = np.log(4 * dim - xz_square) - (np.log(11 * dim / 3 - xsum_temp))
            # CHANGE f(z|x) = sum_i ((x_i - z_i)^2)
            change = np.log(xz_square) - (np.log(dim / 3 + xsum_temp))
            pm_temp += logsumexp([np.log(0.9) + no_change, np.log(0.1) + change])
            if tt.feat_flag[q]:
                tt.feat[q, :] = z_feat
                tt.feat_flag[q] = False
            else:
                feat_temp = tt.alpha_feat * tt.feat[q, :] + (1 - tt.alpha_feat) * z_feat
                tt.feat[q, :] = norm_feat01(feat_temp)
        return logsumexp(for_cost) + pm_temp, tt

    def gate_msmeas_ukf(self, model, gamma, meas, alpha=1, kappa=2, beta=2):
        for s in range(model.N_sensors):
            z = meas[s][:4, :]
            zlength = z.shape[1]
            self.gatemeas[s] = []
            if zlength == 0:
                continue
            gm_s = 0
            gm_e = 0
            for mode in range(len(self.mode)):
                gm_e += self.gm_len[mode]
                for idxp in range(gm_s, gm_e):
                    m = np.append(self.m[:, idxp:idxp + 1], np.zeros((model.z_dim, 1)), axis=0)
                    P = block_diag(*[self.P[:, :, idxp], model.R[mode]])
                    X_ukf, u = ut(m, P, alpha, kappa)
                    noise = X_ukf[model.x_dim:model.x_dim + model.z_dim, :]
                    Z_pred = gen_msobservation_fn_v2(model, X_ukf[0:model.x_dim, :], noise, model.meas_n_mu[:, 0], s)
                    eta = np.dot(Z_pred, u)
                    Sj_temp = Z_pred - np.tile(eta, (1, len(u)))
                    u[0] = u[0] + (1 - alpha ** 2 + beta)
                    Sj = np.linalg.multi_dot([Sj_temp, np.diagflat(u), Sj_temp.T])
                    Vs = cholesky(Sj)
                    inv_sqrt_Sj = np.linalg.inv(Vs)
                    nu = z - eta
                    dist = sum(np.power(np.dot(inv_sqrt_Sj.T, nu), 2))
                    for idxg, idist in enumerate(dist):
                        if idist < gamma:
                            self.gatemeas[s].append(idxg)
        # END

    def gating_feet(self, model, meas, gating_threshold=0.8):  # Euclid distance
        for s in range(model.N_sensors):
            bboxes = meas[s][:4, :]
            zlength = bboxes.shape[1]
            self.gatemeas[s] = []
            if zlength == 0:
                continue
            feet_loc = np.copy(bboxes[0:2, :])
            feet_loc[0, :] = bboxes[0, :] + np.exp(bboxes[2, :]) / 2
            feet_loc[1, :] = bboxes[1, :] + np.exp(bboxes[3, :])
            feet_loc_gp = homtrans(np.linalg.inv(model.cam_mat[s][:, [0, 1, 3]]), feet_loc)

            # self.gatemeas[s] = -1 * np.ones(bboxes.shape[1])
            gm_s = 0
            gm_e = 0
            dists = []
            for mode in range(len(self.mode)):
                gm_e += self.gm_len[mode]
                for idxp in range(gm_s, gm_e):
                    m_feet = self.m[[0, 2], idxp:idxp + 1]
                    euclid_dis = np.linalg.norm(feet_loc_gp - m_feet, axis=0)
                    dists.append(euclid_dis)
            dists = np.logical_or.reduce(np.row_stack(dists) < gating_threshold, 0)
            self.gatemeas[s] = np.nonzero(dists)[0]
        # END

    def not_gating(self, model, z):
        for s in range(model.N_sensors):
            self.gatemeas[s] = list(range(z[s].shape[1]))

    def compute_pS(self, model, k):
        ind = np.argmax(self.mode)
        start = sum(self.gm_len[0:ind])
        end = start + self.gm_len[ind]
        indx = np.argmax(self.w[start:end])
        X = self.m[:, start + indx]
        control_param = 0.6
        ch1 = model.XMAX[0] < X[0] < model.XMAX[1]
        ch2 = model.YMAX[0] < X[2] < model.YMAX[1]
        if ch1 and ch2:
            scene_mask = model.P_S
        else:
            scene_mask = 1 - model.P_S
        pS = scene_mask / (1 + np.exp(-control_param * (k - self.birth_time)))
        return pS

    def cleanup(self):
        w, m, P = np.copy(self.w), np.copy(self.m), np.copy(self.P)
        s_old = 0
        e_old = 0
        s_new = 0
        e_new = 0
        gm_len = np.zeros(len(self.mode), dtype=int)
        for mode in range(len(self.mode)):
            e_old += self.gm_len[mode]
            wc, mc, Pc = cleanup(self.w[s_old:e_old], self.m[:, s_old:e_old], self.P[:, :, s_old:e_old])
            e_new += len(wc)
            w[s_new:e_new], m[:, s_new:e_new], P[:, :, s_new:e_new] = wc, mc, Pc
            gm_len[mode] = len(wc)
            s_old = e_old
            s_new = e_new
        self.gm_len = gm_len
        self.w, self.m, self.P = w[:e_new], m[:, :e_new], P[:, :, :e_new]

    def re_activate(self, m, P, rb, feat, model, time_step):
        self.m[0, :m.shape[1]], self.m[2, :m.shape[1]] = m[0, :], m[2, :]
        self.P = P
        self.w = np.copy(model.w_birth)
        self.r = rb
        self.feat_flag = (np.sum(feat, axis=1) == 0)  # checking whether initial feature is missed
        average_feat = np.sum(feat, axis=0) / sum(1 - self.feat_flag)
        average_feat = norm_feat01(average_feat)
        for q in range(model.N_sensors):
            if self.feat_flag[q]:
                self.feat[q, :] = average_feat
            else:
                feat_temp = self.re_alpha_feat * self.feat[q, :] + (1 - self.re_alpha_feat) * norm_feat01(feat[q, :])
                self.feat[q, :] = norm_feat01(feat_temp)
        self.mode = np.copy(model.mode)
        self.P_S = model.P_S
        self.gatemeas = [np.array([]) for i in range(model.N_sensors)]
        self.last_active = time_step
        self.gm_len = np.ones(len(model.mode), dtype=int)
    #  END
