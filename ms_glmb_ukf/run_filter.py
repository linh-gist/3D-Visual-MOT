import os
from scipy.spatial.distance import cdist
from scipy.stats.distributions import chi2
import numpy as np
from scipy.special import logsumexp
import lap
import time

from cppmsglmb import gibbs_multisensor_approx_cheap, gibbs_multisensor_approx_dprobsample, MCAdaptiveBirth, \
    bboxes_ioi_xyah_back2front_all, bboxes_ioi_xyah_back2front_all_v2, MSGLMB, meanShift, KMeans, multisensor_lapjv
from target import Target, gen_msobservation_fn_v2, norm_feat01, ukf_update_per_sensor
from gen_model import lognormal_with_mean_one


class filter:
    # filter parameters
    def __init__(self, model):
        self.H_upd = 20000  # requested number of updated components/hypotheses
        self.H_max = 8000  # cap on number of posterior components/hypotheses
        self.hyp_threshold = np.log(1e-5)  # pruning threshold for components/hypotheses

        self.L_max = 100  # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold = 1e-5  # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold = 4  # merging threshold for Gaussians in each track - not implemented yet

        self.P_G = 0.999999999999  # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, model.z_dim)  # inv chi^2 dn gamma value
        self.gate_flag = 0  # gating on or off 1/0

        # UKF parameters
        # scale parameter for UKF - choose alpha=1 ensuring lambda=beta and offset of first cov weight is beta
        # for numerical stability
        self.ukf_alpha = 1
        self.ukf_beta = 2  # scale parameter for UKF
        # scale parameter for UKF (alpha=1 preferred for stability, giving lambda=1, offset of beta
        # for first cov weight)
        self.ukf_kappa = 2

        self.run_flag = 'disp'  # 'disp' or 'silence' for on the fly output


class Estimate:
    def __init__(self):
        self.X = {}
        self.N = {}
        self.L = {}
        self.S = {}


'''
ndsub2ind function converts subscripts to linear indices (similar to Matlab sub2ind but for N dimension)
It converts the multi-dimensional indices in `idx` to a linear index and returns the resulting array as `linidx`.

siz [3, 2], idx [2, 1] => we want to obtain index of element (2, 1) in Matrix size 3x2
arr = np.array([[0, 3],
                [1, 4],
                [2, 5]]), result is arr[2, 1] = 5

siz [2, 3, 2], idx [0, 2, 1] => we want to obtain index of element (0, 2, 1) in Matrix size 2x3x2
arr = np.array([[[0, 3], [1, 4], [2, 5]], 
                [[6, 9], [7, 10], [8, 11]]])
The result is 5, but if idx [1, 2, 1], result is 11. If idx is [[0, 2, 1], [1, 2, 1]], result is [5, 11]

=> It can be replaced by numpy.ravel_multi_index(multi_index, dims, mode='raise', order='C')
'''


def ndsub2ind(siz, idx):
    linidx = np.array([])
    if len(idx) == 0:
        return linidx
    else:
        linidx = idx[:, 0].astype(np.int64)
        idx = idx.astype(np.int64)
        totals = np.cumprod(siz, dtype=np.int64)
        for i in range(1, idx.shape[1]):
            linidx = linidx + (idx[:, i]) * totals[i - 1]
    return linidx


def detection_aka_occlusion_model(tindices_tt, model, q, ini_model_check, occ_model_on=True):
    sensor_pos = model.sensor_pos[q]
    cam_mat = model.cam_mat[q]
    num_of_objects = len(tindices_tt)
    if num_of_objects == 0:
        return np.array([])
    X = np.zeros((model.x_dim, num_of_objects))
    for a, tt in enumerate(tindices_tt):  # Evaluate MAP of GM
        ind = np.argmax(tt.mode)
        start = sum(tt.gm_len[0:ind])
        end = start + tt.gm_len[ind]
        indx = np.argmax(tt.w[start:end])
        X[:, a] = tt.m[:, start + indx]
    X[[6, 7, 8], :] = np.exp(X[[6, 7, 8], :])
    pD_test = np.ones(num_of_objects)
    if not (ini_model_check > 0 and occ_model_on):
        return pD_test * model.P_D

    P_3 = X[[0, 2, 4], :]
    dist = np.linalg.norm(P_3 - sensor_pos[:, np.newaxis], axis=0)
    indx = np.argsort(dist)
    check_pool = np.ones(num_of_objects, dtype=int)  # give every object a flag
    for j in range(num_of_objects):  # start looping through objects in the hypothesis
        if check_pool[j]:  # check which object has not been evaluated
            temp = np.dot(cam_mat, np.append(P_3[:, indx[j]], 1))
            test_point_Img_plane = temp[[0, 1]] / temp[2]
            f1 = test_point_Img_plane[0] <= 0
            f2 = test_point_Img_plane[0] >= model.imagesize[0]
            f3 = test_point_Img_plane[1] <= 0
            f4 = test_point_Img_plane[1] >= model.imagesize[1]
            if f1 or f2 or f3 or f4:
                pD_test[indx[j]] = model.Q_D  # if object is not in the image, then assign low pd.
            else:
                pD_test[indx[j]] = model.P_D  # object is in the image. high pd.
            check_pool[j] = 0  # unchecked the object when pd has been assigned
            curr_object = X[:, indx[j]]
            curr_object_centroid = curr_object[[0, 2, 4]]
            curr_object_rx = curr_object[6]
            curr_object_ry = curr_object[7]
            curr_object_h = curr_object[8]
            '''
            % Ellipsoid can be represented as
            v'*A*v + b*x + c = 0
            the idea is to represent/substitute v as a line emanating from sensor_pos. 
            Equation of line in 3D is v(t) = e + t*d where t>=0 and d is a directional unit vector
            This yields: \alpha*k^2 + \beta*k + \gamma, where \alpha = d'*A*d
            \beta = b*d + 2*d'*A*e, \gamma = e'*A*e + b*e + c
            (see: https://www.geometrictools.com/Documentation/PerspectiveProjectionEllipsoid.pdf )
            if the above has two distinct real roots, then the ray between the sensor and object to be evaluated,
            intersects the chosen/current object twice. This confirms object being evaluated is occluded by 
            chosen/current object.
            '''
            A = np.array([[1 / curr_object_rx ** 2, 0, 0],
                          [0, 1 / curr_object_ry ** 2, 0],
                          [0, 0, 1 / curr_object_h ** 2]])
            # calculating b and c
            eig_vals_vec, right_eig = np.linalg.eig(A)
            temp_ellip_c = np.dot(right_eig, curr_object_centroid)
            ggs = -2 * temp_ellip_c * eig_vals_vec
            b_desired = np.dot(ggs.T, right_eig)  # this is b
            J_const = np.sum(np.divide(np.power(ggs.T, 2), 4 * eig_vals_vec))
            for i, value in enumerate(check_pool):
                if value == 0:
                    continue
                evaluate = P_3[:, indx[i]]
                vector_evaluate = evaluate - sensor_pos
                vector_evaluate = vector_evaluate / np.linalg.norm(vector_evaluate)
                alpha = np.dot(np.dot(vector_evaluate.T, A), vector_evaluate)
                beta = np.dot(b_desired, vector_evaluate) + 2 * np.dot(np.dot(vector_evaluate.T, A), sensor_pos)
                gamma = np.dot(np.dot(sensor_pos.T, A), sensor_pos) + np.dot(b_desired, sensor_pos) + (-1 + J_const)
                root = np.roots([alpha, beta, gamma])
                if np.all(np.isreal(root)):
                    check_pool[i] = 0
                    pD_test[indx[i]] = model.Q_D
    if sum(check_pool) > 0:
        print('Not all states are checked - P_d are not assigned entirely!!!')
    return pD_test


def detection_aka_occlusion_model_v2(tindices_tt, model, q, ini_model_check, occ_model_on=True):
    # project target state => compute Intersection Over Area (back2front)
    num_of_objects = len(tindices_tt)
    if num_of_objects == 0:
        return np.array([])
    X = np.zeros((model.x_dim, num_of_objects))
    for a, tt in enumerate(tindices_tt):  # Evaluate MAP of GM
        ind = np.argmax(tt.mode)
        start = sum(tt.gm_len[0:ind])
        end = start + tt.gm_len[ind]
        indx = np.argmax(tt.w[start:end])
        X[:, a] = tt.m[:, start + indx]
    pD_test = np.ones(num_of_objects)
    if not (ini_model_check > 0 and occ_model_on):
        return pD_test * model.P_D
    x2bbox = gen_msobservation_fn_v2(model, X, np.zeros((4, 1)), np.zeros(2), q)
    ioa_temp = bboxes_ioi_xyah_back2front_all(x2bbox.T)
    ioa_temp = 1 - np.max(ioa_temp, axis=1)
    # ioa_temp = bboxes_ioi_xyah_back2front_all_v2(x2bbox.T)
    pD_test = np.clip(ioa_temp, 1 - model.P_D, model.P_D)

    return pD_test


class GLMB:  # delta GLMB
    # Choose the following options (adaptive_birth = 0 [Fix birth], 1 [Monte Carlo], 2 [MeanShift])
    # (0) Fix birth uses re-id feature => [adaptive_birth=0, use_feat=False] => ONLY for CMC dataset
    # (1.1) Monte Carlo Adaptive birth uses re-id feature [adaptive_birth=1, use_feat=True]
    # (1.2) Monte Carlo Adaptive birth does NOT use re-id feature [adaptive_birth=1, use_feat=False]
    # (2.1) KMeans Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
    # (2.2) KMeans Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
    # (3.1) MeanShift Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
    # (3.2) MeanShift Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
    def __init__(self, model, adaptive_birth=1, use_feat=True):
        # initial Numpy Data type for target
        self.x_dim = 4

        self.glmb_update_tt = []  # 1, track table for GLMB
        self.glmb_update_w = np.array([0])  # 2, vector of GLMB component/hypothesis weights
        self.glmb_update_I = [np.array([], dtype=int)]  # 3, cell of GLMB component/hypothesis labels
        self.glmb_update_n = np.array([0])  # 4, vector of GLMB component/hypothesis cardinalities
        self.glmb_update_cdn = np.array([1])  # 5, cardinality distribution of GLMB
        np.set_printoptions(linewidth=2048)  # use in clean_predict() to convert array2string

        self.p_birth = 0.001
        self.tt_birth = []
        # self.MSGLMB = MSGLMB(model.cam_mat)
        self.est = Estimate()
        # Monte Carlo Adaptive Birth for LRFS
        mc_pdf_c = np.log(1.0 / ((model.XMAX[1] - model.XMAX[0]) * (model.YMAX[1] - model.YMAX[0])))
        self.MCAB = MCAdaptiveBirth(model.cam_mat, 100, 500, model.P_D, model.lambda_c[0, 0], mc_pdf_c)
        self.MCAB.setMeasureNoise(model.R[0], model.meas_n_mu[:, 0])
        self.MCAB.setBirthProb(model.rB_min, model.rB_max)
        self.MCAB.setNumSensorDetect(model.num_det)

        self.id = 0
        self.use_feat = use_feat
        self.adaptive_birth = adaptive_birth
        self.feat_dim = 128
        self.prev_tt_glmb_labels = np.array([])
        self.prev_glmb_update_tt = []
        self.pruned_tt = []
        self.pruned_tt_feat = []
        self.pruned_tt_label = []

    def msjointpredictupdate(self, model, filter, meas, k):
        # create birth tracks
        if k == 0:
            meas_ru, meas_Z = [], []
            set_feat_dim = True
            for bbox in meas:
                if bbox.shape[1] > 0 and set_feat_dim:
                    self.feat_dim = bbox.shape[0] - 5  # 4 +1 (bbox[ltwh] + conf)
                    set_feat_dim = False
                bbox = bbox[:4, :]
                temp = 0.99 * np.ones(bbox.shape[1] + 1)
                temp[0] = 1  # for miss-detection
                meas_ru.append(temp)
                meas_Z.append(bbox.copy())
            self.generate_birth(meas_ru, meas_Z, meas, model, k)

        # create surviving tracks - via time prediction (single target CK)
        for tt in self.glmb_update_tt:
            tt.predict(model)

        # create predicted tracks - concatenation of birth and survival
        glmb_predict_tt = self.tt_birth + self.glmb_update_tt  # copy track table back to GLMB struct

        # gating by tracks
        if filter.gate_flag:
            for tt in glmb_predict_tt:
                # tt.gate_msmeas_ukf(model, filter.gamma, meas)
                tt.gating_feet(model, meas)
        else:
            for tt in glmb_predict_tt:
                tt.not_gating(model, meas)

        # precalculation loop for average survival/death probabilities
        avps = np.array([tt.r for tt in self.tt_birth] +
                        [tt.compute_pS(model, k) for tt in self.glmb_update_tt])[:, np.newaxis]
        avqs = np.log(1 - avps)
        avps = np.log(avps)

        # create updated tracks (single target Bayes update)
        m = np.zeros(model.N_sensors, dtype=int)
        for s in range(model.N_sensors):
            m[s] = meas[s].shape[1]  # number of measurements

        # nested for loop over all predicted tracks and sensors - slow way
        # Kalman updates on the same prior recalculate all quantities
        # extra state for not detected for each sensor (meas "0" in pos 1)
        allcostc = [np.NINF * np.ones((len(glmb_predict_tt), 1 + m[s])) for s in range(model.N_sensors)]
        # extra state for not detected for each sensor (meas "0" in pos 1)
        jointcostc = [np.NINF * np.ones((len(glmb_predict_tt), 1 + m[s])) for s in range(model.N_sensors)]
        for s in range(model.N_sensors):
            # allcostc[s] # extra state for not detected for each sensor (meas "0" in pos 1)
            # jointcostc[s] # extra state for not detected for each sensor (meas "0" in pos 1)
            for tabidx, tt in enumerate(glmb_predict_tt):
                for emm in tt.gatemeas[s]:
                    w_temp = tt.ukf_likelihood_per_sensor(meas[s][:, emm], s, model)
                    allcostc[s][tabidx, 1 + emm] = w_temp  # predictive likelihood
            # survived and missed detection | survived and detected targets
            jointcostc[s] = np.tile(avps, (1, m[s] + 1)) + allcostc[s] - (model.lambda_c[s] + model.pdf_c[s])
            jointcostc[s][:, 0] = avps[:, 0]

        # gated measurement index matrix
        gatemeasidxs = [-1 * np.ones((len(glmb_predict_tt), m[s]), dtype="int") for s in range(model.N_sensors)]
        for s in range(model.N_sensors):
            for tabidx, tt in enumerate(glmb_predict_tt):
                gatemeasidxs[s][tabidx, tt.gatemeas[s]] = tt.gatemeas[s]

        # component updates
        runidx = 0
        glmb_nextupdate_w = np.zeros(filter.H_upd * 2)
        glmb_nextupdate_I = []
        glmb_nextupdate_n = np.zeros(filter.H_upd * 2, dtype=int)
        # use to normalize assign_prob using glmb_nextupdate_w, first raw for "measurement" missed detection
        assign_meas = [np.zeros((m[s] + 1, filter.H_upd * 2), dtype=int) for s in range(model.N_sensors)]
        cpreds = len(glmb_predict_tt)
        nbirths = len(self.tt_birth)
        hypoth_num = logsumexp(0.5 * self.glmb_update_w)
        hypoth_num = np.rint(np.exp(np.log(filter.H_upd) + 0.5 * self.glmb_update_w - hypoth_num)).astype(int)

        tt_update_parent = np.array([], dtype="int")
        tt_update_currah = np.empty((0, model.N_sensors), dtype="int")
        tt_update_linidx = np.array([], dtype=np.int64)

        for pidx in range(0, len(self.glmb_update_w)):
            # calculate best updated hypotheses/components
            # indices of all births and existing tracks  for current component
            tindices = np.concatenate((np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx]))
            mindices = []
            tindices_tt = [glmb_predict_tt[i] for i in tindices]
            avpd = np.zeros((len(tindices), model.N_sensors))
            for s in range(model.N_sensors):
                # union indices of gated measurements for corresponding tracks
                ms_indices = np.unique(gatemeasidxs[s][tindices, :])
                if -1 in ms_indices:
                    ms_indices = ms_indices[1:]
                mindices.append(np.insert(1 + ms_indices, 0, 0).astype("int"))
                avpd[:, s] = detection_aka_occlusion_model_v2(tindices_tt, model, s, hypoth_num[pidx])
            avqd = np.log(1 - avpd)
            avpd = np.log(avpd)
            costc = []
            avpp = np.zeros((len(tindices), model.N_sensors))
            for s in range(model.N_sensors):
                take_rows = jointcostc[s][tindices]
                jointcostc_pidx = take_rows[:, mindices[s]]
                jointcostc_pidx[:, 0] = avqd[:, s] + jointcostc_pidx[:, 0]
                jointcostc_pidx[:, 1:] = avpd[:, s:s + 1] + jointcostc_pidx[:, 1:]
                avpp[:, s] = logsumexp(jointcostc_pidx, axis=1)
                costc.append(np.exp(jointcostc_pidx))
            dcost = avqs[tindices]  # death cost
            # scost = np.sum(avpp, axis=1)[:, np.newaxis]  # posterior survival cost
            # dprob = np.exp(dcost - np.logaddexp(dcost, scost))

            uasses = gibbs_multisensor_approx_dprobsample(np.exp(dcost), costc, hypoth_num[pidx])
            # uasses = multisensor_lapjv(costc, 15)  # single hypothesis
            # uasses = gibbs_multisensor_approx_cheap(dprob, costc, hypoth_num[pidx])
            # uasses = np.array(uasses, dtype="f8")
            # uasses[uasses < 0] = -np.inf  # set not born/track deaths to -inf assignment

            # generate corrresponding jointly predicted/updated hypotheses/components
            for hidx in range(len(uasses)):
                update_hypcmp_tmp = uasses[hidx]
                off_idx = update_hypcmp_tmp[:, 0] < 0
                aug_idx = np.column_stack((tindices, update_hypcmp_tmp))  # [tindices, 1 + update_hypcmp_tmp]
                mis_idx = update_hypcmp_tmp == 0
                det_idx = update_hypcmp_tmp > 0
                local_avpdm = avpd
                local_avqdm = avqd
                # re-compute PD after sampling, we can either re-compute or NOT
                # tindices_tt = []
                # for idx_tt, value_tt in enumerate(off_idx):
                #     if not value_tt:
                #         tindices_tt.append(glmb_predict_tt[idx_tt])
                # for s in range(model.N_sensors):
                #     avpd_temp = detection_aka_occlusion_model(tindices_tt, model, s, hypoth_num[pidx])
                #     local_avqdm[~off_idx, s] = np.log(1 - avpd_temp)
                #     local_avpdm[~off_idx, s] = np.log(avpd_temp)
                repeated_lambda_c = np.tile(model.lambda_c.T, (len(tindices), 1))
                repeated_pdf_c = np.tile(model.pdf_c.T, (len(tindices), 1))
                update_hypcmp_idx = np.zeros(len(off_idx), dtype=np.int64)
                update_hypcmp_idx[off_idx] = -1  # -np.inf
                # update_hypcmp_idx[~off_idx] = ndsub2ind(np.insert(1 + m, 0, cpreds), aug_idx[~off_idx, :])
                update_hypcmp_idx[~off_idx] = np.ravel_multi_index(aug_idx[~off_idx, :].T, np.insert(1 + m, 0, cpreds))
                num_trk = sum(update_hypcmp_idx >= 0)

                sum_temp = m[np.newaxis, :] @ (model.lambda_c + model.pdf_c)
                sum_temp += sum(avps[tindices[~off_idx]]) + sum(avqs[tindices[off_idx]])
                sum_temp += sum(local_avpdm[det_idx]) + sum(local_avqdm[mis_idx])
                sum_temp -= sum(repeated_lambda_c[det_idx] + repeated_pdf_c[det_idx])
                glmb_nextupdate_w[runidx] = sum_temp + self.glmb_update_w[pidx]  # hypothesis/component weight

                # Get measurement index from uasses (make sure minus 1 from [mindices+1])
                for s in range(model.N_sensors):
                    miss_idx = update_hypcmp_tmp[mis_idx[:, s], s].astype(int)  # first raw for missed detection
                    meas_idx = update_hypcmp_tmp[det_idx[:, s], s].astype(int)
                    assign_meas[s][miss_idx, runidx] = 1
                    assign_meas[s][meas_idx, runidx] = 1  # Setting index of measurements associate with a track

                if num_trk > 0:
                    # hypothesis/component tracks (via indices to track table)
                    glmb_nextupdate_I.append(np.arange(len(tt_update_parent), len(tt_update_parent) + num_trk))
                else:
                    glmb_nextupdate_I.append([])
                glmb_nextupdate_n[runidx] = num_trk  # hypothesis/component cardinality
                runidx = runidx + 1

                tt_update_parent = np.append(tt_update_parent, tindices[~off_idx])
                tt_update_currah = np.row_stack((tt_update_currah, update_hypcmp_tmp[~off_idx, :].astype("int")))
                tt_update_linidx = np.append(tt_update_linidx, update_hypcmp_idx[update_hypcmp_idx >= 0])
        # END

        # component updates via posterior weight correction (including generation of track table)
        ttU_allkey, ttU_oldidx, ttU_newidx = np.unique(tt_update_linidx, return_index=True, return_inverse=True)
        tt_update_msqz = np.zeros((len(ttU_allkey), 1))
        tt_update = []
        for tabidx in range(len(ttU_allkey)):
            oldidx = ttU_oldidx[tabidx]
            preidx = tt_update_parent[oldidx]
            meascomb = tt_update_currah[oldidx, :]

            # kalman update for this track and all joint measurements
            qz_temp, tt = glmb_predict_tt[preidx].ukf_update(meas, meascomb, model, k)
            tt_update_msqz[tabidx] = qz_temp

            tt_update.append(tt)
        # END

        for pidx in range(runidx):
            glmb_nextupdate_I[pidx] = ttU_newidx[glmb_nextupdate_I[pidx]]
            glmb_nextupdate_w[pidx] = glmb_nextupdate_w[pidx] + sum(tt_update_msqz[glmb_nextupdate_I[pidx]])

        # normalize weights
        glmb_nextupdate_w = glmb_nextupdate_w[:runidx]
        glmb_nextupdate_n = glmb_nextupdate_n[:runidx]
        glmb_nextupdate_w = glmb_nextupdate_w - logsumexp(glmb_nextupdate_w)

        # Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking
        meas_ru, meas_Z = [], []
        for s, bbox in enumerate(meas):
            rA = assign_meas[s][:, :runidx] @ np.exp(glmb_nextupdate_w)  # adaptive birth weight for each measurement
            rA = np.clip(rA, np.spacing(0), 1 - np.spacing(0))
            # By notation, let rA,+(0) = 0 which is intuitive as it suggests that a missed detection
            # did not associate with any tracks in the existing hypotheses.
            rA[0] = 0
            bbox = bbox[:4, :]
            meas_ru.append(1 - rA)  # rU
            meas_Z.append(bbox)
        self.generate_birth(meas_ru, meas_Z, meas, model, k)

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.NINF * np.ones(max(glmb_nextupdate_n) + 1)
        for card in range(0, max(glmb_nextupdate_n) + 1):
            # extract probability of n targets
            idx = (glmb_nextupdate_n == card)
            if sum(idx) > 0:
                glmb_nextupdate_cdn[card] = logsumexp(glmb_nextupdate_w[idx])
        glmb_nextupdate_cdn = np.exp(glmb_nextupdate_cdn)

        # copy glmb update to the next time step
        self.glmb_update_tt = tt_update  # 1, copy track table back to GLMB struct
        self.glmb_update_w = glmb_nextupdate_w  # 2
        self.glmb_update_I = glmb_nextupdate_I  # 3
        self.glmb_update_n = glmb_nextupdate_n  # 4
        self.glmb_update_cdn = glmb_nextupdate_cdn  # 5

        # remove duplicate entries and clean track table
        self.clean_predict()
        self.clean_update(k)

        for tt in self.glmb_update_tt:  # pruning, merging, capping Gaussian mixture components
            tt.cleanup()

    def mc_adaptive_birth(self, meas_ru, meas_z, model):
        # Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking
        m_birth, r_birth, sols_birth = self.MCAB.sample_adaptive_birth(meas_ru, meas_z)
        m_b_final, P_b_final = [], []
        for idx in range(len(r_birth)):
            m_mode = np.copy(model.m_birth)
            P_mode = np.copy(model.P_birth)
            for imode in range(len(model.mode)):
                if imode == 0:
                    m_mode[:, imode] = m_birth[idx]
                else:  # imode=1
                    m_mode[[0, 2], imode] = m_birth[idx][[0, 2]]
            m_b_final.append(m_mode)
            P_b_final.append(P_mode)
        return m_b_final, r_birth, P_b_final, sols_birth

    def mc_adaptive_birth_efficient(self, meas_ru, meas_z, model):
        # Only birth measurements with low association probability, e.g. rU > 0.9
        meas_ru_keep = []
        meas_z_keep = []
        num_meas = 0
        for idx, ru_sensor in enumerate(meas_ru):
            ru_keep = np.nonzero(ru_sensor > model.tau_ru)[0]
            meas_ru_keep.append(ru_sensor[ru_keep])
            meas_z_keep.append(meas_z[idx][:, ru_keep[1:] - 1])
            num_meas += len(ru_keep)
        if num_meas <= model.N_sensors:  # No measurement
            return [], [], [], []
        meas_ru, meas_z = meas_ru_keep, meas_z_keep
        sols, centroids = self.MCAB.sample_mc_sols(meas_ru, meas_z)
        sols = np.array(sols)
        r_b = []
        m_b = []
        P_b = []
        sols_keep = []
        for idx, sol in enumerate(sols):
            if sum(sol > 0) <= model.num_det:
                continue  # discarded solutions with few detections
            sol = np.array(sol) - 1  # restore original measurement index 0-|Z|
            m_temp = np.copy(model.m_birth[:, 0:1])
            m_temp[[0, 2], 0] = centroids[idx]
            Ptemp = np.copy(model.P_birth)[:, :, 0]
            q_z = np.zeros(model.N_sensors)
            for q, jdx in enumerate(sol):
                if jdx < 0:
                    continue
                qt, mt, Pt = ukf_update_per_sensor(meas_z[q][:, jdx:jdx + 1], m_temp, Ptemp, q, 0, model)
                q_z[q] = qt + np.log(meas_ru[q][jdx + 1] + np.spacing(0))
                m_temp = mt
                Ptemp = Pt
            r_b.append(np.sum(q_z))
            m_mode = np.copy(model.m_birth)
            P_mode = np.copy(model.P_birth)
            for imode in range(len(model.mode)):
                if imode == 0:
                    m_mode[:, imode:imode + 1] = m_temp
                else:  # imode=1
                    m_mode[[0, 2], imode:imode + 1] = m_temp[[0, 2]]
                P_mode[:, :, imode] = Ptemp
            m_b.append(m_mode)
            P_b.append(P_mode)
            sols_keep.append(sols[idx])
        r_b_final = []
        m_b_final = []
        P_b_final = []
        sols_final = []
        if len(r_b) > 0:
            r_b = np.array(r_b)
            r_b = np.exp(r_b - logsumexp(r_b))
        for idx in range(len(r_b)):  # prune low weight birth
            if r_b[idx] < model.rB_min:
                continue
            r_b_final.append(min(r_b[idx], model.rB_max))
            m_b_final.append(m_b[idx])
            P_b_final.append(P_b[idx])
            sols_final.append(sols_keep[idx])
        return m_b_final, r_b_final, P_b_final, sols_final

    def kmeans_adaptive_birth(self, meas_ru, meas_z, model):
        # Only birth measurements with low association probability, e.g. rU > 0.9
        meas_ru_keep = []
        meas_z_keep = []
        num_meas = 0
        for idx, ru_sensor in enumerate(meas_ru):
            ru_keep = np.nonzero(ru_sensor > model.tau_ru)[0]
            meas_ru_keep.append(ru_sensor[ru_keep])
            meas_z_keep.append(meas_z[idx][:, ru_keep[1:] - 1])
            num_meas += len(ru_keep)
        if num_meas <= model.N_sensors:  # No measurement
            return [], [], [], []
        # idxs ~ [[sensor index, measurement index, cluster index]]
        idxs, clusters = KMeans(1000).run(model.cam_mat, meas_z_keep)
        idxs = np.array(idxs)
        r_b = []
        m_b = []
        P_b = []
        sols = []
        lambda_b = 4  # The Labeled Multi-Bernoulli Filter, eq (75)
        for idx, cluster in enumerate(clusters):
            sol_idx = idxs[idxs[:, 2] == idx]
            sol = np.zeros(model.N_sensors, dtype=int)
            rB_temp = np.zeros(model.N_sensors)
            rU_temp = np.zeros(model.N_sensors)
            # idx_map ~ [sensor index, measurement index, cluster index]
            for idx_map in sol_idx:
                s, meas_idx = idx_map[0], idx_map[1] + 1
                if rU_temp[s] < meas_ru_keep[s][meas_idx]:  # if same sensor, replace with higher rU
                    rU_temp[s] = meas_ru_keep[s][meas_idx]
                    not_assigned_sum = sum(meas_ru_keep[s]) + np.spacing(1)
                    rB_temp[s] = min(model.rB_max, (meas_ru_keep[s][meas_idx]) / not_assigned_sum * lambda_b)
                    sol[s] = meas_idx
            m_mode = np.copy(model.m_birth)
            P_mode = np.copy(model.P_birth)
            for imode in range(len(model.mode)):
                m_mode[[0, 2], imode] = cluster
            r_b.append(max(rB_temp))
            m_b.append(m_mode)
            P_b.append(P_mode)
            sols.append(sol)
        return m_b, r_b, P_b, sols

    def meanshift_adaptive_birth(self, meas_ru, meas_z, model):
        # Only birth measurements with low association probability, e.g. rU > 0.9
        meas_ru_keep = []
        meas_z_keep = []
        num_meas = 0
        for idx, ru_sensor in enumerate(meas_ru):
            ru_keep = np.nonzero(ru_sensor > model.tau_ru)[0]
            meas_ru_keep.append(ru_sensor[ru_keep])
            meas_z_keep.append(meas_z[idx][:, ru_keep[1:] - 1])
            num_meas += len(ru_keep)
        if num_meas <= model.N_sensors:  # No measurement
            return [], [], [], []
        meas_ru, meas_z = meas_ru_keep, meas_z_keep
        sols, centroids = meanShift(model.cam_mat, meas_z, 0.6)
        sols = np.array(sols)
        r_b = []
        m_b = []
        P_b = []
        sols_keep = []
        for idx, sol in enumerate(sols):
            if sum(sol > 0) <= model.num_det:
                continue  # discarded solutions with few detections
            sol = np.array(sol) - 1  # restore original measurement index 0-|Z|
            m_temp = np.copy(model.m_birth[:, 0:1])
            m_temp[[0, 2], 0] = centroids[idx]
            Ptemp = np.copy(model.P_birth)[:, :, 0]
            q_z = np.zeros(model.N_sensors)
            for q, jdx in enumerate(sol):
                if jdx < 0:
                    continue
                qt, mt, Pt = ukf_update_per_sensor(meas_z[q][:, jdx:jdx + 1], m_temp, Ptemp, q, 0, model)
                q_z[q] = qt + np.log(meas_ru[q][jdx + 1] + np.spacing(0))
                m_temp = mt
                Ptemp = Pt
            r_b.append(np.sum(q_z))
            m_mode = np.copy(model.m_birth)
            P_mode = np.copy(model.P_birth)
            for imode in range(len(model.mode)):
                if imode == 0:
                    m_mode[:, imode:imode + 1] = m_temp
                else:  # imode=1
                    m_mode[[0, 2], imode:imode + 1] = m_temp[[0, 2]]
                P_mode[:, :, imode] = Ptemp
            m_b.append(m_mode)
            P_b.append(P_mode)
            sols_keep.append(sols[idx])
        r_b_final = []
        m_b_final = []
        P_b_final = []
        sols_final = []
        if len(r_b) > 0:
            r_b = np.array(r_b)
            r_b = np.exp(r_b - logsumexp(r_b))
        for idx in range(len(r_b)):  # prune low weight birth
            if r_b[idx] < model.rB_min:
                continue
            r_b_final.append(min(r_b[idx], model.rB_max))
            m_b_final.append(m_b[idx])
            P_b_final.append(P_b[idx])
            sols_final.append(sols_keep[idx])
        return m_b_final, r_b_final, P_b_final, sols_final

    # (adaptive_birth = 0[Fix birth], 1[Monte Carlo], 2[Kmeans], 3[MeanShift])
    def generate_birth(self, meas_ru, meas_z, meas, model, k):
        self.tt_birth = []

        # (1) fix birth
        if self.adaptive_birth == 0:
            for tabbidx in range(len(model.r_birth)):
                m_temp, r_b, P_b = np.copy(model.m_birth), model.r_birth[tabbidx], model.P_birth
                tt = Target(m_temp, P_b, r_b, self.id, None, model, k, False)
                self.id += 1
                self.tt_birth.append(tt)
            return

        # (2) adaptive birth
        if self.adaptive_birth == 1:
            m_birth, r_birth, P_birth, sols_birth = self.mc_adaptive_birth_efficient(meas_ru, meas_z, model)
        elif self.adaptive_birth == 2:
            m_birth, r_birth, P_birth, sols_birth = self.kmeans_adaptive_birth(meas_ru, meas_z, model)
        else:  # self.adaptive_birth == 3:
            m_birth, r_birth, P_birth, sols_birth = self.meanshift_adaptive_birth(meas_ru, meas_z, model)
        if k == 0:  # no information of birth (at the first time step), increase birth probability
            r_birth = np.ones(len(r_birth)) * 0.7
        if len(sols_birth) == 0:
            return
        if len(self.pruned_tt) == 0:
            for idx, rbb in enumerate(r_birth):
                feat = np.zeros((model.N_sensors, self.feat_dim))
                for idx_s in range(model.N_sensors):
                    m_idx = sols_birth[idx][idx_s]
                    if m_idx:
                        feat[idx_s, :] = meas[idx_s][5:, m_idx - 1]
                tt = Target(m_birth[idx], P_birth[idx], rbb, self.id, feat, model, k, self.use_feat)
                self.id += 1
                self.tt_birth.append(tt)
            return

        # (3) Perform re-appearing tracks and adding new birth targets
        track_features = np.asarray(self.pruned_tt_feat)
        allcost = np.ones((len(self.pruned_tt), len(sols_birth)))
        for idxsol, sol in enumerate(sols_birth):
            cdist_mat = np.ones((len(self.pruned_tt), model.N_sensors))
            for s in range(model.N_sensors):
                m_idx = sol[s]
                if m_idx:
                    cdist_mat[:, s] = cdist(track_features[:, s, :], norm_feat01(meas[s][5:, m_idx - 1:m_idx]).T)[:, 0]
            cost_s = np.min(cdist_mat, axis=1)
            allcost[:, idxsol] = cost_s
        cost_len = (k - np.array([tt.birth_time for tt in self.pruned_tt], dtype='float'))
        cost_len /= np.array([len(tt.ah) for tt in self.pruned_tt], dtype='float')  # survival_len / association_len
        allcost = cost_len[:, None] * allcost
        pruned_tt_label = np.array(self.pruned_tt_label)
        u_label = np.unique(self.pruned_tt_label)
        u_mapping = np.zeros((len(u_label), len(sols_birth)), dtype=int)
        u_allcost = np.zeros((len(u_label), len(sols_birth)))
        for idx, label in enumerate(u_label):
            label_cost = np.copy(allcost)
            label_cost[pruned_tt_label != label] = np.inf
            for jdx in range(len(sols_birth)):
                u_allcost[idx, jdx] = np.amin(label_cost[:, jdx])
                u_mapping[idx, jdx] = np.argmin(label_cost[:, jdx])
        # LapJV (minimization), take negative logarithm to find a solution that maximizing likelihood
        assignments = lap.lapjv(u_allcost, extend_cost=True, cost_limit=0.3, return_cost=False)
        uc_sum = np.sum(u_allcost)
        l_reappear = []
        for idx, sol in enumerate(sols_birth):
            feat = np.zeros((model.N_sensors, self.feat_dim))
            for idx_s in range(model.N_sensors):
                m_idx = sol[idx_s]
                if m_idx:
                    feat[idx_s, :] = meas[idx_s][5:, m_idx - 1]
            # m_temp = np.tile(m_birth[idx], (len(model.mode), 1)).T
            if assignments[1][idx] >= 0:
                tt_idx = u_mapping[assignments[1][idx], idx]
                target = self.pruned_tt[tt_idx]
                r_b = min(model.r_birth[0], (uc_sum - u_allcost[assignments[1][idx], idx]) / uc_sum)
                target.re_activate(m_birth[idx], P_birth[idx], r_b, feat, model, k)
                l_reappear.append(target.l)
                self.tt_birth.append(target)
            else:  # re-id feature distance (a pruned_tt & a new birth) is not close => initiate new birth
                tt = Target(m_birth[idx], P_birth[idx], r_birth[idx], self.id, feat, model, k, self.use_feat)
                if self.adaptive_birth == 2:
                    tt.P = P_birth[idx]
                self.id += 1
                self.tt_birth.append(tt)
        in1d = np.nonzero(np.in1d(self.pruned_tt_label, l_reappear))[0]
        for same_idx in range(len(in1d) - 1, -1, -1):  # remove backward
            del self.pruned_tt[in1d[same_idx]]
            del self.pruned_tt_feat[in1d[same_idx]]
            del self.pruned_tt_label[in1d[same_idx]]

    def clean_predict(self):
        # hash label sets, find unique ones, merge all duplicates
        glmb_raw_hash = np.zeros(len(self.glmb_update_w), dtype=np.dtype('<U2048'))
        for hidx in range(0, len(self.glmb_update_w)):
            hash_str = np.array2string(np.sort(self.glmb_update_I[hidx]), separator='*')[1:-1]
            glmb_raw_hash[hidx] = hash_str

        cu, _, ic = np.unique(glmb_raw_hash, return_index=True, return_inverse=True, axis=0)

        glmb_temp_w = np.NINF * np.ones((len(cu)))
        glmb_temp_I = [np.array([]) for i in range(0, len(ic))]
        glmb_temp_n = np.zeros((len(cu)), dtype=int)
        for hidx in range(0, len(ic)):
            glmb_temp_w[ic[hidx]] = logsumexp([glmb_temp_w[ic[hidx]], self.glmb_update_w[hidx]])
            glmb_temp_I[ic[hidx]] = self.glmb_update_I[hidx]
            glmb_temp_n[ic[hidx]] = self.glmb_update_n[hidx]

        self.glmb_update_w = glmb_temp_w  # 2
        self.glmb_update_I = glmb_temp_I  # 3
        self.glmb_update_n = glmb_temp_n  # 4

    def clean_update(self, time_step):
        # flag used tracks
        usedindicator = np.zeros(len(self.glmb_update_tt), dtype=int)
        for hidx in range(0, len(self.glmb_update_w)):
            usedindicator[self.glmb_update_I[hidx]] = usedindicator[self.glmb_update_I[hidx]] + 1
        trackcount = sum(usedindicator > 0)

        # remove unused tracks and reindex existing hypotheses/components
        newindices = np.zeros(len(self.glmb_update_tt), dtype=int)
        newindices[usedindicator > 0] = np.arange(0, trackcount)
        glmb_clean_tt = [self.glmb_update_tt[i] for i, indicator in enumerate(usedindicator) if indicator > 0]
        self.glmb_update_tt = glmb_clean_tt  # 1

        glmb_clean_I = []
        for hidx in range(0, len(self.glmb_update_w)):
            glmb_clean_I.append(newindices[self.glmb_update_I[hidx]])
        self.glmb_update_I = glmb_clean_I  # 3

        if not self.use_feat:
            return
        # remove pruned targets that are kept for 50 frames
        remove_indices = [idx for idx, tt in enumerate(self.pruned_tt) if time_step - tt.last_active > 50]
        remove_indices = sorted(remove_indices, reverse=True)
        for remove_index in remove_indices:
            del self.pruned_tt[remove_index]
            del self.pruned_tt_feat[remove_index]
            del self.pruned_tt_label[remove_index]
        # find pruned targets
        curr_tt_labels = np.unique([t.l for t in glmb_clean_tt])
        pruned_labels = np.setdiff1d(self.prev_tt_glmb_labels, curr_tt_labels)
        if len(pruned_labels):
            for i, tt in enumerate(self.prev_glmb_update_tt):
                if tt.l in pruned_labels and np.sum(tt.feat_flag) == 0 and len(tt.ah) > 3:
                    self.pruned_tt.append(tt)
                    self.pruned_tt_feat.append(tt.feat)
                    self.pruned_tt_label.append(tt.l)
        self.prev_tt_glmb_labels = curr_tt_labels
        self.prev_glmb_update_tt = glmb_clean_tt

    def prune(self, filter):
        # prune components with weights lower than specified threshold
        idxkeep = np.nonzero(self.glmb_update_w > filter.hyp_threshold)[0]
        glmb_out_w = self.glmb_update_w[idxkeep]
        glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
        glmb_out_n = self.glmb_update_n[idxkeep]

        glmb_out_w = glmb_out_w - logsumexp(glmb_out_w)
        glmb_out_cdn = np.NINF * np.ones((max(glmb_out_n) + 1))
        for card in range(0, np.max(glmb_out_n) + 1):
            idx = glmb_out_n == card
            if sum(idx) > 0:
                glmb_out_cdn[card] = logsumexp(glmb_out_w[idx])
        glmb_out_cdn = np.exp(glmb_out_cdn)

        self.glmb_update_w = glmb_out_w  # 2
        self.glmb_update_I = glmb_out_I  # 3
        self.glmb_update_n = glmb_out_n  # 4
        self.glmb_update_cdn = glmb_out_cdn  # 5

    def cap(self, filter):
        # cap total number of components to specified maximum
        if len(self.glmb_update_w) > filter.H_max:
            idxsort = np.argsort(-self.glmb_update_w)
            idxkeep = idxsort[0:filter.H_max]
            glmb_out_w = self.glmb_update_w[idxkeep]
            glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
            glmb_out_n = self.glmb_update_n[idxkeep]

            glmb_out_w = glmb_out_w - logsumexp(glmb_out_w)
            glmb_out_cdn = np.NINF * np.ones(max(glmb_out_n) + 1)
            for card in range(0, max(glmb_out_n) + 1):
                idx = glmb_out_n == card
                if sum(idx) > 0:
                    glmb_out_cdn[card] = logsumexp(glmb_out_w[idx])
            glmb_out_cdn = np.exp(glmb_out_cdn)

            self.glmb_update_w = glmb_out_w  # 2
            self.glmb_update_I = glmb_out_I  # 3
            self.glmb_update_n = glmb_out_n  # 4
            self.glmb_update_cdn = glmb_out_cdn  # 5

    def extract_estimates(self, model):
        # extract estimates via best cardinality, then
        # best component/hypothesis given best cardinality, then
        # best means of tracks given best component/hypothesis and cardinality

        # extract MAP cardinality and corresponding highest weighted component
        N = np.argmax(self.glmb_update_cdn)
        X = np.zeros((model.x_dim, N))
        L = np.zeros((2, N), dtype=int)
        S = np.zeros(N, dtype=np.dtype('int'))  # mode_type ["Upright", "Fallen"]

        idxcmp = np.argmax(-1 * (self.glmb_update_w * (self.glmb_update_n == N))).astype(int)
        for n in range(0, N):
            tt = self.glmb_update_tt[self.glmb_update_I[idxcmp][n]]
            ind = np.argmax(tt.mode)
            start = sum(tt.gm_len[0:ind])
            end = start + tt.gm_len[ind]
            indx = np.argmax(tt.w[start:end])
            X[:, n] = tt.m[:, start + indx]
            L[:, n] = [tt.birth_time, tt.l]  # use to debug (visualization)
            S[n] = ind  # model.mode_type[ind]

        return X, N, L, S

    def display_diaginfo(self, k, est, filter, H_predict, H_posterior, H_prune, H_cap):
        if filter.run_flag != 'silence':
            print(' time= ', str(k),
                  ' #eap cdn=', "{:10.4f}".format(np.arange(0, (len(self.glmb_update_cdn))) @ self.glmb_update_cdn),
                  ' #var cdn=',
                  "{:10.4f}".format(np.arange(0, (len(self.glmb_update_cdn))) ** 2 @ self.glmb_update_cdn -
                                    (np.arange(0, (len(self.glmb_update_cdn))) @ self.glmb_update_cdn) ** 2),
                  ' #est card=', "{:10.4f}".format(est[1]),
                  ' #comp pred=', "{:10.4f}".format(H_predict),
                  ' #comp post=', "{:10.4f}".format(H_posterior),
                  ' #comp updt=', "{:10.4f}".format(H_cap),
                  ' #trax updt=', "{:10.4f}".format(len(self.glmb_update_tt)))

    def save_est_motformat(self, root, dataset, save_modes=False):
        # save_format = '{frame},{id}, -1, -1, -1, -1, -1,{x},{y},{z},{wx},{wy},{h}, [{modes}]\n'
        est_list = []
        for frame_id in range(len(self.est.X)):
            targets = self.est.X[frame_id]
            for eidx, tt in enumerate(targets.T):
                x, y, z = tt[[0, 2, 4]]
                wx, wy, h = tt[[6, 7, 8]]
                tt_id = self.est.L[frame_id][:, eidx]
                tt_id = tt_id[1]
                modes = self.est.S[frame_id][eidx]
                if save_modes:
                    line = [frame_id + 1, tt_id, -1, -1, -1, -1, -1, x, y, z, wx, wy, h, modes]
                else:
                    line = [frame_id + 1, tt_id, -1, -1, -1, -1, -1, x, y, z, wx, wy, h]
                est_list.append(line)
        est = np.array(est_list)
        est[:, [10, 11, 12]] = np.exp(est[:, [10, 11, 12]])
        np.savetxt(os.path.join(root, 'EST_{}_WORLD_CENTROID.txt'.format(dataset)), est)
        return est

    def run(self, model, dataset, meas):
        filter_params = filter(model)
        st = time.time()  # get the start time
        # measurement in each sensor [0,1,2,3 (bbox: L T W H), 4 (conf), 5:end (re-id feature)]
        # start_off = int(len(meas) / 2)  # turn off all cameras for 30f
        for k, bboxes in enumerate(meas):
            # if start_off < k < start_off + 30:
            #     for s in range(model.N_sensors):
            #         bboxes[s] = np.empty((bboxes[s].shape[0], 0))
            # joint prediction and update
            self.msjointpredictupdate(model, filter_params, bboxes, k)
            H_posterior = len(self.glmb_update_w)

            # pruning and truncation
            self.prune(filter_params)
            H_prune = len(self.glmb_update_w)
            self.cap(filter_params)
            H_cap = len(self.glmb_update_w)
            self.clean_update(k)

            # state estimation and display diagnostics
            est = self.extract_estimates(model)
            self.est.X[k], self.est.N[k], self.est.L[k], self.est.S[k] = est
            self.display_diaginfo(k, est, filter_params, H_posterior, H_posterior, H_prune, H_cap)
            # est = self.MSGLMB.run_msglmb_ukf(bboxes, k)
            # self.est.X[k], self.est.N[k], self.est.L[k] = est
        et = time.time()  # get the end time
        print('Total Execution Time:', str(et - st), 'seconds', ', FPS', str(len(meas) / (et - st)))
        return self.save_est_motformat("./results", dataset, save_modes=True)

    def runcpp(self, model, dataset, meas, adaptive_birth=1, use_feat=True):
        msglmbcpp = MSGLMB(model.cam_mat, dataset, adaptive_birth, use_feat)
        st = time.time()  # get the start time
        # measurement in each sensor [0,1,2,3 (bbox: L T W H), 4 (conf), 5:end (re-id feature)]
        # start_off = int(len(meas) / 2)  # turn off all cameras for 30f
        for k, bboxes in enumerate(meas):
            # if start_off < k < start_off + 30:
            #     for s in range(model.N_sensors):
            #         bboxes[s] = np.empty((bboxes[s].shape[0], 0))
            est = msglmbcpp.run_msglmb_ukf(bboxes, k)
            self.est.X[k], self.est.N[k], self.est.L[k], self.est.S[k] = est
        et = time.time()  # get the end time
        print('Total Execution Time:', str(et - st), 'seconds', ', FPS', str(len(meas) / (et - st)))
        return self.save_est_motformat("./results", dataset, save_modes=True)
    # END
