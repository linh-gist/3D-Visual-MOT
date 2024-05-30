//
// Created by linh on 2022-03-31.
//

#ifndef UKF_TARGET_TARGET_HPP
#define UKF_TARGET_TARGET_HPP

#include "Model.hpp"
#include <numeric>
#include <algorithm>


#ifndef M_PI
#define M_PI 3.14159265358979323846 // numpy pi: 3.141592653589793
#endif

using namespace std;
using namespace Eigen;

tuple<MatrixXd, VectorXd> ut(VectorXd m, MatrixXd P, double alpha, double kappa) {
    int n_x = m.size();
    double lambda_ = alpha * alpha * (n_x + kappa) - n_x;
    MatrixXd Psqrtm = ((n_x + lambda_) * P).llt().matrixU().transpose(); // upper-triangular Cholesky
    MatrixXd temp = MatrixXd::Zero(n_x, 2 * n_x + 1);
    temp(all, seq(1, n_x)) = -Psqrtm;
    temp(all, seq(n_x + 1, 2 * n_x)) = Psqrtm;
    MatrixXd X = temp.colwise() + m;
    VectorXd W = VectorXd::Ones(2 * n_x + 1);
    W = 0.5 * W;
    W(0) = lambda_;
    W = W / (n_x + lambda_);
    return {X, W};
}

MatrixXd gen_msobservation_fn(MatrixXd cam_mat, MatrixXd X, MatrixXd W, Vector2d imagesize) {
    if (X.rows() == 0) {
        return MatrixXd(0, 0);
    }
    X(seq(6, 8), all) = X(seq(6, 8), all).array().exp();  // Carefully check reference or pointer
    MatrixXd bbs_noiseless(4, X.cols());
    for (int i = 0; i < X.cols(); i++) {
        VectorXd ellipsoid_c(3);
        ellipsoid_c << X(0, i), X(2, i), X(4, i); // xc, yc, zc
        // Quadric general equation 0 = Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J
        // Q = [A D/2 E/2 G/2;
        //      D/2 B F/2 H/2;
        //      E/2 F/2 C I/2;
        //      G/2 H/2 I/2 J];
        // calculations for A, B, C, rx, ry, hh = X[6, i], X[7, i], X[8, i]  # half length radius (x, y, z)
        double A = 1 / (X(6, i) * X(6, i));
        double B = 1 / (X(7, i) * X(7, i));
        double C = 1 / (X(8, i) * X(8, i));
        // calculations for D, E, F, no rotation (axis-aligned) means D, E, F = 0
        double D = 0;
        double E = 0;
        double F = 0;
        // calculations for G, H, I, J
        MatrixXd PSD(3, 3); // np.diag([A, B, C])
        PSD.setZero();
        PSD(0, 0) = A;
        PSD(1, 1) = B;
        PSD(2, 2) = C;
        EigenSolver<MatrixXd> es(PSD); // This constructor calls compute() to compute the values and vectors.
        MatrixXd eig_vals = es.eigenvalues().real();
        MatrixXd right_eig = es.eigenvectors().real(); // [V,D] = eig(A), right eigenvectors, so that A*V = V*D
        VectorXd temp_ellip_c = right_eig.transpose() * ellipsoid_c;
        VectorXd ggs = -2 * temp_ellip_c.array() * eig_vals.array();
        VectorXd desired = ggs.transpose() * right_eig;
        double G = desired[0];
        double H = desired[1];
        double I = desired[2];
        double J = -1 + (ggs.array() * ggs.array() / (4 * eig_vals.array())).sum();
        MatrixXd Q(4, 4);
        Q << A, D / 2, E / 2, G / 2,
                D / 2, B, F / 2, H / 2,
                E / 2, F / 2, C, I / 2,
                G / 2, H / 2, I / 2, J; // 4x4 matrix
        MatrixXd C_t = cam_mat * Q.inverse() * cam_mat.transpose();
        MatrixXd CI = C_t.inverse(); // 3x3 matrix
        MatrixXd C_strip = CI(seq(0, 1), seq(0, 1));
        es.compute(C_strip);
        eig_vals = es.eigenvalues().real();
        right_eig = es.eigenvectors().real(); // [V,D] = eig(A), right eigenvectors, so that A*V = V*D
        VectorXd x_and_y_vec = 2 * CI(seq(0, 1), 2); // extrack D and E
        VectorXd x_and_y_vec_transformed = x_and_y_vec.transpose() * right_eig;
        VectorXd h_temp = (x_and_y_vec_transformed.array() / eig_vals.array()) / 2;
        VectorXd h_temp_squared = eig_vals.array() * (h_temp.array() * h_temp.array());
        VectorXd h = -1 * h_temp;
        VectorXd ellipse_c = right_eig * h;
        double offset = -1 * (h_temp_squared.sum()) + CI(2, 2);
        VectorXd bbs_temp(4);
        if ((-offset / eig_vals(0) > 0) && (-offset / eig_vals(1) > 0)) {
            VectorXd uu = right_eig.col(0) * sqrt(-offset / eig_vals(0));
            VectorXd vv = right_eig.col(1) * sqrt(-offset / eig_vals(1));
            VectorXd e = (uu.array() * uu.array() + vv.array() * vv.array()).sqrt();
            MatrixXd bbox(2, 2);
            bbox.col(0) = ellipse_c - e;
            bbox.col(1) = ellipse_c + e;

            double tl0 = bbox.row(0).minCoeff(); // top_left0
            double tl1 = bbox.row(1).minCoeff(); // top_left1
            double br0 = bbox.row(0).maxCoeff(); // bottm_right0
            double br1 = bbox.row(1).maxCoeff();//  bottm_right1

            bbs_temp << tl0, tl1, log((br0 - tl0)), log((br1 - tl1));
        } else {
            // top_left = [1 1];
            // bottm_right = imagesize
            bbs_temp << 1, 1, log((imagesize(0) - 1)), log((imagesize(1) - 1));
        }
        bbs_noiseless.col(i) = bbs_temp;
    }
    return bbs_noiseless + W;  // bounding measurement
}

MatrixXd gen_msobservation_fn_v2(const MatrixXd &c, MatrixXd X, MatrixXd wNoise, const VectorXd &meas_n_mu_mode) {
    if (X.rows() == 0) {
        return MatrixXd::Zero(0, 0);
    }
    X(seq(6, 8), all) = X(seq(6, 8), all).array().exp();  // Carefully check reference or pointer
    MatrixXd vet2(4, 6);
    MatrixXd bbs_noiseless(4, X.cols());
    VectorXi state_ids(6);
    state_ids << 0, 2, 4, 6, 7, 8;
    for (int i = 0; i < X.cols(); i++) {
        VectorXd temp_coli = X.col(i);
        VectorXd temp = temp_coli(state_ids);
        vet2.col(0) << temp[0] + temp[3], temp[1], temp[2], 1;// -right
        vet2.col(1) << temp[0] - temp[3], temp[1], temp[2], 1;// -left
        vet2.col(2) << temp[0], temp[1] + temp[4], temp[2], 1;// | right
        vet2.col(3) << temp[0], temp[1] - temp[4], temp[2], 1;// | left
        vet2.col(4) << temp[0], temp[1], temp[2] + temp[5], 1;
        vet2.col(5) << temp[0], temp[1], temp[2] - temp[5], 1;

        MatrixXd temp_c = c * vet2;
        MatrixXd vertices = temp_c(seq(0, 1), all).array().rowwise() / temp_c(2, all).array();
        double x_2 = vertices.row(0).maxCoeff();
        double x_1 = vertices.row(0).minCoeff();
        double y_2 = vertices.row(1).maxCoeff();
        double y_1 = vertices.row(1).minCoeff();
        bbs_noiseless.col(i) << x_1, y_1, log(x_2 - x_1) + meas_n_mu_mode[0], log(y_2 - y_1) + meas_n_mu_mode[1];
    }
    return bbs_noiseless.array() + wNoise.array();
}

tuple<double, VectorXd, MatrixXd>
ukf_update_per_sensor(VectorXd z, VectorXd m, MatrixXd P, int s, int mode, Model model,
                      double alpha = 1, double kappa = 2, double beta = 2) {
    bool ch1 = m(0) > model.XMAX(0) && m(0) < model.XMAX(1);
    bool ch2 = m(2) > model.YMAX(0) && m(2) < model.YMAX(1);
    bool ch3 = m(4) > model.ZMAX(0) && m(4) < model.ZMAX(1);
    if (!(ch1 && ch2 && ch3)) {
        return {log(std::nexttoward(0.0, 1.0L)), m, P};  // qz_temp
    }
    VectorXd mtemp = VectorXd::Zero(model.x_dim + model.z_dim);
    mtemp(seq(0, model.x_dim - 1)) = m;
    MatrixXd Ptemp = MatrixXd::Zero(model.x_dim + model.z_dim, model.x_dim + model.z_dim);
    Ptemp.topLeftCorner(model.x_dim, model.x_dim) = P;
    Ptemp.bottomRightCorner(model.z_dim, model.z_dim) = model.R[mode];
    MatrixXd X_ukf;
    VectorXd u;
    std::tie(X_ukf, u) = ut(mtemp, Ptemp, alpha, kappa);
    int start = model.x_dim;
    int end = model.x_dim + model.z_dim - 1;
    MatrixXd x_state = X_ukf(seq(0, start - 1), all);
    MatrixXd noise = X_ukf(seq(start, end), all);
    MatrixXd Z_pred = gen_msobservation_fn_v2(model.camera_mat[s], x_state, noise, model.meas_n_mu.col(0));
    VectorXd eta = Z_pred * u;
    MatrixXd S_temp = Z_pred.colwise() - eta;
    u(0) = u(0) + (1 - alpha * alpha + beta);
    MatrixXd S = S_temp * u.asDiagonal() * S_temp.transpose();
    MatrixXd Vs = S.llt().matrixU();
    double det_S = Vs.diagonal().prod();
    det_S = det_S * det_S;
    MatrixXd inv_sqrt_S = Vs.inverse();
    MatrixXd iS = inv_sqrt_S * inv_sqrt_S.transpose();
    MatrixXd G_temp = X_ukf(seq(0, model.x_dim - 1), all).colwise() - m;
    MatrixXd G = G_temp * u.asDiagonal() * S_temp.transpose();
    MatrixXd K = G * iS;
    VectorXd z_eta = z - eta;
    double qz_temp = -0.5 * (z.size() * log(2 * M_PI) + log(det_S) + z_eta.transpose() * iS * z_eta);
    VectorXd m_temp = m + K * z_eta;
    MatrixXd P_temp = P - G * iS * G.transpose();

    return {qz_temp, m_temp, P_temp};
}

tuple<VectorXd, MatrixXd, vector<MatrixXd>> cleanup(VectorXd w, MatrixXd m, vector<MatrixXd> P,
                                                    double elim_th = 1e-5, double merge_th = 4, int l_max = 10) {
    if (w.size() <= 1) {
        return {w, m, P};
    }
    VectorXd w_temp(w.size());
    MatrixXd m_temp(m.rows(), m.cols());
    vector<MatrixXd> P_temp(P.size());
    // Gaussian prune, remove components that have weight lower than a threshold
    w = w.array().exp();
    int sIndex = 0;
    for (int i = 0; i < w.size(); i++) {
        if (w(i) > elim_th) {
            w_temp(sIndex) = w(i);
            m_temp.col(sIndex) = m.col(i);
            P_temp[sIndex] = P[i];
            sIndex++;
        }
    }
    w = w_temp(seq(0, sIndex - 1));
    m = m_temp(all, seq(0, sIndex - 1));
    P_temp.resize(sIndex);
    P = vector<MatrixXd>(P_temp); // constructor method, Deep copy

    // Merging Gaussian mixture components using Mahalanobis distance
    vector<int> I(w.size());
    std::iota(I.begin(), I.end(), 0);
    sIndex = 0;
    while (I.size()) {
        int j;
        w.maxCoeff(&j);
        vector<int> Ij;
        MatrixXd iPt = P[j].inverse();
        for (int i: I) {
            VectorXd xi_xj = m.col(i) - m.col(j);
            double val = (xi_xj.transpose() * iPt) * xi_xj;
            if (val <= merge_th) {
                Ij.push_back(i);
            }
        }
        VectorXi merge_idxs = VectorXi::Map(Ij.data(), Ij.size());
        double w_new_t = w(merge_idxs).sum();
        MatrixXd mIj = m(all, merge_idxs).array();
        VectorXd wIj = w(merge_idxs);
        MatrixXd x_newTmp = mIj.array().rowwise() * wIj.transpose().array();
        VectorXd x_new_t = x_newTmp.rowwise().sum();
        MatrixXd P_new_t = MatrixXd::Zero(P[j].rows(), P[j].cols());
        for (int ii: Ij) {
            P_new_t = P_new_t.array() + (P[ii] * w(ii)).array();
        }
        x_new_t = x_new_t / w_new_t;
        P_new_t = P_new_t / w_new_t;
        w_temp[sIndex] = w_new_t;
        m_temp.col(sIndex) = x_new_t;
        P_temp[sIndex] = P_new_t;

        vector<int> iITemp;
        std::set_difference(I.begin(), I.end(), Ij.begin(), Ij.end(), std::inserter(iITemp, iITemp.begin()));
        I = iITemp;
        w(Ij).array() = -1;
        sIndex += 1;
    }
    w = w_temp(seq(0, sIndex - 1));
    m = m_temp(all, seq(0, sIndex - 1));
    P_temp.resize(sIndex);
    P = vector<MatrixXd>(P_temp);

    // Gaussian cap, limit on number of Gaussians in each track
    if (w.size() > l_max) {
        vector<int> idx(w.size());
        iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(),
                    [&w](int i1, int i2) { return w[i1] > w[i2]; });
        sIndex = 0;
        for (int i: idx) {
            w_temp(sIndex) = w(i);
            m_temp.col(sIndex) = m.col(i);
            P_temp[sIndex] = P[i];
            sIndex++;
            if (sIndex >= l_max) {
                break;
            }
        }
        VectorXd w_new = w_temp(seq(0, l_max - 1));
        w = w_new * (w.sum() / w_new.sum());
        m = m_temp(all, seq(0, l_max - 1));
        P_temp.resize(sIndex);
        P = vector<MatrixXd>(P_temp);
    }
    return {w.array().log(), m, P};
}

VectorXd norm_feat01(VectorXd x) {
    double min_x = x.minCoeff();
    x = (x.array() - min_x) / (x.maxCoeff() - min_x);
    return x / x.norm();
}

MatrixXd homtrans(const MatrixXd &T, const MatrixXd &z) {
    // https://www.petercorke.com/RTB/r9/html/homtrans.html
    MatrixXd e2h = MatrixXd::Ones(3, z.cols());
    e2h(seq(0, 1), seq(0, z.cols() - 1)) = z; // E2H Euclidean to homogeneous
    MatrixXd temp = T * e2h; // H2E Homogeneous to Euclidean
    MatrixXd pt = temp(seq(0, 1), all).array().rowwise() / temp.row(2).array();
    return pt;
}


/* track table for GLMB (cell array of structs for individual tracks)
  (1) mR: existence probability
  (2) Gaussian Mixture mW (weight), mM (mean), mP (covariance matrix)
  (3) Label: birth time & index of target at birth time step
  (4) mGateMeas: indexes gating measurement (using  Chi-squared distribution)
  (5) mMode: [0.6, 0.4] probability of being ["Upright", "Fallen"]
  (6) mFeat: array of re-identification feature of Target in all sensors
 */
class Target {
private:

    Model mModel;
    VectorXd mWg;
    MatrixXd mMg;
    vector<MatrixXd> mPg;
    bool mUseFeat;
    double mAlphaFeat;
    double mReAlphaFeat;
    vector<MatrixXd> mP;
    double mPS;
    double mPD;

    double log_sum_exp(VectorXd arr) {
        int count = arr.size();
        if (count > 0) {
            double maxVal = arr.maxCoeff();
            double sum = 0;
            for (int i = 0; i < count; i++) {
                sum += exp(arr(i) - maxVal);
            }
            return log(sum) + maxVal;
        } else {
            return 0.0;
        }
    }

public:
    MatrixXd mM;
    VectorXd mW;
    VectorXd mMode;
    vector<VectorXi> mGateMeas;
    double mR;
    int mL;
    int mLastActive;
    int mBirthTime;
    VectorXi mGmLen;
    MatrixXd mFeat;
    VectorXi mFeatFlag;
    std::vector<VectorXi> mAh; // association history

    Target() {

    }

    void setInitCovariance(MatrixXd pP) {
        for (int i = 0; i < mP.size(); i++) {
            mP[i] = pP;
        }
    }

    Target(MatrixXd m, vector<MatrixXd> Pb, double prob_birth, int label, MatrixXd feat, Model model, int birth_time,
           bool use_feat = true) {
        mM = m;
        mP = Pb;
        mW = model.w_birth;
        mR = prob_birth;
        mPS = model.P_S;
        mPD = model.P_D;
        mL = label;
        mModel = model;
        mGateMeas.resize(mModel.N_sensors);
        mMode = model.mode;

        // wg, mg, Pg, ..., store temporary Gaussian mixtures while predicting and updating
        int max_cpmt = 100;
        int x_dim = 9;
        mWg.resize(max_cpmt);
        mMg.resize(x_dim, max_cpmt);
        mPg.resize(max_cpmt);
        mGmLen = VectorXi::Ones(model.mode.size());  // number of Gaussian mixtures (block of GM for each mode)

        mUseFeat = use_feat;
        if (use_feat) {
            mAlphaFeat = 0.9;
            mReAlphaFeat = 0.2;
            mFeatFlag = (feat.rowwise().sum().array() == 0).cast<int>();
            VectorXd averageFeat = norm_feat01(feat.colwise().sum() / (1 - mFeatFlag.array()).sum());
            mFeat.resize(model.N_sensors, averageFeat.size());
            for (int s = 0; s < model.N_sensors; s++) {
                if (mFeatFlag[s]) {
                    mFeat.row(s) = averageFeat;
                } else {
                    mFeat.row(s) = norm_feat01(feat.row(s));
                }
            }
        }
        mLastActive = birth_time;
        mBirthTime = birth_time;
    }

    void copy(const Target tt) { // deep copy
        mW = tt.mW;
        mMode = tt.mMode;
        mGateMeas = tt.mGateMeas;
        mLastActive = tt.mLastActive;
        mGmLen = tt.mGmLen;
        mUseFeat = tt.mUseFeat;
        mAlphaFeat = tt.mAlphaFeat;
        mFeatFlag = tt.mFeatFlag;
        mP = tt.mP;
        mPS = tt.mPS;
        mPD = tt.mPD;
        mAh = tt.mAh;
    }

    void predict() {
        // this is to offset the normal mean because of lognormal multiplicative noise.
        VectorXd offset = VectorXd::Zero(mModel.x_dim);
        int mode_len = mMode.size();
        VectorXd mode_predict = VectorXd::Zero(mode_len);
        int gm_tmp_idx = 0;
        for (int n_mode = 0; n_mode < mode_len; n_mode++) {
            VectorXd mode_predict_temp = VectorXd::Zero(mode_len);
            int gm_s = 0;  // Start of GM index
            int gm_e = 0;  // End of GM index
            int idx_noise = 0;
            for (int c_mode = 0; c_mode < mode_len; c_mode++) {
                if (c_mode == n_mode) {  // transition to same mode
                    if (c_mode == 0) {  // transition from Upright to Upright
                        offset(seq(6, 8)) << mModel.n_mu(0, 0), mModel.n_mu(0, 0), mModel.n_mu(1, 0);
                        idx_noise = 0;
                    }
                    if (c_mode == 1) {  // transition from Fallen to Fallen
                        offset(seq(6, 8)) << mModel.n_mu(0, 1), mModel.n_mu(0, 1), mModel.n_mu(1, 1);
                        idx_noise = 1;
                    }
                } else {  // transition to different mode
                    offset(seq(6, 8)) << mModel.n_mu(0, 2), mModel.n_mu(0, 2), mModel.n_mu(0, 2);
                    idx_noise = 2;
                }
                gm_e += mGmLen[c_mode];
                MatrixXd m_per_mode = mM.colwise() + offset;
                for (int l = gm_s; l < gm_e; l++) {
                    mWg[gm_tmp_idx] = mW[l];
                    // kalman filter prediction for a single component
                    mMg.col(gm_tmp_idx) = mModel.F * m_per_mode.col(l);
                    mPg[gm_tmp_idx] = mModel.Q[idx_noise] + ((mModel.F * mP[l]) * mModel.F.transpose());
                    gm_tmp_idx += 1;
                }
                gm_s = gm_e;
                mode_predict_temp[c_mode] = mModel.mode_trans_matrix(c_mode, n_mode) + mMode[c_mode];
            }
            int start_norm = gm_tmp_idx - mGmLen.sum();
            mWg(seq(start_norm, gm_tmp_idx - 1)).array() -= log_sum_exp(mWg(seq(start_norm, gm_tmp_idx - 1)));
            mode_predict[n_mode] = log_sum_exp(mode_predict_temp);
        }
        mMode = mode_predict;
        mGmLen = VectorXi::Ones(mMode.size()) * mGmLen.sum();
        mW = mWg(seq(0, gm_tmp_idx - 1));
        mM = mMg(all, seq(0, gm_tmp_idx - 1));
        mP.resize(gm_tmp_idx);
        for (int i = 0; i < gm_tmp_idx; i++) {
            mP[i] = mPg[i];
        }
    }

    double ukf_likelihood_per_sensor(VectorXd z, int s) {
        VectorXd z_bbox = z(seq(0, 3));
        VectorXd z_feat = norm_feat01(z(seq(5, z.size() - 1)));
        VectorXd for_cost = VectorXd::Zero(mMode.size());
        int gm_s = 0;
        int gm_e = 0;
        double prob = 0;
        for (int mode = 0; mode < mMode.size(); mode++) {
            double ratio = exp(z_bbox[3] - z_bbox[2]);
            if (mode == 0) {
                prob = 1 * (ratio - 1);
            }
            if (mode == 1) {
                prob = -1 * (ratio - 1);
            }
            double q_mode = prob + mMode[mode];
            VectorXd q_z = VectorXd::Zero(mGmLen[mode]);
            gm_e += mGmLen[mode];
            for (int idxp = gm_s; idxp < gm_e; idxp++) {
                VectorXd mIdx = mM.col(idxp);
                MatrixXd PIdx = mP[idxp];
                double qzTmp;
                VectorXd mTmp;
                MatrixXd pTmp;
                tie(qzTmp, mTmp, pTmp) = ukf_update_per_sensor(z_bbox, mIdx, PIdx, s, mode, mModel);
                q_z[idxp - gm_s] = qzTmp;
            }
            VectorXd w_temp = q_z + mW(seq(gm_s, gm_e - 1));
            for_cost[mode] = q_mode + log_sum_exp(w_temp);
            gm_s = gm_e;
        }
        double pm_temp = 0;
        if (mUseFeat) {
            int dim = mFeat.row(s).size();
            ArrayXd x_z = mFeat.row(s).transpose() - z_feat;
            double xz_square = x_z.square().sum();
            double xsum_temp = mFeat.row(s).array().square().sum() - mFeat.row(s).sum();
            double no_change = log(4 * dim - xz_square) - log(dim * 11.0 / 3 - xsum_temp);
            double change = log(xz_square) - (log(dim * 1.0 / 3 + xsum_temp));
            std::vector<double> a = {log(0.9) + no_change, log(0.1) + change};
            pm_temp = log_sum_exp(VectorXd::Map(a.data(), a.size()));
        }
        return log_sum_exp(for_cost) + pm_temp;
    }

    tuple<double, Target> ukf_update(vector<MatrixXd> z, VectorXi nestmeasidxs, int time_step) {
        VectorXd for_cost = VectorXd::Zero(mMode.size());
        Target tt(mM, mP, mR, mL, mFeat, mModel, mBirthTime, mUseFeat); // Deep copy
        tt.copy(*this);
        VectorXi ah(mModel.N_sensors + 1); // 0 frame index, 1->s measurement index from sensors
        ah[0] = time_step;
        ah(seq(1, mModel.N_sensors)) = nestmeasidxs;
        tt.mAh.push_back(ah);
        nestmeasidxs = nestmeasidxs.array() - 1;  // restore original measurement index 0-|Z|
        int gm_s = 0;
        int gm_e = 0;
        for (int mode = 0; mode < tt.mMode.size(); mode++) {
            VectorXd q_mode = VectorXd::Zero(mModel.N_sensors);
            MatrixXd q_z = MatrixXd::Zero(mModel.N_sensors, tt.mGmLen[mode]);
            gm_e += mGmLen[mode];
            double prob = 1;
            for (int q = 0; q < nestmeasidxs.size(); q++) {
                int idx = nestmeasidxs[q];
                if (idx < 0) {
                    continue;
                }
                MatrixXd z_bbox = z[q](seq(0, 3), all);
                double ratio = exp(z_bbox(3, idx) - z_bbox(2, idx));
                if (mode == 0) {
                    prob = 1 * (ratio - 1);
                }
                if (mode == 1) {
                    prob = -1 * (ratio - 1);
                }
                q_mode[q] = prob + tt.mMode[mode];
                for (int idxp = gm_s; idxp < gm_e; idxp++) {
                    VectorXd mIdx = tt.mM.col(idxp);
                    MatrixXd PIdx = tt.mP[idxp];
                    double qt;
                    VectorXd mt;
                    MatrixXd Pt;
                    tie(qt, mt, Pt) = ukf_update_per_sensor(z_bbox.col(idx), mIdx, PIdx, q, mode, mModel);
                    q_z(q, idxp - gm_s) = qt;
                    tt.mM.col(idxp) = mt;
                    tt.mP[idxp] = Pt;
                }
            }
            VectorXd w_temp = ((VectorXd) q_z.colwise().sum()) + tt.mW(seq(gm_s, gm_e - 1));
            tt.mW(seq(gm_s, gm_e - 1)) = q_mode.sum() + w_temp.array();
            for_cost[mode] = q_mode.sum() + log_sum_exp(w_temp);
            gm_s = gm_e;
        }
        gm_s = 0;
        gm_e = 0;
        for (int mode = 0; mode < tt.mMode.size(); mode++) {
            gm_e += mGmLen[mode];
            tt.mW(seq(gm_s, gm_e - 1)) = tt.mW(seq(gm_s, gm_e - 1)).array() - log_sum_exp(for_cost);
            tt.mMode[mode] = log_sum_exp(tt.mW(seq(gm_s, gm_e - 1)));
            tt.mW(seq(gm_s, gm_e - 1)) = tt.mW(seq(gm_s, gm_e - 1)).array() - tt.mMode[mode];
            gm_s = gm_e;
        }
        // Update re-identification feature information
        tt.mLastActive = time_step;
        double pm_temp = 0;
        for (int s = 0; s < nestmeasidxs.size(); s++) {
            int idx = nestmeasidxs[s];
            if (idx < 0 || !mUseFeat) {
                continue;
            }
            VectorXd z_feat = norm_feat01(z[s](seq(5, z[s].rows() - 1), idx));
            int dim = mFeat.row(s).size();
            ArrayXd x_z = mFeat.row(s).transpose() - z_feat;
            double xz_square = x_z.square().sum();
            double xsum_temp = mFeat.row(s).array().square().sum() - mFeat.row(s).sum();
            // NO CHANGE f(z|x) = sum_i (4 - (x_i - z_i)^2)  => g(z | x, varrho=0) = f(z|x) / \int (f(z|x)dz)
            double no_change = log(4 * dim - xz_square) - (log(11.0 * dim / 3 - xsum_temp));
            // CHANGE f(z|x) = sum_i ((x_i - z_i)^2)
            double change = log(xz_square) - (log(dim * 1.0 / 3 + xsum_temp));
            // consider pm in prediction is [0.1, 0.9]
            std::vector<double> a = {log(0.9) + no_change, log(0.1) + change};
            pm_temp += log_sum_exp(VectorXd::Map(a.data(), a.size()));
            if (mFeatFlag[s]) {
                tt.mFeat.row(s) = z_feat;
                tt.mFeatFlag[s] = false;
            } else {
                VectorXd feat_temp = tt.mAlphaFeat * tt.mFeat.row(s);
                feat_temp = feat_temp + (1 - tt.mAlphaFeat) * z_feat;
                tt.mFeat.row(s) = norm_feat01(feat_temp);
            }
        }
        return {log_sum_exp(for_cost) + pm_temp, tt};
    }

    void gating_feet(vector<MatrixXd> Zz, double gating_threshold = 0.8) {  // Euclid distance
        for (int s = 0; s < Zz.size(); s++) {
            MatrixXd bboxes = Zz[s](seq(0, 3), all);
            int zlength = bboxes.cols();
            mGateMeas[s] = -1 * VectorXi::Ones(zlength);
            if (zlength == 0) {
                continue;
            }
            MatrixXd feet_loc = bboxes(seq(0, 1), all);
            feet_loc.row(0) = bboxes.row(0).array() + bboxes.row(2).array().exp() / 2;
            feet_loc.row(1) = bboxes.row(1).array() + bboxes.row(3).array().exp();
            VectorXi cIndxs(3);
            cIndxs << 0, 1, 3;
            MatrixXd feet_loc_gp = homtrans(mModel.camera_mat[s](all, cIndxs).inverse(), feet_loc);

            int gm_s = 0;
            int gm_e = 0;
            for (int mode = 0; mode < mMode.size(); mode++) {
                gm_e += mGmLen[mode];
                for (int idxp = gm_s; idxp < gm_e; idxp++) {
                    VectorXi xyidx(2);
                    xyidx << 0, 2;
                    VectorXd m_feet = mM(xyidx, idxp);
                    MatrixXd vec_subtract = feet_loc_gp.colwise() - m_feet;
                    VectorXd euclid_dis = vec_subtract.colwise().norm();
                    for (int midx = 0; midx < euclid_dis.size(); midx++) {
                        if (euclid_dis(midx) < gating_threshold) {
                            mGateMeas[s][midx] = midx;
                        }
                    }
                }
            }
        }
    }

    void not_gating(vector<MatrixXd> Zz) {
        mGateMeas.resize(mModel.N_sensors);
        for (int s = 0; s < mModel.N_sensors; s++) {
            mGateMeas[s] = VectorXi::LinSpaced(Zz[s].cols(), 0, Zz[s].cols() - 1);
        }
    }

    double computePS(int k) {
        int ind;
        mMode.maxCoeff(&ind);
        int start = mGmLen(seq(0, ind - 1)).sum();
        int end = start + mGmLen[ind];
        int indx;
        mW(seq(start, end - 1)).maxCoeff(&indx);
        VectorXd X = mM.col(start + indx);
        double control_param = 0.6;
        bool ch1 = (mModel.XMAX[0] < X[0]) && (X[0] < mModel.XMAX[1]);
        bool ch2 = (mModel.YMAX[0] < X[2]) && (X[2] < mModel.YMAX[1]);
        double scene_mask = 0;
        if (ch1 && ch2) {
            scene_mask = mModel.P_S;
        } else {
            scene_mask = 1 - mModel.P_S;
        }
        double pS = scene_mask / (1 + exp(-control_param * (k - mBirthTime)));
        return pS;
    }

    void cleanupTarget() {
        VectorXd wTemp = mW;
        MatrixXd mTemp = mM;
        vector<MatrixXd> PTemp = vector<MatrixXd>(mP);
        int s_old = 0;
        int e_old = 0;
        int new_idx = 0;
        VectorXi gm_len = VectorXi::Zero(mMode.size());
        VectorXd wc;
        MatrixXd mc;
        vector<MatrixXd> Pc;
        for (int mode = 0; mode < mMode.size(); mode++) {
            e_old += mGmLen[mode];
            VectorXd wIdx = mW(seq(s_old, e_old - 1));
            MatrixXd mIdx = mM(all, seq(s_old, e_old - 1));
            vector<MatrixXd> PIdx(e_old - s_old);
            std::copy(mP.begin() + s_old, mP.begin() + e_old, PIdx.begin());
            std::tie(wc, mc, Pc) = cleanup(wIdx, mIdx, PIdx);
            for (int i = 0; i < wc.size(); i++) {
                wTemp(new_idx) = wc(i);
                mTemp.col(new_idx) = mc.col(i);
                PTemp[new_idx] = Pc[i];
                new_idx = new_idx + 1;
            }
            gm_len[mode] = wc.size();
            s_old = e_old;
        }
        mGmLen = gm_len;
        mW = wTemp(seq(0, new_idx - 1));
        mM = mTemp(all, seq(0, new_idx - 1));
        mP = vector<MatrixXd>(PTemp);
        mP.resize(new_idx);
    }

    void re_activate(MatrixXd m, vector<MatrixXd> Pb, double rb, MatrixXd feat, int time_step) {
        mM(0, seq(0, m.cols() - 1)) = m.row(0);
        mM(2, seq(0, m.cols() - 1)) = m.row(2);
        mP = Pb;
        mW = mModel.w_birth;
        mR = rb;
        mFeatFlag = (feat.rowwise().sum().array() == 0).cast<int>();
        VectorXd averageFeat = norm_feat01(feat.colwise().sum() / (1 - mFeatFlag.array()).sum());
        mFeat.resize(mModel.N_sensors, averageFeat.size());
        for (int s = 0; s < mModel.N_sensors; s++) {
            if (mFeatFlag[s]) {
                mFeat.row(s) = averageFeat;
            } else {
                VectorXd featTemp = mReAlphaFeat * mFeat.row(s);
                featTemp = featTemp + (1 - mReAlphaFeat) * norm_feat01(feat.row(s));
                mFeat.row(s) = norm_feat01(featTemp);
            }
        }
        mMode = mModel.mode;
        mPS = mModel.P_S;
        mGateMeas.resize(mModel.N_sensors);
        mLastActive = time_step;
        mGmLen = VectorXi::Ones(mModel.mode.size());
    }
};

#endif //UKF_TARGET_TARGET_HPP
