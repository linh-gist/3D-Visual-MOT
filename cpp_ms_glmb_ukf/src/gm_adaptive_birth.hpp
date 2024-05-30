//
// Created by linh on 2022-03-17.
// C++ Implementation  of Algorithm 1 Multi-sensor Adaptive Birth Gibbs Sampler, (VII). GAUSSIAN LIKELIHOODS
// Trezza, A., BUCCI, D. J., & Varshney, P. K. (2022).
// Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking.
// IEEE Transactions on Signal Processing.
//

#ifndef GIBBS_MS_ADAPTIVE_BIRTH_H
#define GIBBS_MS_ADAPTIVE_BIRTH_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>     /* MatrixBase::inversemethode is definded in the Eigen::LU */
#include <random>       /*uniform distribution*/
//#include <iostream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using namespace std;

class AdaptiveBirth {
private:
    /* Dynamic */
    MatrixXd m_F; // Motion model: state transition matrix
    MatrixXd m_B; // use to construct process noise covariance
    MatrixXd m_Q;
    int m_x_dim;

    /* Sensor */
    int m_num_sensors;
    VectorXd m_PD; // detection probability
    VectorXd m_lambda_c; // poisson average rate of uniform clutter (per scan)
    VectorXd m_pdf_c; // uniform clutter density
    vector<MatrixXd> m_R; // observation noise covariance from each sensor
    int m_z_dim;
    VectorXd m_log_QD;
    vector<MatrixXd> m_invR;
    VectorXd m_log_detR;
    vector<MatrixXd> m_H; // observation matrix of each sensor
    vector<MatrixXd> m_H_invR;
    VectorXd m_PD_kappa;
    VectorXd m_log_PD_kappa;

    /* Adaptive Birth */
    VectorXd m_mu0;
    MatrixXd m_P0;
    int m_H_gibbs;
    MatrixXd m_invP0;
    double m_tau_rU;
    int m_num_miss_thres; // number of sensors needed to detect object and form a new birth target
    double m_r_bmax;

    VectorXd m_b_J0;
    vector<MatrixXd> m_b_J;
    double m_c_J0;
    vector<MatrixXd> m_c_J;
    double m_pi_zdim;

    std::mt19937 m_generator;
    uniform_real_distribution<double> m_distribution;

    std::tuple<MatrixXd, MatrixXd> ekf_update_mat() {
        MatrixXd H(3, 6), U(3, 3);
        H << 1., 0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0., 0.,
                0., 0., 0., 0., 1., 0.;
        U << 1.0, 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
        return {H, U};
    }

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
    AdaptiveBirth(){};

    void init_parameters(int num_sensors, MatrixXd F, MatrixXd Q, double PD, double lambda_c, double pdf_c, MatrixXd R,
                         VectorXd mu0, MatrixXd P0, double tau_rU, int H_gibbs, int num_miss, double r_bmax) {
        // MODEL - dynamic
        m_F = F;
        m_Q = Q;
        m_x_dim = 6;

        // MODEL - sensor
        m_num_sensors = num_sensors; // N_sensors
        m_PD = VectorXd::Ones(m_num_sensors);
        m_PD = PD * m_PD;
        m_lambda_c = VectorXd::Ones(m_num_sensors);
        m_lambda_c = lambda_c * m_lambda_c;
        m_pdf_c = VectorXd::Ones(m_num_sensors);
        m_pdf_c = pdf_c * m_pdf_c;
        m_R.resize(m_num_sensors);
        m_z_dim = 3;
        for (int i = 0; i < m_num_sensors; i++) {
            m_R[i] = R;
        }
        m_log_QD = (1 - m_PD.array()).log();
        m_invR.resize(m_num_sensors);
        MatrixXd H, U;
        std::tie(H, U) = ekf_update_mat();
        m_H.resize(m_num_sensors);
        m_log_detR.resize(m_num_sensors);
        m_H_invR.resize(m_num_sensors);
        for (int i = 0; i < m_num_sensors; i++) {
            m_H[i] = H;
            m_R[i] = (U * m_R[i]) * U.transpose();
            m_invR[i] = m_R[i].inverse();
            m_log_detR(i) = log(m_R[i].determinant());
            m_H_invR[i] = m_H[i].transpose() * m_R[i].inverse(); // H'R^(-1)
        }
        m_PD_kappa = m_PD.array() / (m_lambda_c.array() * m_pdf_c.array());
        m_log_PD_kappa = m_PD_kappa.array().log();

        // MODEL - adaptive_birth
        m_mu0 = mu0;
        m_H_gibbs = H_gibbs; // 1000
        m_P0 = P0;
        m_invP0 = m_P0.inverse();
        m_tau_rU = tau_rU; // 0.9
        m_num_miss_thres = num_miss; // 2
        m_r_bmax = r_bmax; //0.01

        // pre-compute for eq(41-45)
        m_b_J0 = m_invP0 * m_mu0; // first term of eq(44)
        m_b_J.resize(m_num_sensors);
        m_c_J0 = (m_mu0.transpose() * m_invP0) * m_mu0; // first term of eq(45)
        m_c_J.resize(m_num_sensors);
        m_pi_zdim = m_z_dim * log(2 * M_PI);

        // inital for random number used in Gibbs sampling
        std::random_device rand_dev;
        m_generator = std::mt19937(rand_dev());
        m_distribution = uniform_real_distribution<double>(0.0, 1.0);
    }

    AdaptiveBirth(int num_sensors) {
        // MODEL - dynamic
        m_F.resize(6, 6);
        m_F << 1., 1., 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0.,
                0., 0., 1., 1., 0., 0.,
                0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 1., 1.,
                0., 0., 0., 0., 0., 1.;
        m_B.resize(6, 3);
        m_B << 0.5, 0., 0.,
                1., 0., 0.,
                0., 0.5, 0.,
                0., 1., 0.,
                0., 0., 0.5,
                0., 0., 1.;
        float sigma_v = 0.0225; // 0.15^2
        m_Q = m_B * sigma_v * m_B.transpose();
        m_x_dim = 6;

        // MODEL - sensor
        m_num_sensors = num_sensors; // N_sensors
        m_PD = VectorXd::Ones(m_num_sensors);
        m_PD = 0.95 * m_PD;
        m_lambda_c = VectorXd::Ones(m_num_sensors);
        m_lambda_c = 15 * m_lambda_c;
        m_pdf_c = VectorXd::Ones(m_num_sensors);
        m_pdf_c = 1.0e-06 * m_pdf_c;
        m_R.resize(m_num_sensors);
        m_z_dim = 3;
        MatrixXd R(m_z_dim, m_z_dim);
        R << 100.0, 0., 0.,
                0., 100.0, 0.,
                0., 0., 100.0;
        for (int i = 0; i < m_num_sensors; i++) {
            m_R[i] = R;
        }

        m_log_QD = (1 - m_PD.array()).log();
        m_invR.resize(m_num_sensors);
        MatrixXd H, U;
        std::tie(H, U) = ekf_update_mat();
        m_H.resize(m_num_sensors);
        m_log_detR.resize(m_num_sensors);
        m_H_invR.resize(m_num_sensors);
        for (int i = 0; i < m_num_sensors; i++) {
            m_H[i] = H;
            m_R[i] = (U * m_R[i]) * U.transpose();
            m_invR[i] = m_R[i].inverse();
            m_log_detR(i) = log(m_R[i].determinant());
            m_H_invR[i] = m_H[i].transpose() * m_R[i].inverse(); // H'R^(-1)
        }
        m_PD_kappa = m_PD.array() / (m_lambda_c.array() * m_pdf_c.array());
        m_log_PD_kappa = m_PD_kappa.array().log();

        // MODEL - adaptive_birth
        m_mu0.resize(m_x_dim);
        m_mu0 << 0., 0., 0., 0., 0., 0.;
        m_H_gibbs = 1000;
        m_P0.resize(m_x_dim, m_x_dim);
        m_P0 << 4000000.0, 0.0, 0., 0., 0., 0.,
                0., 2500.0, 0., 0., 0., 0.,
                0., 0., 4000000., 0., 0., 0.,
                0., 0., 0., 2500.0, 0., 0.,
                0., 0., 0., 0., 4000000.0, 0.,
                0., 0., 0., 0., 0., 2500.0;
        m_invP0 = m_P0.inverse();
        m_num_miss_thres = 2;
        m_tau_rU = 0.9;
        m_r_bmax = 0.01;

        // pre-compute for eq(41-45)
        m_b_J0 = m_invP0 * m_mu0; // first term of eq(44)
        m_b_J.resize(m_num_sensors);
        m_c_J0 = (m_mu0.transpose() * m_invP0) * m_mu0; // first term of eq(45)
        m_c_J.resize(m_num_sensors);
        m_pi_zdim = m_z_dim * log(2 * M_PI);

        // inital for random number used in Gibbs sampling
        std::random_device rand_dev;
        m_generator = std::mt19937(rand_dev());
        m_distribution = uniform_real_distribution<double>(0.0, 1.0);
    }

    tuple<MatrixXd, vector<MatrixXd>, VectorXd> sample_adaptive_birth(vector<VectorXd> meas_rU, vector<MatrixXd> Z) {
        /* (I). Initiate value before running Gibbs sampling */
        vector<int> num_meas(m_num_sensors, 0);
        vector<MatrixXd> log_meas_rU(m_num_sensors); // rU = 1 - rA
        for (int sidx = 0; sidx < m_num_sensors; sidx++) {
            vector<int> idxkeep;
            for (int i = 1; i < meas_rU[sidx].size(); i++) {
                if (meas_rU[sidx](i) > m_tau_rU) {
                    idxkeep.push_back(i);
                    num_meas[sidx] += 1;
                }
            }
            VectorXi idxkeep_eigen = VectorXi::Map(idxkeep.data(), idxkeep.size()).array();
            VectorXi meas_keep = VectorXi::Zero(idxkeep.size() + 1);
            meas_keep(seq(1, idxkeep.size())) = idxkeep_eigen; //0 at the front, keep miss-detection
            log_meas_rU[sidx] = meas_rU[sidx](meas_keep).array().log();
            MatrixXd temp = Z[sidx](all, idxkeep_eigen.array() - 1);
            MatrixXd b_J_temp(m_x_dim, num_meas[sidx]);
            VectorXd c_J_temp(num_meas[sidx]);
            for (int midx = 0; midx < num_meas[sidx]; midx++) {
                b_J_temp.col(midx) = m_H_invR[sidx] * temp.col(midx); // H'R^(-1)Z second term of eq(44)
                c_J_temp(midx) = temp.col(midx).transpose() * m_invR[sidx] * temp.col(midx); // second, eq(45)
            }
            m_b_J[sidx] = b_J_temp;
            m_c_J[sidx] = c_J_temp;
        }
        /* (II). Gibbs sampling, according to Theorem V.1. eq(27), solution for Gaussian model eq (46a, 46b) */
        vector<vector<int>> assignments(m_H_gibbs, vector<int>(m_num_sensors, 0));
        vector<int> currsol(m_num_sensors, 0);
        assignments[0] = currsol; // use miss-detection tuple as initial sol
        for (int sol = 1; sol < m_H_gibbs; sol++) {
            VectorXi sensor_indices = VectorXi::LinSpaced(m_num_sensors, 0, m_num_sensors - 1);
            std::shuffle(sensor_indices.begin(), sensor_indices.end(), m_generator);
            for (int sidx : sensor_indices) {
                // (1). compute the sampling distribution
                VectorXi sensor_flag = VectorXi::Ones(m_num_sensors);
                sensor_flag(sidx) = 0; // rule out sensor 'sidx'
                MatrixXd M_J = m_invP0;
                double c_J = m_c_J0;
                VectorXd b_J = m_b_J0;
                for (int isidx = 0; isidx < m_num_sensors; isidx++) {
                    // exclude the current sensor (due to J^(-s))
                    // only add the detected (due to constrain j^(s)>0)
                    if (sensor_flag(isidx) && currsol[isidx] > 0) {
                        M_J = M_J + m_H_invR[isidx] * m_H[isidx]; // eq (43)
                        // currsol[isidx] in [0, |Zs|], 0:missed detection, 1-|Zs|:measurement index
                        b_J = b_J + m_b_J[isidx].col(currsol[isidx] - 1); // eq (44)
                        c_J = c_J + m_c_J[isidx](currsol[isidx] - 1); // eq (45)
                    }
                }
                VectorXd log_samp_dist(num_meas[sidx] + 1);
                double log_Phi_J = -0.5 * (c_J - b_J.transpose() * M_J.inverse() * b_J); // eq (42)
                log_samp_dist(0) = m_log_QD(sidx) - 0.5 * log(M_J.determinant()) + log_Phi_J; // eq (46a)
                // compute the measurement terms of the sampling distribution
                M_J = M_J + m_H_invR[sidx] * m_H[sidx]; // add the term of current sensor in
                double front = -0.5 * (m_pi_zdim + log(M_J.determinant()) + m_log_detR(sidx)) + m_log_PD_kappa(sidx);
                for (int midx = 1; midx < num_meas[sidx] + 1; midx++) {
                    VectorXd b_J_temp = b_J + m_b_J[sidx].col(midx - 1);
                    double c_J_temp = c_J + m_c_J[sidx](midx - 1);
                    double log_Phi_J_temp = -0.5 * (c_J_temp - b_J_temp.transpose() * M_J.inverse() * b_J_temp);
                    log_samp_dist(midx) = log_meas_rU[sidx](midx - 1) + (front + log_Phi_J_temp); // eq(46b)
                }

                // (2). categorical sampling
                VectorXd temp_samp_dist = (log_samp_dist.array() - log_sum_exp(log_samp_dist)).exp();
                VectorXd cdf(num_meas[sidx] + 1);
                cdf[0] = temp_samp_dist[0];
                for (int i = 1; i < num_meas[sidx] + 1; i++) {
                    cdf[i] = cdf[i - 1] + temp_samp_dist[i];
                }
                int sum_cdf = 0;
                double rand_num = m_distribution(m_generator);
                for (int i = 0; i < num_meas[sidx] + 1; i++) {
                    sum_cdf += (int) (cdf[i] < (rand_num * cdf[num_meas[sidx]]));
                }
                currsol[sidx] = sum_cdf;
            }
            assignments[sol] = currsol;
            // reset the current solution to the all-missed detection measurement tuple to encourage exploration
            //if (sol % 100 == 0) {
            //    std::fill(currsol.begin(), currsol.end(), 0);
            //}
        }
        std::sort(assignments.begin(), assignments.end());
        assignments.erase(std::unique(assignments.begin(), assignments.end()), assignments.end()); // unique solutions

        /* (III). Construct birth from the sampled solution */
        int num_sols = assignments.size();
        MatrixXd m_birth(m_x_dim, num_sols);
        vector<MatrixXd> P_birth(num_sols, MatrixXd(m_x_dim, m_x_dim));
        VectorXd log_r_b(num_sols);

        int bidx = 0;
        double log_det_P0 = log(m_P0.determinant());
        double log_const_2pi = 0;
        for (int sidx = 0; sidx < m_num_sensors; sidx++) {
            log_const_2pi = log_const_2pi + m_z_dim * log(2 * M_PI) + m_log_detR(sidx);
        }
        for (int solidx = 0; solidx < num_sols; solidx++) {
            vector<int> sol = assignments[solidx];
            int sum_sol = 0;
            std::for_each(sol.begin(), sol.end(), [&](int n) {
                sum_sol += (int) (n > 0);
            });
            if (sum_sol < m_num_miss_thres) { // at least n component of J are detected to form a new birth target.
                continue;
            }
            MatrixXd M_J = m_invP0;
            VectorXd b_J = m_b_J0;
            double c_J = m_c_J0;
            double log_rU = 0;
            double log_sum_PQD = 0; // sum(log_Q_D(~det_flag)) + sum(log_P_D_on_kappa(det_flag); det_flag = sol > 1;
            for (int sidx = 0; sidx < m_num_sensors; sidx++) {
                if (sol[sidx] > 0) {
                    M_J = M_J + m_H_invR[sidx] * m_H[sidx];
                    b_J = b_J + m_b_J[sidx].col(sol[sidx] - 1);
                    c_J = c_J + m_c_J[sidx](sol[sidx] - 1);
                    log_sum_PQD += m_log_PD_kappa(sidx);
                } else {
                    log_sum_PQD += m_log_QD(sidx);
                }
                log_rU = log_rU + log_meas_rU[sidx](sol[sidx]);
            }
            //det_flag = sol > 1;
            MatrixXd F_inv_M_J = m_F * M_J.inverse();
            m_birth.col(bidx) = F_inv_M_J * b_J;
            P_birth[bidx] = F_inv_M_J * m_F.transpose() + m_Q;
            double log_Phi_J = -0.5 * (c_J - b_J.transpose() * M_J.inverse() * b_J);
            double log_psi_J = -0.5 * (log_det_P0 + log(M_J.determinant()) + log_const_2pi) + log_sum_PQD + log_Phi_J;
            log_r_b(bidx) = log_rU + log_psi_J;
            bidx = bidx + 1;
        }
        P_birth.resize(bidx); // P_birth.erase(P_birth.begin() + bidx, P_birth.end());
        VectorXd r_b = log_r_b(seq(0, bidx - 1));
        r_b = (r_b.array() - log_sum_exp(r_b)).array().exp();
        r_b = (r_b.array() < m_r_bmax).select(r_b, m_r_bmax).array() + std::nexttoward(0.0, 1.0L); // r_bmax = 0.01
        return {m_birth(all, seq(0, bidx - 1)), P_birth, r_b};
    }
};


#endif //GIBBS_MS_ADAPTIVE_BIRTH_H
