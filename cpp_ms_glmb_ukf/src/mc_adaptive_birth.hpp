//
// Created by linh on 2022-04-11.
//

#ifndef MSADAPTIVEBIRTH_MS_ADAPTIVE_BIRTH_HPP
#define MSADAPTIVEBIRTH_MS_ADAPTIVE_BIRTH_HPP

#include <vector>
#include <set>
#include <Eigen/Core>
#include <Eigen/LU>  /* MatrixBase::inversemethode is definded in the Eigen::LU */
#include <random>        /*uniform distribution*/
//#include <iostream>
#include <numeric>   /*std::accumulate*/
//#include <armadillo>
#include <EigenRand/EigenRand>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MIN_DOUBLE
#define MIN_DOUBLE std::nexttoward(0.0, 1.0L)
#endif

using namespace Eigen;
using namespace std;
using param_type = std::uniform_int_distribution<>::param_type;

/* State vector form [x dx y dy z dz ...]*/

MatrixXd gen_msobservation_fn_v2(const MatrixXd &c, MatrixXd X, const VectorXd &meas_n_mu_mode) {
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
    return bbs_noiseless;
}

VectorXd z2x(const Vector2d &z, const MatrixXd &T) {
    // https://www.petercorke.com/RTB/r9/html/homtrans.html
    Vector3d e2h = Vector3d::Ones();
    e2h(seq(0, 1)) = z; // E2H Euclidean to homogeneous
    Vector3d temp = T * e2h; // H2E Homogeneous to Euclidean
    Vector2d pt = temp(seq(0, 1)) / temp(2);
    return pt;
}

MatrixXd jacobian_z2x(const Vector2d &initial, const MatrixXd &T, double delta = 1e-3) {
    // https://rh8liuqy.github.io/Finite_Difference.html
    int nrow = z2x(initial, T).size();
    int ncol = initial.size();
    MatrixXd output = MatrixXd::Zero(nrow, ncol);
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            VectorXd ej = VectorXd::Zero(ncol);
            ej[j] = 1;
            double dij = (z2x(initial + delta * ej, T)[i] -
                          z2x(initial - delta * ej, T)[i]) / (2 * delta);
            output(i, j) = dij;
        }
    }
    return output;
}

VectorXd likelihood(MatrixXd X, VectorXd z, MatrixXd cam_mat, MatrixXd modelR, VectorXd meas_n_mu) {
    MatrixXd Phi = gen_msobservation_fn_v2(cam_mat, X, meas_n_mu);
    MatrixXd nu = Phi.colwise() - z; //  -1.0 * (Phi.colwise() - z);
    MatrixXd esq_temp = nu.array() * (modelR.inverse() * nu).array(); // solve for R*x=nu
    VectorXd log_gz_vals = -0.5 * (esq_temp.colwise().sum().array() + z.size() * log(2 * M_PI)
                                   + log(modelR.determinant()));
    return log_gz_vals;
}

double likelihood_pairwise_gaussian(VectorXd mu, VectorXd z) {
    MatrixXd SIGMA(2, 2);
    SIGMA << 0.125, 0.0, // assume SIGMA = Covariance + R (Noise)
            0.0, 0.125;  // assume SIGMA = diag(0.15, 0.15) + diag(0.15, 0.15)
    VectorXd nu = mu - z;
    double esq_temp = nu.transpose() * SIGMA.inverse() * nu;
    double log_gz_vals = -0.5 * (esq_temp + z.size() * log(2 * M_PI) + log(SIGMA.determinant()));
    return log_gz_vals;
}

VectorXd logProposal(MatrixXd X, VectorXd x_o_z, MatrixXd cov) {
    // observable part of the sampled states
    MatrixXd x_o_s = X(seq(0, 2, 2), all);
    // project z to the observable space:
    MatrixXd nu = x_o_s.colwise() - x_o_z;
    MatrixXd esq_temp = nu.array() * (cov.inverse() * nu).array(); // solve for cov*x=nu
    VectorXd log_gz_vals = -0.5 * (esq_temp.colwise().sum().array() + x_o_z.size() * log(2 * M_PI)
                                   + log(cov.determinant()));
    return log_gz_vals;
}

// This modelR (3D projected noise position) differs from 'mModelR' (bbox detection noise on images)
tuple<VectorXd, MatrixXd> proposalParameter(const MatrixXd &modelR, MatrixXd cam_mat, VectorXd z) {
    Vector3i c_idxs;
    c_idxs << 0, 1, 3;
    MatrixXd T = cam_mat(all, c_idxs).inverse();
    z(0) += exp(z(2)) / 2;
    z(1) += exp(z(3));
    Matrix2d H_J = jacobian_z2x(z(seq(0, 1)), T);
    MatrixXd cov = H_J * modelR * H_J.transpose(); // eq (31)
    Vector2d mu = z2x(z(seq(0, 1)), T);
    return {mu, cov};
}

class MCAdaptiveBirth { // Monte Carlo Approximation
private:
    int mNumSensors;
    double mTaurU;
    int mNumParticles;
    double mLogPDonKappa;
    double mLogQD;
    int mNumMissThres; // at least get detection from n sensors
    int mNumSamples;
    int mDimZ;
    int mDimX;

    MatrixXd mModelR; // bbox detection noise on images [top, left, log(w), log(h)]
    double mRbMax;
    double mRbMin;

    std::mt19937_64 mGenerator;
    uniform_real_distribution<double> mGenDouble;
    uniform_int_distribution<int> mGenInt;

    vector<MatrixXd> mCamMat;
    vector<MatrixXd> mCamMatHomtrans;
    VectorXd mMeasNmu;
    VectorXd mMuInit;
    MatrixXd mCovInit;
    Vector3d mMeanLogNorm;

    static double log_sum_exp(VectorXd arr) {
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

    static tuple<double, double> lognormal_with_mean_one(double percen) {
        double percen_v = pow(percen, 2);
        double std_dev = sqrt(log(percen_v + 1));
        double mean = -pow(std_dev, 2) / 2;
        return {mean, std_dev};
    }

public:
    MCAdaptiveBirth() {}

    MCAdaptiveBirth(vector<MatrixXd> cMat, int nSamples, int nParticles, double PD, double lambdaC, double pdfC) {
        mNumSensors = cMat.size();
        mTaurU = 0.9;
        mNumParticles = nParticles;
        mLogPDonKappa = log(PD) - lambdaC + pdfC; // input of lambdaC & pdfC is logarithm scale
        mLogQD = log(1 - PD);
        mNumMissThres = 2; // at least get detection from n sensors
        mNumSamples = nSamples;
        mDimZ = 4;
        mDimX = 9;
        mRbMax = 0.001;
        mRbMin = 1e-5;

        // Initiate generator for random number used in Gibbs sampling
        std::random_device rd;
        mGenerator = std::mt19937_64(rd());
        mGenDouble = uniform_real_distribution<double>(0.0, 1.0);
        mGenInt = uniform_int_distribution<int>(0, mNumSensors);

        // mode 0 (standing), assume a person stands and enters a scene (NOT fallen)
        double meas_n_mu0, meas_n_std_dev0, meas_n_mu1, meas_n_std_dev1;
        tie(meas_n_mu0, meas_n_std_dev0) = lognormal_with_mean_one(0.1);
        tie(meas_n_mu1, meas_n_std_dev1) = lognormal_with_mean_one(0.05);
        VectorXd d0(4);
        d0 << 20, 20, meas_n_std_dev0, meas_n_std_dev1;
        d0 = (d0.array() * d0.array());
        mModelR = d0.asDiagonal();
        mMeasNmu.resize(2);
        mMeasNmu << meas_n_mu0, meas_n_mu1;

        mCamMat = cMat;
        Vector3i c_idxs;
        c_idxs << 0, 1, 3;
        for (int i = 0; i < cMat.size(); i++) {
            MatrixXd T = cMat[i](all, c_idxs).inverse();
            mCamMatHomtrans.push_back(T);
        }
        // Setting up prior distribution parameters
        double mw, stdw;
        std::tie(mw, stdw) = lognormal_with_mean_one(0.1);
        mMuInit.resize(mDimX);
        VectorXd covInit(mDimX);
        mMuInit << 6.3 / 2, 0.0, 3.41 / 2, 0, 1.65 / 2, 0, log(0.3) + mw, log(0.3) + mw, log(0.84) + mw;
        covInit << 6.3 / 2, 0.001, 3.41 / 2, 0.001, 1.65 / 2, 0.001, stdw, stdw, stdw;
        mCovInit = covInit.asDiagonal();
        mMeanLogNorm << log(0.3), log(0.3), log(0.84);
    }

    void setMeasureNoise(MatrixXd R, VectorXd nMu) {
        mModelR = R;
        mMeasNmu = nMu;
    }

    void setPriorParams(VectorXd mu, VectorXd cov) {
        mMuInit = mu;
        mCovInit = cov.asDiagonal();
    }

    void setBirthProb(double min_rb, double max_rb) {
        mRbMax = max_rb;
        mRbMin = min_rb;
    }

    void setNumSensorDetect(int nSensors) {
        mNumMissThres = nSensors;
    }

    int preProcessMeasurements(vector<VectorXd> meas_rU, vector<MatrixXd> Z, vector<MatrixXd> &pMu,
                               vector<MatrixXd> &pCov, vector<int> &num_meas, vector<int> &meas_flag,
                               vector<VectorXd> &log_meas_rU) {
        /* (I). pre-compute proposal parameters (mu, cov) for ONLY (x, y) */
        for (int i = 0; i < Z.size(); i++) {
            if (Z[i].size() == 0) {
                continue;
            }
            MatrixXd zTmp = Z[i](seq(0, 1), all);
            // get center bottom of bbox
            zTmp.row(0) = zTmp.row(0).array() + Z[i].row(2).array().exp() / 2;
            zTmp.row(1) = zTmp.row(1).array() + Z[i].row(3).array().exp();
            MatrixXd covTmp(2, 2 * Z[i].cols());
            MatrixXd xyFeetTmp(2, Z[i].cols());
            for (int j = 0; j < Z[i].cols(); j++) {
                // Jacobian
                // MatrixXd H_J = jacobian_z2x(zTmp(all, j), mCamMatHomtrans[i]);
                // covTmp(all, seq(2 * j, 2 * j + 1)) = H_J * mModelR(seq(0, 1), seq(0, 1)) * H_J.transpose();
                VectorXd d1(2);
                d1 << 0.25, 0.25; // choose a covariance instead of using Jacobian
                covTmp(all, seq(2 * j, 2 * j + 1)) = (MatrixXd) d1.asDiagonal();
                // Homogeneous transformation: center-bottom of bbox => real-world feet location
                xyFeetTmp.col(j) = z2x(zTmp.col(j), mCamMatHomtrans[i]);
            }
            pCov[i] = covTmp;
            pMu[i] = xyFeetTmp;
        }

        /* (II). Only birth measurements with low association probability, e.g. rU > 0.9 */
        int total_num_meas = 0;
        for (int sidx = 0; sidx < mNumSensors; sidx++) {
            vector<int> idxkeep;
            for (int i = 1; i < meas_rU[sidx].size(); i++) {
                if (meas_rU[sidx](i) > mTaurU) {
                    idxkeep.push_back(i);
                    num_meas[sidx] += 1;
                }
            }
            if (num_meas[sidx]) {
                meas_flag.push_back(sidx);
                total_num_meas += num_meas[sidx];
            }
            VectorXi idxkeep_eigen = VectorXi::Map(idxkeep.data(), idxkeep.size()).array();
            VectorXi meas_keep = VectorXi::Zero(idxkeep.size() + 1);
            meas_keep(seq(1, idxkeep.size())) = idxkeep_eigen; //0 at the front, keep miss-detection
            log_meas_rU[sidx] = meas_rU[sidx](meas_keep).array().log();
            MatrixXd temp = Z[sidx](all, idxkeep_eigen.array() - 1);
            Z[sidx] = temp;
        }
        return total_num_meas;
    }

    /* Discard solutions
        (1) Solutions with number of detections less than a threshold mNumMissThres
        (2) Solutions is subset of other solutions e.g. s1 = [1, 2, 0, 0], s2 = [1, 2, 3, 4]
            We discard s1 because '1'&'2' are in s2 (0 for missed detection, 1,2,3,4 measurement index)
            If s1 = [1, 2, 0, 0], s2 = [1, 0, 0, 2], we keep both s1 & s2
    */
    vector<vector<int>> discardSolutions(vector<vector<int>> assignments) {
        vector<vector<int>> new_assignments(assignments.size());
        int sIdx = 0;
        for (int i = 0; i < assignments.size(); i++) {
            int solSum = 0;
            for (int j = 0; j < mNumSensors; j++) {
                if (assignments[i][j] > 0) {
                    solSum += 1;
                }
            }
            if (solSum <= mNumMissThres) {
                continue; // can discarded all miss-detection solutions
            }
            vector<int> temp = assignments[i];
            bool check = false;
            for (int j = 0; j < sIdx; j++) {
                bool is_subset = true; // temp is subset of assignments[i]
                int check_biggerset = 0; // new_assignments[j] is subset of temp
                for (int k = 0; k < temp.size(); k++) {
                    if (temp[k] != new_assignments[j][k] && temp[k] > 0) {
                        is_subset = false;
                        if (new_assignments[j][k] > 0) {
                            break;
                        }
                    }
                    if (temp[k] == new_assignments[j][k] || (temp[k] != new_assignments[j][k] && temp[k] > 0)) {
                        check_biggerset++;
                    }
                }
                if (is_subset) {
                    check = true;
                    break;
                }
                if (check_biggerset == temp.size()) {
                    new_assignments.erase(new_assignments.begin() + j);
                    j--; // Decrement the index to correctly check the next solution
                    sIdx--;
                }
            }
            if (check) {
                continue;
            }
            new_assignments[sIdx] = temp;
            sIdx = sIdx + 1;
        }
        new_assignments.resize(sIdx);
        return new_assignments;
    }

    tuple<vector<VectorXd>, vector<double>, vector<vector<int>>>
    sample_adaptive_birth(vector<VectorXd> meas_rU, vector<MatrixXd> Z) {
        vector<int> meas_flag;
        vector<int> num_meas(mNumSensors, 0);
        vector<VectorXd> log_meas_rU(mNumSensors);
        vector<int> currsol(mNumSensors, 0); // 0: missed detection index, 1-|Z| measurement index
        vector<MatrixXd> pMu(mNumSensors);
        vector<MatrixXd> pCov(mNumSensors);

        /* (I). pre-compute */
        int total_num_meas = preProcessMeasurements(meas_rU, Z, pMu, pCov, num_meas, meas_flag, log_meas_rU);
        // Skip if there is no measurement
        if (total_num_meas == 0) {
            return {vector<VectorXd>(0), vector<double>(0), vector<vector<int>>(0)};
        }

        /* (II). Gibbs sampling */
        vector<vector<int>> assignments(mNumSamples, vector<int>(mNumSensors, 0));
        // initialise to work around the non-miss-detection requirement
        // randomly initialize solution with at least 1 detection
        int sel_sensor = meas_flag[mGenInt(mGenerator, param_type(0, meas_flag.size() - 1))];
        currsol[sel_sensor] = mGenInt(mGenerator, param_type(1, num_meas[sel_sensor]));
        assignments[0] = currsol;

        for (int sol = 1; sol < mNumSamples; sol++) {
            VectorXi sensor_indices = VectorXi::LinSpaced(mNumSensors, 0, mNumSensors - 1);
            std::shuffle(sensor_indices.begin(), sensor_indices.end(), mGenerator);
            int check_all_miss = 0;
            for (int sidx : sensor_indices) {
                VectorXd log_samp_dist = VectorXd::Zero(num_meas[sidx] + 1);
                // sample the observable states
                for (int midx = 0; midx < num_meas[sidx] + 1; midx++) {
                    vector<int> sampsol(currsol);
                    sampsol[sidx] = midx;
                    vector<int> det_idx;
                    bool nall_miss_detect = false;
                    for (int i = 0; i < mNumSensors; i++) {
                        if (sampsol[i] > 0) {
                            det_idx.push_back(i);
                            nall_miss_detect = true;
                        }
                    }
                    /* start constructing proposal distribution */
                    // initialise samples, sample from prior distribution
                    Eigen::Rand::MvNormalGen<double, Eigen::Dynamic> genInit{mMuInit, mCovInit};
                    MatrixXd x_samples = genInit.generate(mGenerator, mNumParticles);

                    int s_sensor;
                    if (nall_miss_detect) {
                        // pick a random non-miss-detection sample from J,  s' in the paper
                        s_sensor = det_idx[mGenInt(mGenerator, param_type(0, det_idx.size() - 1))];
                        int s_midx = sampsol[s_sensor] - 1;
                        // compute parameters of the proposal distribution
                        VectorXd mu = pMu[s_sensor].col(s_midx);
                        MatrixXd cov = pCov[s_sensor](all, seq(2 * s_midx, 2 * s_midx + 1));
                        // sample from the proposal distribution
                        Eigen::Rand::MvNormalGen<double, Eigen::Dynamic> proposal{mu, cov};
                        MatrixXd x_o_samples = proposal.generate(mGenerator, mNumParticles);

                        // copy the observable part of state to the initial samples
                        x_samples((seq(0, 2, 2)), all) = x_o_samples;
                    }
                    // start computing the sampling distribution
                    // compute the proposal distribution and psi (for eqn 32)
                    VectorXd log_proposal_qz = VectorXd::Zero(mNumParticles); // log_proposal_qz_sample_sensor
                    MatrixXd log_psiz = MatrixXd::Ones(mNumSensors, mNumParticles); // from eq(10)
                    for (int isidx = 0; isidx < mNumSensors; isidx++) {
                        int imidx = sampsol[isidx];
                        if (imidx > 0) {
                            VectorXd isidx_z = Z[isidx].col(imidx - 1);
                            VectorXd temp = likelihood(x_samples, isidx_z, mCamMat[isidx], mModelR, mMeasNmu);
                            log_psiz.row(isidx) = mLogPDonKappa + temp.array();
                            if (isidx == s_sensor) {
                                VectorXd mu = pMu[isidx].col(imidx - 1);
                                MatrixXd cov = pCov[isidx](all, seq(2 * (imidx - 1), 2 * (imidx - 1) + 1));
                                log_proposal_qz = logProposal(x_samples, mu, cov);
                            }
                        } else {
                            log_psiz.row(isidx) *= mLogQD;
                        }
                    }
                    VectorXd log_psiz_temp = log_psiz.colwise().sum();
                    VectorXd w_samples = log_psiz_temp - log_proposal_qz;
                    double log_psi_bar = log_sum_exp(w_samples) - log(mNumParticles);
                    log_samp_dist(midx) = log_meas_rU[sidx](midx) + log_psi_bar;
                }
                // run categorical sampling
                VectorXd temp_samp_dist = (log_samp_dist.array() - log_sum_exp(log_samp_dist)).exp();
                double rand_num = mGenDouble(mGenerator);
                VectorXd cdf(num_meas[sidx] + 1);
                cdf[0] = temp_samp_dist[0];
                for (int i = 1; i < num_meas[sidx] + 1; i++) {
                    cdf[i] = cdf[i - 1] + temp_samp_dist[i];
                }
                int sum_cdf = 0;
                for (int i = 0; i < num_meas[sidx] + 1; i++) {
                    sum_cdf += (int) (cdf[i] < (rand_num * cdf[num_meas[sidx]]));
                }
                currsol[sidx] = sum_cdf; // 0 -> |M|, 0: missed, 1 -> |M|: measurement index
                check_all_miss += sum_cdf;
            }
            // discard all-miss-detection solution, replace it with a random, solution with at least 1 detection
            if (check_all_miss == 0) {
                for (int iisidx = 0; iisidx < mNumSensors; iisidx++) {
                    currsol[iisidx] = mGenInt(mGenerator, param_type(0, num_meas[iisidx]));
                    int sel_sensor = meas_flag[mGenInt(mGenerator, param_type(0, meas_flag.size() - 1))];
                    // exclude miss-detection
                    currsol[sel_sensor] = mGenInt(mGenerator, param_type(1, num_meas[sel_sensor]));
                }
            }
            assignments[sol] = currsol;
            // reset the current solution to the all-missed detection measurement tuple to encourage exploration
            //if (sol % 100 == 0) {
            //    std::fill(currsol.begin(), currsol.end(), 0);
            //}
        }
        std::sort(assignments.begin(), assignments.end());
        assignments.erase(std::unique(assignments.begin(), assignments.end()), assignments.end()); // unique solutions
        /* (II). Discard solutions */
        assignments = discardSolutions(assignments);

        /* (III). Construct birth from the sampled solution */
        int bidx = 0;
        int num_sols = assignments.size();
        MatrixXd m_birth(mDimX, num_sols);
        VectorXd log_r_b(num_sols);
        vector<vector<int>> temp_sols_birth;
        for (int solidx = 0; solidx < num_sols; solidx++) {
            vector<int> sol = assignments[solidx];
            // start constructing proposal distribution
            vector<int> det_idx; // idx of sensors with detection
            int sol_sum = 0;
            for (int i = 0; i < mNumSensors; i++) {
                if (sol[i] > 0) {
                    det_idx.push_back(i);
                    sol_sum += 1;
                }
            }
            temp_sols_birth.push_back(sol);
            // pick a random non-miss-detection sample from J,  // s' in the paper
            int s_sensor = det_idx[mGenInt(mGenerator, param_type(0, det_idx.size() - 1))];
            int s_midx = sol[s_sensor] - 1;
            // compute parameters of the proposal distribution
            VectorXd mu = pMu[s_sensor].col(s_midx);
            MatrixXd cov = pCov[s_sensor](all, seq(2 * s_midx, 2 * s_midx + 1));
            // sample from the proposal distribution
            Eigen::Rand::MvNormalGen<double, Eigen::Dynamic> proposal{mu, cov};
            MatrixXd x_o_samples = proposal.generate(mGenerator, mNumParticles);
            Eigen::Rand::MvNormalGen<double, Eigen::Dynamic> genInit{mMuInit, mCovInit};
            MatrixXd x_samples = genInit.generate(mGenerator, mNumParticles);

            x_samples((seq(0, 2, 2)), all) = x_o_samples;

            VectorXd log_proposal_qz = VectorXd::Zero(mNumParticles); // log_proposal_qz_sample_sensor
            MatrixXd log_psiz = MatrixXd::Ones(mNumSensors, mNumParticles);
            double curr_log_meas_rU = 0;
            for (int isidx = 0; isidx < mNumSensors; isidx++) {
                int imidx = sol[isidx];
                if (imidx > 0) {
                    VectorXd temp = likelihood(x_samples, Z[isidx].col(imidx - 1), mCamMat[isidx], mModelR, mMeasNmu);
                    log_psiz.row(isidx) = mLogPDonKappa + temp.array();
                    if (isidx == s_sensor) {
                        VectorXd mu = pMu[isidx].col(imidx - 1);
                        MatrixXd cov = pCov[isidx](all, seq(2 * (imidx - 1), 2 * (imidx - 1) + 1));
                        log_proposal_qz = logProposal(x_samples, mu, cov);
                    }
                } else {
                    log_psiz.row(isidx) *= mLogQD;
                }
                curr_log_meas_rU = curr_log_meas_rU + log_meas_rU[isidx](imidx);
            }
            VectorXd log_psiz_temp = log_psiz.colwise().sum();
            VectorXd w_samples = log_psiz_temp - log_proposal_qz;
            double logsumexp_w_samples = log_sum_exp(w_samples);
            w_samples = (w_samples.array() - logsumexp_w_samples).exp();

            // compute birth weights
            // very few unique indices after weighted resampling, no need to use Expectation Maximization
            // use weighted mean as an approximation
            VectorXd weighted_mean = (x_samples.array().rowwise() * w_samples.array().transpose()).rowwise().sum();
            double log_psi_bar = logsumexp_w_samples - log(mNumParticles);
            log_r_b(bidx) = curr_log_meas_rU + log_psi_bar;
            m_birth.col(bidx) = weighted_mean;
            bidx += 1;
        }

        VectorXd r_b = log_r_b(seq(0, bidx - 1));
        r_b = (r_b.array() - log_sum_exp(r_b)).array().exp();
        // prune low weight birth
        vector<vector<int>> sols_birth;
        vector<VectorXd> sols_m;
        vector<double> sols_r;
        for (int i = 0; i < bidx; i++) {
            if (r_b(i) > mRbMin) {
                sols_m.push_back(m_birth(all, i));
                sols_r.push_back(std::min(r_b(i), mRbMax));
                sols_birth.push_back(temp_sols_birth[i]);
            }
        }

        return {sols_m, sols_r, sols_birth};
    }

    tuple<vector<vector<int>>, vector<vector<float>>> sample_mc_sols(vector<VectorXd> meas_rU, vector<MatrixXd> Z) {
        vector<int> meas_flag;
        vector<int> num_meas(mNumSensors, 0);
        vector<VectorXd> log_meas_rU(mNumSensors);
        vector<int> currsol(mNumSensors, 0); // 0: missed detection index, 1-|Z| measurement index
        vector<MatrixXd> pMu(mNumSensors);
        vector<MatrixXd> pCov(mNumSensors);

        /* (I). pre-compute */
        int total_num_meas = preProcessMeasurements(meas_rU, Z, pMu, pCov, num_meas, meas_flag, log_meas_rU);
        // Skip if there is no measurement
        if (total_num_meas == 0) {
            return {vector<vector<int>>(0), vector<vector<float>>(0)};
        }

        /* (II). Gibbs sampling */
        vector<vector<int>> assignments(mNumSamples, vector<int>(mNumSensors, 0));
        // initialise to work around the non-miss-detection requirement
        // randomly initialize solution with at least 1 detection
        int sel_sensor = meas_flag[mGenInt(mGenerator, param_type(0, meas_flag.size() - 1))];
        currsol[sel_sensor] = mGenInt(mGenerator, param_type(1, num_meas[sel_sensor]));
        assignments[0] = currsol;

        int s_sensor = sel_sensor;
        for (int sol = 1; sol < mNumSamples; sol++) {
            VectorXi sensor_indices = VectorXi::LinSpaced(mNumSensors, 0, mNumSensors - 1);
            std::shuffle(sensor_indices.begin(), sensor_indices.end(), mGenerator);
            int check_all_miss = 0;
            for (int sidx : sensor_indices) {
                VectorXd log_samp_dist = VectorXd::Zero(num_meas[sidx] + 1);
                // sample the observable states
                for (int midx = 0; midx < num_meas[sidx] + 1; midx++) {
                    vector<int> sampsol(currsol);
                    sampsol[sidx] = midx;
                    vector<int> det_idx;
                    bool nall_miss_detect = false;
                    for (int i = 0; i < mNumSensors; i++) {
                        if (sampsol[i] > 0) {
                            det_idx.push_back(i);
                            nall_miss_detect = true;
                        }
                    }
                    /* start constructing proposal distribution */
                    if (nall_miss_detect) {
                        // pick a random non-miss-detection sample from J,  s' in the paper
                        s_sensor = det_idx[mGenInt(mGenerator, param_type(0, det_idx.size() - 1))];
                    }
                    // start computing the sampling distribution
                    VectorXd mu_ssensor = pMu[s_sensor].col(sampsol[s_sensor] - 1);
                    VectorXd log_psiz = VectorXd::Ones(mNumSensors);
                    for (int isidx = 0; isidx < mNumSensors; isidx++) {
                        int imidx = sampsol[isidx];
                        if (imidx > 0) {
                            VectorXd isidx_z = pMu[isidx].col(imidx - 1);
                            double temp = likelihood_pairwise_gaussian(mu_ssensor, isidx_z);
                            log_psiz(isidx) = mLogPDonKappa + temp;
                        } else {
                            log_psiz(isidx) = mLogQD;
                        }
                    }
                    log_samp_dist(midx) = log_meas_rU[sidx](midx) + log_psiz.sum();
                }
                // run categorical sampling
                VectorXd temp_samp_dist = (log_samp_dist.array() - log_sum_exp(log_samp_dist)).exp();
                double rand_num = mGenDouble(mGenerator);
                VectorXd cdf(num_meas[sidx] + 1);
                cdf[0] = temp_samp_dist[0];
                for (int i = 1; i < num_meas[sidx] + 1; i++) {
                    cdf[i] = cdf[i - 1] + temp_samp_dist[i];
                }
                int sum_cdf = 0;
                for (int i = 0; i < num_meas[sidx] + 1; i++) {
                    sum_cdf += (int) (cdf[i] < (rand_num * cdf[num_meas[sidx]]));
                }
                currsol[sidx] = sum_cdf; // 0 -> |M|, 0: missed, 1 -> |M|: measurement index
                check_all_miss += sum_cdf;
            }
            // discard all-miss-detection solution, replace it with a random, solution with at least 1 detection
            if (check_all_miss == 0) {
                for (int iisidx = 0; iisidx < mNumSensors; iisidx++) {
                    currsol[iisidx] = mGenInt(mGenerator, param_type(0, num_meas[iisidx]));
                    sel_sensor = meas_flag[mGenInt(mGenerator, param_type(0, meas_flag.size() - 1))];
                    // exclude miss-detection
                    currsol[sel_sensor] = mGenInt(mGenerator, param_type(1, num_meas[sel_sensor]));
                }
            }
            assignments[sol] = currsol;
        }
        std::sort(assignments.begin(), assignments.end());
        assignments.erase(std::unique(assignments.begin(), assignments.end()), assignments.end()); // unique solutions
        /* (II). Discard solutions */
        assignments = discardSolutions(assignments);
        vector<vector<float>> centers(assignments.size());
        for (int i = 0; i < assignments.size(); i++) {
            vector<float> center(2, 0);
            int number_meas = 0;
            for (int j = 0; j < assignments[i].size(); j++) {
                int meas_idx = assignments[i][j];
                if (meas_idx > 0) {
                    center[0] = center[0] + pMu[j](0, meas_idx - 1);
                    center[1] = center[1] + pMu[j](1, meas_idx - 1);
                    number_meas++;
                }
            }
            if (number_meas > 0) {
                center[0] = center[0] / number_meas;
                center[1] = center[1] / number_meas;
            }
            centers[i] = center;
        }
        return {assignments, centers};
    }
};

#endif //MSADAPTIVEBIRTH_MS_ADAPTIVE_BIRTH_HPP
