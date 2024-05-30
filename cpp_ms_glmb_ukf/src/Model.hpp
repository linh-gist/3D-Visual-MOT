//
// Created by linh on 2022-04-01.
//

#ifndef UKF_TARGET_MODEL_HPP
#define UKF_TARGET_MODEL_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>  /* MatrixBase::inversemethode is definded in the Eigen::LU */
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

Vector2d lognormal_with_mean_one(double percen) {
    // input is std dev of multiplicative lognormal noise.
    Vector2d mean_std;
    double percen_v = percen * percen;
    mean_std(1) = sqrt(log(percen_v + 1));
    mean_std(0) = -mean_std(1) * mean_std(1) / 2;
    return mean_std;
}

class Model {
public:
    // basic parameters
    int N_sensors;
    int x_dim;
    int z_dim;
    Vector2d XMAX;
    Vector2d YMAX;
    Vector2d ZMAX;
    VectorXd mode;
    vector<string> mode_type;

    // camera positions, image size and room dimensions
    MatrixXd sensor_pos;
    Vector2d imagesize;

    // load camera parameters
    vector<MatrixXd> camera_mat;

    // dynamical model parameters (CV model)
    MatrixXd F;
    vector<MatrixXd> Q;
    MatrixXd mode_trans_matrix;

    // measurement parameters
    vector<MatrixXd> R;
    MatrixXd meas_n_mu;
    MatrixXd n_mu;

    // survival/death parameters
    double P_S;
    double Q_S;

    // detection probabilities
    double P_D;
    double Q_D;

    MatrixXd m_birth;
    vector<MatrixXd> P_birth;
    vector<double> r_birth;
    VectorXd w_birth;
    double lambda_c;
    double pdf_c;

    double taurU;
    double numDet;
    double rBMin;
    double rBMax;

    Model() {};

    Model(vector<MatrixXd> camera_mat, string dataset) {
        // CMC dataset
        N_sensors = 4;
        XMAX << 2.03, 6.3; // 2.03 5.77 6.3
        YMAX << 0.00, 3.41; // [0.05 3.41];
        x_dim = 9;
        z_dim = 4;

        ZMAX << 0, 3; // 5.77
        mode_type = {"Upright", "Fallen"};  // modes

        sensor_pos.resize(4, 3);
        sensor_pos << 0.21, 3.11, 2.24,
                7.17, 3.34, 2.16,
                7.55, 0.47, 2.16,
                0.21, 1.26, 2.20;
        imagesize << 1920, 1024;

        this->camera_mat = camera_mat;

        double T = 1.0; // sampling period
        F.resize(9, 9);
        F << 1., T, 0., 0., 0., 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 1., T, 0., 0., 0., 0., 0.,
                0., 0., 0., 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 1., T, 0., 0., 0.,
                0., 0., 0., 0., 0., 1., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 1., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 1.;

        Vector2d n_mu_std0 = lognormal_with_mean_one(0.06);
        Vector2d n_mu_std1 = lognormal_with_mean_one(0.02);
        vector<string> cmc123 = {"CMC1", "CMC2", "CMC3", "WILDTRACK"};
        vector<string> cmc45 = {"CMC4", "CMC5"};
        if (std::find(cmc123.begin(), cmc123.end(), dataset) != cmc123.end()) {
            double sigma_v = 0.035;
            Vector2d B0;
            B0 << T * T / 2.0, T;
            B0 = sigma_v * B0;
            MatrixXd B1 = MatrixXd::Zero(3, 3);
            B1(0, 0) = n_mu_std0(1); // sigma_radius
            B1(1, 1) = n_mu_std0(1); // sigma_radius
            B1(2, 2) = n_mu_std1(1); // sigma_heig
            Matrix3d I3 = Matrix3d::Identity();
            MatrixXd P2(I3.rows() * B0.rows(), I3.cols() * B0.cols());
            P2.setZero();
            for (int i = 0; i < I3.RowsAtCompileTime; i++) { // Kronecker product of two arrays I3*B0
                P2.block(i * B0.rows(), i * B0.cols(), B0.rows(), B0.cols()) = I3(i, i) * B0;
            }
            MatrixXd B = MatrixXd::Zero(9, 6);
            B.topLeftCorner(6, 3) = P2;
            B.bottomRightCorner(3, 3) = B1;
            Q.resize(1);
            Q[0] = B * B.transpose();
            r_birth.resize(1);
            r_birth[0] = 0.004;
            mode.resize(1);
            mode[0] = log(1.0);
            w_birth.resize(1);
            w_birth << log(1.0);
            Vector2d muStd = lognormal_with_mean_one(0.1);
            m_birth.resize(x_dim, 1);
            m_birth.col(0) << 2.3, 0.0, 1.2, 0, 0.825, 0, log(0.3) + muStd(0), log(0.3) + muStd(0), log(0.84) +
                                                                                                    muStd(0);
            VectorXd b(9);
            b << 0.25, 0.1, 0.25, 0.1, 0.15, 0.1, muStd(1), muStd(1), muStd(1);
            MatrixXd diag_b = b.asDiagonal();
            P_birth.resize(1);
            P_birth[0] = diag_b * diag_b.transpose();
            // Markov transition matrix for mode 0 is standing 1 is fall, # probability of survival
            mode_trans_matrix.resize(2, 2);
            mode_trans_matrix << 0.99, 0.01,
                    0.99, 0.01;
            mode_trans_matrix = mode_trans_matrix.array().log();
            n_mu.resize(2, 1);
            n_mu << n_mu_std0(0),
                    n_mu_std1(0);

            // Adaptive birth parameters
            taurU = 0.9; // Only birth measurements with low association probability, e.g. rU > 0.9
            numDet = 2; // discarded solutions with few detections
            rBMin = 1e-5; // cap birth probability
            rBMax = 0.001;
        }
        if (std::find(cmc45.begin(), cmc45.end(), dataset) != cmc45.end()) {
            // transition for standing to standing
            double sigma_v = 0.035;
            Vector2d B0;
            B0 << T * T / 2.0, T;
            B0 = sigma_v * B0;
            MatrixXd B1 = MatrixXd::Zero(3, 3);
            B1(0, 0) = n_mu_std0(1); // sigma_radius
            B1(1, 1) = n_mu_std0(1); // sigma_radius
            B1(2, 2) = n_mu_std1(1); // sigma_heig
            Matrix3d I3 = Matrix3d::Identity();
            MatrixXd P2(I3.rows() * B0.rows(), I3.cols() * B0.cols());
            P2.setZero();
            for (int i = 0; i < I3.RowsAtCompileTime; i++) { // Kronecker product of two arrays I3*B0
                P2.block(i * B0.rows(), i * B0.cols(), B0.rows(), B0.cols()) = I3(i, i) * B0;
            }
            MatrixXd B = MatrixXd::Zero(9, 6);
            B.topLeftCorner(6, 3) = P2;
            B.bottomRightCorner(3, 3) = B1;
            Q.resize(3);
            n_mu.resize(2, 3);
            Q[0] = B * B.transpose();
            n_mu.col(0) << n_mu_std0(0), n_mu_std1(0);

            // transition for falling to falling
            Vector2d muStd0 = lognormal_with_mean_one(0.4);
            Vector2d muStd1 = lognormal_with_mean_one(0.2);
            B1(0, 0) = muStd0(1); // sigma_radius
            B1(1, 1) = muStd0(1); // sigma_radius
            B1(2, 2) = muStd1(1); // sigma_heig
            I3 = Matrix3d::Identity();
            P2.setZero();
            for (int i = 0; i < I3.RowsAtCompileTime; i++) { // Kronecker product of two arrays I3*B0
                P2.block(i * B0.rows(), i * B0.cols(), B0.rows(), B0.cols()) = I3(i, i) * B0;
            }
            B = MatrixXd::Zero(9, 6);
            B.topLeftCorner(6, 3) = P2;
            B.bottomRightCorner(3, 3) = B1;
            Q[1] = B * B.transpose();
            n_mu.col(1) << muStd0(0), muStd1(0);

            // transition from standing to fallen (vice versa)
            sigma_v = 0.07;
            B0 << T * T / 2.0, T;
            B0 = sigma_v * B0;
            Vector2d muStdTemp = lognormal_with_mean_one(0.1);
            B1(0, 0) = muStdTemp(1); // sigma_radius
            B1(1, 1) = muStdTemp(1); // sigma_radius
            B1(2, 2) = muStdTemp(1); // sigma_heig
            I3 = Matrix3d::Identity();
            P2.setZero();
            for (int i = 0; i < I3.RowsAtCompileTime; i++) { // Kronecker product of two arrays I3*B0
                P2.block(i * B0.rows(), i * B0.cols(), B0.rows(), B0.cols()) = I3(i, i) * B0;
            }
            B = MatrixXd::Zero(9, 6);
            B.topLeftCorner(6, 3) = P2;
            B.bottomRightCorner(3, 3) = B1;
            Q[2] = B * B.transpose();
            n_mu.col(2) << muStdTemp(0), 0;

            r_birth.resize(1);
            r_birth[0] = 0.001;
            mode.resize(2);
            mode[0] = log(0.6);
            mode[1] = log(0.4);
            w_birth.resize(2);
            w_birth << log(1.0), log(1.0);
            Vector2d muStd = lognormal_with_mean_one(0.2);
            m_birth.resize(x_dim, 2);
            m_birth.col(0) << 2.3, 0, 1.2, 0, 0.825, 0, log(0.3) + muStd(0), log(0.3) + muStd(0),
                    log(0.84) + muStd(0);
            m_birth.col(1) << 2.3, 0, 1.2, 0, 0.825 / 2, 0, log(0.84) + muStd(0), log(0.84) + muStd[0],
                    log(0.3) + muStd(0);
            VectorXd b(9);
            b << 0.25, 0.1, 0.25, 0.1, 0.15, 0.1, muStd(1), muStd(1), muStd(1);
            MatrixXd diag_b = b.asDiagonal();
            P_birth.resize(2);
            P_birth[0] = diag_b * diag_b.transpose();
            P_birth[1] = diag_b * diag_b.transpose();
            // Markov transition matrix for mode 0 is standing 1 is fall, # probability of survival
            mode_trans_matrix.resize(2, 2);
            mode_trans_matrix << 0.6, 0.4,
                    0.4, 0.6;
            mode_trans_matrix = mode_trans_matrix.array().log();

            // Adaptive birth parameters
            taurU = 0.9; // Only birth measurements with low association probability, e.g. rU > 0.9
            numDet = 2; // discarded solutions with few detections
            rBMin = 1e-5; // cap birth probability
            rBMax = 0.001;
        }

        if (dataset.compare("WILDTRACK") == 0) {
            N_sensors = 7;
            XMAX << -3.0, 9.0;
            YMAX << -8.975, 26.9999479167;

            sensor_pos.resize(N_sensors, 3); // see wildtrack.py, read_intrinsic_extrinsic()
            sensor_pos << 9.0954644, -5.8439021, 2.8889993,
                    -1.1404234, 23.7757403, 1.9939244,
                    9.0098934, 17.7145008, 2.6480724,
                    9.0027305, -5.8456390, 2.7708968,
                    -3.9927943, 9.3807866, 1.6824897,
                    -1.6290514, -10.6383487, 2.2454923,
                    11.7571154, 1.8746221, 3.3953657;

            double sigma_v = 0.15;
            Vector2d B0;
            B0 << T * T / 2.0, T;
            B0 = sigma_v * B0;
            MatrixXd B1 = MatrixXd::Zero(3, 3);
            B1(0, 0) = n_mu_std0(1); // sigma_radius
            B1(1, 1) = n_mu_std0(1); // sigma_radius
            B1(2, 2) = n_mu_std1(1); // sigma_heig
            Matrix3d I3 = Matrix3d::Identity();
            MatrixXd P2(I3.rows() * B0.rows(), I3.cols() * B0.cols());
            P2.setZero();
            for (int i = 0; i < I3.RowsAtCompileTime; i++) { // Kronecker product of two arrays I3*B0
                P2.block(i * B0.rows(), i * B0.cols(), B0.rows(), B0.cols()) = I3(i, i) * B0;
            }
            MatrixXd B = MatrixXd::Zero(9, 6);
            B.topLeftCorner(6, 3) = P2;
            B.bottomRightCorner(3, 3) = B1;
            Q.resize(1);
            Q[0] = B * B.transpose();
            Vector2d muStd = lognormal_with_mean_one(0.1);
            muStd(0);
            VectorXd b(9);
            b << 0.5, 0.55, 0.5, 0.55, 0.15, 0.1, muStd(1), muStd(1), muStd(1);
            MatrixXd diag_b = b.asDiagonal();
            P_birth.resize(1);
            P_birth[0] = diag_b * diag_b.transpose();

            // Adaptive birth parameters
            taurU = 0.9; // Only birth measurements with low association probability, e.g. rU > 0.9
            numDet = 1; // discarded solutions with few detections
            rBMin = 1e-25; // cap birth probability
            rBMax = 0.06;
        }

        // mode 0 (standing)
        Vector2d meas_mu_std0 = lognormal_with_mean_one(0.1);
        Vector2d meas_mu_std1 = lognormal_with_mean_one(0.05);
        VectorXd D_temp(4);
        D_temp << 20, 20, meas_mu_std0(1), meas_mu_std1(1);  // 80 80 wildtrack
        MatrixXd D = D_temp.asDiagonal();
        R.resize(2);
        R[0] = D * D.transpose();
        meas_n_mu.resize(2, 2);
        meas_n_mu.col(0) << meas_mu_std0(0), meas_mu_std1(0);
        // mode 1 (fallen)
        meas_mu_std0 = lognormal_with_mean_one(0.05);
        meas_mu_std1 = lognormal_with_mean_one(0.1);
        D_temp << 20, 20, meas_mu_std0(1), meas_mu_std1(1);
        D = D_temp.asDiagonal();
        R[1] = D * D.transpose();
        meas_n_mu.col(1) << meas_mu_std0(0), meas_mu_std1(0);

        P_S = 0.999999999;
        Q_S = 1 - P_S;
        P_D = 0.97;
        Q_D = 1 - P_D;
        lambda_c = log(5);
        int width = 1920;
        int height = 1024;
        pdf_c = log(1 / ((width - 1) * (height - 1) * log(width - 1) * log(height - 1)));
    };
};

#endif //UKF_TARGET_MODEL_HPP
