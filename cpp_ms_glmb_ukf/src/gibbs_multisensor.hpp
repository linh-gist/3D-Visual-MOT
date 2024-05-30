//
// Created by linh on 2022-03-10.
//
#include <vector>
#include <random>        /*uniform distribution*/
#include <Eigen/Dense>
//#include <iostream>
//#include "lapjv/lapjv_eigen.cpp"

using namespace Eigen;
using namespace std;

// neglogdcost,neglogdprob, neglogcostc,num_comp_request
vector<MatrixXi> gibbs_multisensor_approx_cheap(VectorXd dprob, vector<MatrixXd> costc, int num_samples) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    uniform_real_distribution<double> distribution(0.0, 1.0); // Uniformly distributed random numbers

    int num_tracks = dprob.size();
    int num_sensors = costc.size();

    if (num_samples == 0) {
        vector<MatrixXi> assignments(num_samples, MatrixXi(num_tracks, num_sensors));
        return assignments;
    }

    MatrixXi currsoln = MatrixXi::Zero(num_tracks, num_sensors);
    vector<vector<int>> std_assignments(num_samples, vector<int>(num_tracks * num_sensors));
    MatrixXi::Map(std_assignments[0].data(), currsoln.rows(), currsoln.cols()) = currsoln; //zero, missed detection

    VectorXi m_ones = VectorXi::Ones(num_sensors); // minus one vector
    m_ones *= -1;

    for (int sol = 1; sol < num_samples; sol++) {
        for (int var = 0; var < num_tracks; var++) {
            // cdf = cumsum([dprob(var),1-dprob(var)]); // Bina == 1 is death, Bina == 2 is alive
            // Bina = sum(cdf < rand*cdf(end)) + 1; // if Bina == 2
            if (dprob(var) < distribution(generator)) {// shorten two Matlab lines (above)
                for (int s = 0; s < num_sensors; s++) {
                    VectorXd tempsamp = costc[s].row(var);
                    int cs_rows = costc[s].rows();
                    int cs_cols = costc[s].cols();
                    vector<double> cdf(cs_cols, 0.0); // cumulative sum of "tempsamp"
                    for (int i = 0; i < cs_rows; i++) {
                        if (i == var) { // rule out solution in question
                            continue;
                        }
                        int indx_temp = currsoln(i, s);
                        if (indx_temp > 0) {// rule out missed d term and correct the index for 'tempsamp'
                            tempsamp(indx_temp) = 0;
                        }
                    }
                    cdf[0] = tempsamp[0];
                    for (int i = 1; i < cs_cols; i++) {
                        cdf[i] = cdf[i - 1] + tempsamp[i];
                    }
                    int sum_cdf = 0;
                    double rand = distribution(generator);
                    for (int i = 0; i < cs_cols; i++) {
                        sum_cdf += (int) (cdf[i] < (rand * cdf[cs_cols - 1]));
                    }
                    currsoln(var, s) = sum_cdf;
                }
            } else {
                currsoln.row(var) = m_ones;
            }
        }
        MatrixXi::Map(std_assignments[sol].data(), currsoln.rows(), currsoln.cols()) = currsoln;
    }
    // find unique solutions
    std::sort(std_assignments.begin(), std_assignments.end());
    std_assignments.erase(std::unique(std_assignments.begin(), std_assignments.end()), std_assignments.end());

    vector<MatrixXi> assignments(std_assignments.size(), MatrixXi(num_tracks, num_sensors));
    for (int i = 0; i < assignments.size(); i++) {
        // Add each vector row to the MatrixXd.
        for (std::size_t r = 0; r < num_sensors; r++) {
            assignments[i].col(r) = VectorXi::Map(&std_assignments[i][0] + r * num_tracks, num_tracks);
        }
    }
    return assignments;
}

vector<MatrixXi> multisensor_lapjv(vector<MatrixXd> costc, float match_thresh = 15) {
    int num_tracks = costc[0].rows();
    int num_sensors = costc.size();

    MatrixXi lapjv_assignments = MatrixXi::Ones(num_tracks, num_sensors).array() * -1;
    for (int s = 0; s < num_sensors; s++) {
        // LapJV (minimization), take negative logarithm to find a solution that maximizing likelihood
        if (costc[s].size() != 0) {
            VectorXi x_assignments; // track index
            VectorXi y_assignments = VectorXi::Ones(costc[s].cols()).array() * -1; // measurement index
            lapjv(-(costc[s].array().log()), x_assignments, y_assignments, match_thresh);
            for (int il = 0; il < num_tracks; il++) {
                if (x_assignments(il) >= 0) {
                    lapjv_assignments(il, s) = x_assignments(il);
                }
            }
        }
    }
    return {lapjv_assignments};
}

vector<MatrixXi> gibbs_multisensor_approx_dprobsample(VectorXd dcost, vector<MatrixXd> costc, int num_samples) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    uniform_real_distribution<double> distribution(0.0, 1.0); // Uniformly distributed random numbers

    int num_tracks = dcost.size();
    int num_sensors = costc.size();

    if (num_samples == 0) {
        vector<MatrixXi> assignments(num_samples, MatrixXi(num_tracks, num_sensors));
        return assignments;
    }

    MatrixXi currsoln = MatrixXi::Zero(num_tracks, num_sensors);
    vector<vector<int>> std_assignments(num_samples, vector<int>(num_tracks * num_sensors));
    MatrixXi::Map(std_assignments[0].data(), currsoln.rows(), currsoln.cols()) = currsoln; //zero, missed detection

    VectorXi m_ones = VectorXi::Ones(num_sensors); // minus one vector
    m_ones *= -1;
    VectorXd store_for_deathcost = VectorXd::Ones(num_sensors);

    for (int sol = 1; sol < num_samples; sol++) {
        for (int var = 0; var < num_tracks; var++) {
            // Calculate death probability: dprob = dcost / (dcost + scost)
            for (int s = 0; s < num_sensors; s++) {
                VectorXd tempsamp_p0 = costc[s].row(var);
                for (int i = 0; i < num_tracks; i++) {
                    if (i == var || currsoln(i, s) <= 0) { // rule out solution in question and missed detection
                        continue;
                    }
                    int indx_temp = currsoln(i, s); // restore to measurement index 1-|M|
                    tempsamp_p0(indx_temp) = 0;
                }
                store_for_deathcost(s) = tempsamp_p0.sum();
            }
            double dprob_var = dcost(var) / (dcost(var) + store_for_deathcost.prod());

            // cdf = cumsum([dprob(var),1-dprob(var)]); // Bina == 1 is death, Bina == 2 is alive
            // Bina = sum(cdf < rand*cdf(end)) + 1; // if Bina == 2
            if (dprob_var < distribution(generator)) {// shorten two Matlab lines (above)
                for (int s = 0; s < num_sensors; s++) {
                    VectorXd tempsamp = costc[s].row(var);
                    int cs_rows = costc[s].rows();
                    int cs_cols = costc[s].cols();
                    vector<double> cdf(cs_cols, 0.0); // cumulative sum of "tempsamp"
                    for (int i = 0; i < cs_rows; i++) {
                        if (i == var) { // rule out solution in question
                            continue;
                        }
                        int indx_temp = currsoln(i, s);
                        if (indx_temp > 0) {// rule out missed d term and correct the index for 'tempsamp'
                            tempsamp(indx_temp) = 0;
                        }
                    }
                    cdf[0] = tempsamp[0];
                    for (int i = 1; i < cs_cols; i++) {
                        cdf[i] = cdf[i - 1] + tempsamp[i];
                    }
                    int sum_cdf = 0;
                    double rand = distribution(generator);
                    for (int i = 0; i < cs_cols; i++) {
                        sum_cdf += (int) (cdf[i] < (rand * cdf[cs_cols - 1]));
                    }
                    currsoln(var, s) = sum_cdf;
                }
            } else {
                currsoln.row(var) = m_ones;
            }
        }
        MatrixXi::Map(std_assignments[sol].data(), currsoln.rows(), currsoln.cols()) = currsoln;
    }
    // find unique solutions
    std::sort(std_assignments.begin(), std_assignments.end());
    std_assignments.erase(std::unique(std_assignments.begin(), std_assignments.end()), std_assignments.end());

    vector<MatrixXi> assignments(std_assignments.size(), MatrixXi(num_tracks, num_sensors));
    for (int i = 0; i < assignments.size(); i++) {
        // Add each vector row to the MatrixXd.
        for (std::size_t r = 0; r < num_sensors; r++) {
            assignments[i].col(r) = VectorXi::Map(&std_assignments[i][0] + r * num_tracks, num_tracks);
        }
    }
    return assignments;
}