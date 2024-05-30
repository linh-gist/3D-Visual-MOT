//
// Created by linh on 2022-04-03.
//

#ifndef UKF_TARGET_FILTER_HPP
#define UKF_TARGET_FILTER_HPP

class Filter {
public:
    int H_upd;  // requested number of updated components/hypotheses
    int H_max; // cap on number of posterior components/hypotheses
    double hyp_threshold;  // pruning threshold for components/hypotheses
    int L_max;  // limit on number of Gaussians in each track - not implemented yet
    double elim_threshold;  // pruning threshold for Gaussians in each track - not implemented yet
    double merge_threshold;  // merging threshold for Gaussians in each track - not implemented yet
    double P_G;  // gate size in percentage
    double gamma;  // inv chi^2 dn gamma value
    double gate_flag;  // gating on or off 1/0

    // UKF parameters
    double ukf_alpha;
    double ukf_beta;  // scale parameter for UKF
    double ukf_kappa;

    Filter() {
        H_upd = 20000;  // requested number of updated components/hypotheses
        H_max = 8000; // cap on number of posterior components/hypotheses
        hyp_threshold = log(1e-5);  // pruning threshold for components/hypotheses

        L_max = 100;  // limit on number of Gaussians in each track - not implemented yet
        elim_threshold = 1e-5;  // pruning threshold for Gaussians in each track - not implemented yet
        merge_threshold = 4;  // merging threshold for Gaussians in each track - not implemented yet

        P_G = 0.999999999999;  // gate size in percentage
        gamma = 62.19979206;  // chi2.ppf(self.P_G, mModel.z_dim), inv chi^2 dn gamma value
        gate_flag = 0;  // gating on or off 1/0

        // UKF parameters
        // scale parameter for UKF - choose alpha=1 ensuring lambda=beta and offset of first cov weight is beta
        // for numerical stability
        ukf_alpha = 1;
        ukf_beta = 2;  // scale parameter for UKF
        // scale parameter for UKF (alpha=1 preferred for stability, giving lambda=1, offset of beta
        // for first cov weight)
        ukf_kappa = 2;
    }
};

#endif //UKF_TARGET_FILTER_HPP
