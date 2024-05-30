//
// Created by linh on 2022-04-03.
//

#ifndef UKF_TARGET_MS_GLMB_UKF_HPP
#define UKF_TARGET_MS_GLMB_UKF_HPP

#include <set>
#include <unordered_map>
#include <numeric> // iota
#include <unsupported/Eigen/Polynomials>
#include "Filter.hpp"
#include "Model.hpp"
#include "Target.hpp"
#include "meanShift.hpp"
#include "kmeans.hpp"
//#include "utils.hpp"
//#include "gibbs_multisensor.hpp"
//#include "mc_adaptive_birth.hpp"
//#include "lapjv/lapjv_eigen.cpp"

#define NINF -std::numeric_limits<double>::infinity()
#define INF std::numeric_limits<double>::infinity()
typedef Eigen::Matrix<int64_t, Dynamic, 1> VectorXi64;

// Handle large integer linear index
// Alternative method is that convert each row index to string
// [7,12,0,5,0,3,5,0] => "7,12,0,5,0,3,5,0" representing unique value (consider as an index)
VectorXi64 ndsub2ind64(VectorXi siz, MatrixXi idx32) {
    if (idx32.cols() == 0) {
        return VectorXi64(0);
    } else {
        Matrix<int64_t, Dynamic, Dynamic> idx = idx32.cast<int64_t>();
        VectorXi64 linidx = idx.col(0);
        int64_t cumprod = siz(0);
        for (int i = 1; i < idx.cols(); i++) {
            linidx = linidx + idx.col(i) * cumprod;
            cumprod = (cumprod * siz(i));
        }
        return linidx;
    }
}

VectorXi ndsub2ind(VectorXi siz, MatrixXi idx) {
    if (idx.cols() == 0) {
        return VectorXi(0);
    } else {
        VectorXi linidx = idx.col(0);
        int cumprod = siz(0);
        for (int i = 1; i < idx.cols(); i++) {
            linidx = linidx + idx.col(i) * cumprod;
            cumprod = (cumprod * siz(i));
        }
        return linidx;
    }
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

VectorXd detection_aka_occlusion_model(vector<Target> tindices_tt, Model model, int q, int ini_model_check,
                                       bool occ_model_on = true) {
    VectorXd sensor_pos = model.sensor_pos.row(q);
    MatrixXd cam_mat = model.camera_mat[q];
    int num_of_objects = tindices_tt.size();
    if (num_of_objects == 0) {
        return VectorXd(0);
    }
    MatrixXd X = MatrixXd::Zero(model.x_dim, num_of_objects);
    for (int a = 0; a < tindices_tt.size(); a++) {  // Evaluate MAP of GM
        int ind;
        Target tt = tindices_tt[a];
        tt.mMode.maxCoeff(&ind);
        int start = tt.mGmLen(seq(0, ind - 1)).sum();
        int end = start + tt.mGmLen[ind];
        int indx;
        tt.mW(seq(start, end - 1)).maxCoeff(&indx);
        X.col(a) = tt.mM.col(start + indx);
    }
    X(seq(6, 8), all) = X(seq(6, 8), all).array().exp();
    VectorXd pD_test = VectorXd::Ones(num_of_objects);
    if (!(ini_model_check > 0 && occ_model_on)) {
        return pD_test * model.P_D;
    }

    MatrixXd P_3 = X(seq(0, 4, 2), all);
    VectorXd dist = (P_3.colwise() - sensor_pos).colwise().norm();
    vector<int> indx(dist.size());
    std::iota(indx.begin(), indx.end(), 0);
    stable_sort(indx.begin(), indx.end(), [&dist](int i1, int i2) { return dist[i1] < dist[i2]; });
    VectorXi check_pool = VectorXi::Ones(num_of_objects);  // give every object a flag
    for (int j = 0; j < num_of_objects; j++) {  // start looping through objects in the hypothesis
        if (!check_pool[j]) {  // check which object has not been evaluated
            continue;
        }
        VectorXd temp = VectorXd::Ones(P_3.rows() + 1);
        temp(seq(0, P_3.rows() - 1)) = P_3.col(indx[j]);
        temp = cam_mat * temp;
        VectorXd test_point_Img_plane = temp(seq(0, 1)) / temp(2);
        bool f1 = test_point_Img_plane[0] <= 0;
        bool f2 = test_point_Img_plane[0] >= model.imagesize[0];
        bool f3 = test_point_Img_plane[1] <= 0;
        bool f4 = test_point_Img_plane[1] >= model.imagesize[1];
        if (f1 || f2 || f3 || f4) {
            pD_test[indx[j]] = model.Q_D;  // if object is not in the image, then assign low pd.
        } else {
            pD_test[indx[j]] = model.P_D;  // object is in the image. high pd.
        }
        check_pool[j] = 0;  // unchecked the object when pd has been assigned
        VectorXd curr_object = X.col(indx[j]);
        VectorXd curr_object_centroid = curr_object(seq(0, 4, 2));
        double curr_object_rx = curr_object[6];
        double curr_object_ry = curr_object[7];
        double curr_object_h = curr_object[8];
        /*
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
        */
        MatrixXd A(3, 3);
        A << 1 / pow(curr_object_rx, 2), 0, 0,
                0, 1 / pow(curr_object_ry, 2), 0,
                0, 0, 1 / pow(curr_object_h, 2);
        // calculating b and c
        EigenSolver<MatrixXd> es(A);
        VectorXd eig_vals_vec = es.pseudoEigenvalueMatrix().diagonal();
        MatrixXd right_eig = es.pseudoEigenvectors(); // column vectors
        VectorXd temp_ellip_c = right_eig * curr_object_centroid;
        VectorXd ggs = -2 * temp_ellip_c.array() * eig_vals_vec.array();
        VectorXd b_desired = ggs.transpose() * right_eig;  // this is b
        double J_const = (ggs.array().square() / (4 * eig_vals_vec.array())).sum();
        for (int i = 0; i < check_pool.size(); i++) {
            if (check_pool[i] == 0) {
                continue;
            }
            VectorXd evaluate = P_3.col(indx[i]);
            VectorXd vector_evaluate = evaluate - sensor_pos;
            vector_evaluate = vector_evaluate / vector_evaluate.norm();
            double alpha = (vector_evaluate.transpose() * A) * vector_evaluate;
            double beta1 = (vector_evaluate.transpose() * A) * sensor_pos;
            double beta = (b_desired.transpose() * vector_evaluate) + 2 * beta1;
            double gamma1 = (b_desired.transpose() * sensor_pos);
            double gamma = ((sensor_pos.transpose() * A) * sensor_pos) + gamma1 + (-1 + J_const);
            VectorXd coeff(3); // coeff is in reverse order compared to Matlab, Numpy function "roots"
            coeff << gamma, beta, alpha; // p[n] * x**n + p[n-1] * x**(n-1) + ... + p[1]*x + p[0]
            Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
            solver.compute(coeff);
            bool check_real = true;
            const Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType &r = solver.roots();
            for (std::complex<double> cr: r) {
                if (cr.imag() != 0) {
                    check_real = false;
                    break;
                }
            }
            if (check_real) {
                check_pool[i] = 0;
                pD_test[indx[i]] = model.Q_D;
            }
        }
    }
    if (check_pool.sum() > 0) {
        cout << "Not all states are checked - P_d are not assigned entirely!!!";
    }
    return pD_test;
}


VectorXd detection_aka_occlusion_model_v2(vector<Target> tindices_tt, Model model, int q, int ini_model_check,
                                          bool occ_model_on = true) {
    // project target state => compute Intersection Over Area (back2front)
    int num_of_objects = tindices_tt.size();
    if (num_of_objects == 0) {
        return VectorXd(0);
    }
    MatrixXd X = MatrixXd::Zero(model.x_dim, num_of_objects);
    for (int a = 0; a < tindices_tt.size(); a++) {  // Evaluate MAP of GM
        int ind;
        Target tt = tindices_tt[a];
        tt.mMode.maxCoeff(&ind);
        int start = tt.mGmLen(seq(0, ind - 1)).sum();
        int end = start + tt.mGmLen[ind];
        int indx;
        tt.mW(seq(start, end - 1)).maxCoeff(&indx);
        X.col(a) = tt.mM.col(start + indx);
    }
    VectorXd pD_test = VectorXd::Ones(num_of_objects);
    if (!(ini_model_check > 0 && occ_model_on)) {
        return pD_test * model.P_D;
    }
    MatrixXd noise = MatrixXd::Zero(4, X.cols());
    MatrixXd x2bbox = gen_msobservation_fn_v2(model.camera_mat[q], X, noise, VectorXd::Zero(2));
    MatrixXd ioa_temp = bboxes_ioi_xyah_back2front_all(x2bbox.transpose());
    VectorXd ioa_max = 1 - ioa_temp.rowwise().maxCoeff().array(); //bboxes_ioi_xyah_back2front_all_v2(x2bbox.transpose());
    pD_test = ioa_max.array().min(model.P_D).max(1 - model.P_D);

    return pD_test;
}

class MSGLMB {
private:
    vector<Target> glmb_update_tt;   // (1) track table for GLMB (individual tracks)
    VectorXd glmb_update_w;          // (2) vector of GLMB component/hypothesis weights
    vector<VectorXi> glmb_update_I;  // (3) cell of GLMB component/hypothesis labels (in track table)
    VectorXi glmb_update_n;          // (4) vector of GLMB component/hypothesis cardinalities
    VectorXd glmb_update_cdn;        // (5) cardinality distribution of GLMB
    vector<Target> tt_birth;

    Filter filter;
    Model mModel;
    MCAdaptiveBirth mMCAB;

    int mId;
    int mFeatDim = 128;
    bool mUseFeat;
    int mAdaptiveBirth;
    vector<Target> mPrunedTargets;//
    vector<MatrixXd> mPrunedTargetsFeat;
    vector<int> mPrunedTargetsLabel;
    vector<Target> mPrevGlmbUpdateTt;//
    set<int> mPreviousTargetLabel;

    void msjointpredictupdate(Model model, Filter filter, vector<MatrixXd> measZ, int k) {
        // create birth tracks
        if (k == 0) {
            vector<VectorXd> meas_ru;
            vector<MatrixXd> meas_Z;
            bool set_feat_dim = true;
            for (MatrixXd bboxZ: measZ) {
                if (bboxZ.cols() > 0 && set_feat_dim) {
                    mFeatDim = bboxZ.rows() - 5;  // 4 + 1(bbox[ltwh] + conf)
                    set_feat_dim = false;
                }
                MatrixXd bbox = bboxZ(seq(0, 3), all);
                VectorXd temp = 0.99 * VectorXd::Ones(bbox.cols() + 1);
                temp[0] = 1;  // for miss - detection
                meas_ru.push_back(temp);
                meas_Z.push_back(bbox);
            }
            generate_birth(meas_ru, meas_Z, measZ, k);
        }
        // create surviving tracks - via time prediction (single target CK)
        for (Target &tt: glmb_update_tt) {
            tt.predict();
        }
        // create predicted tracks - concatenation of birth and survival
        vector<Target> glmb_predict_tt(tt_birth);  // copy track table back to GLMB struct
        glmb_predict_tt.insert(glmb_predict_tt.end(), glmb_update_tt.begin(), glmb_update_tt.end());
        // gating by tracks
        if (filter.gate_flag) {
            for (Target &tt: glmb_predict_tt) {
                //tt.gate_msmeas_ukf(filter.gamma, meas)
                tt.gating_feet(measZ);
            }
        } else {
            for (Target &tt: glmb_predict_tt) {
                tt.not_gating(measZ);
            }
        }
        // precalculation loop for average survival/death probabilities
        int cpreds = glmb_predict_tt.size();
        VectorXd avps(cpreds);
        for (int tabidx = 0; tabidx < tt_birth.size(); tabidx++) {
            avps(tabidx) = glmb_predict_tt[tabidx].mR;
        }
        for (int tabidx = tt_birth.size(); tabidx < cpreds; tabidx++) {
            avps(tabidx) = glmb_predict_tt[tabidx].computePS(k);
        }
        VectorXd avqs = (1 - avps.array()).log();
        avps = avps.array().log();

        // create updated tracks (single target Bayes update)
        // nested for loop over all predicted tracks and sensors - slow way
        // Kalman updates on the same prior recalculate all quantities
        VectorXi m(model.N_sensors);
        vector<MatrixXd> allcostc(model.N_sensors);
        // extra state for not detected for each sensor (meas "0" in pos 1)
        vector<MatrixXd> jointcostc(model.N_sensors);
        // posterior probability of target survival (after measurement updates)
        MatrixXd avpp = MatrixXd::Zero(cpreds, model.N_sensors);
        // gated measurement index matrix
        vector<MatrixXi> gatemeasidxs(model.N_sensors);
        for (int s = 0; s < model.N_sensors; s++) {
            m[s] = measZ[s].cols();  // number of measurements
            // allcostc[s] # extra state for not detected for each sensor (meas "0" in pos 1)
            // jointcostc[s] # extra state for not detected for each sensor (meas "0" in pos 1)
            allcostc[s] = NINF * MatrixXd::Ones(cpreds, 1 + m(s));
            jointcostc[s] = NINF * MatrixXd::Ones(cpreds, 1 + m(s));

            gatemeasidxs[s] = MatrixXi::Ones(cpreds, m(s));
            gatemeasidxs[s] *= -1;

            for (int tabidx = 0; tabidx < cpreds; tabidx++) {
                Target tt = glmb_predict_tt[tabidx];
                for (int emm = 0; emm < m(s); emm++) {
                    if (tt.mGateMeas[s](emm) >= 0) {
                        // unnormalized updated weights
                        double w_temp = tt.ukf_likelihood_per_sensor(measZ[s].col(emm), s);
                        // predictive likelihood
                        allcostc[s](tabidx, 1 + emm) = w_temp;// + std::nexttoward(0.0, 1.0L);
                        gatemeasidxs[s](tabidx, tt.mGateMeas[s](emm)) = tt.mGateMeas[s](emm);
                    }
                }
                jointcostc[s] = (allcostc[s].colwise() + avps).array() - (model.lambda_c + model.pdf_c);
                jointcostc[s].col(0) = avps;
            }
        }

        // component updates
        int runidx = 0;
        VectorXd glmb_nextupdate_w(filter.H_upd * 2);
        vector<VectorXi> glmb_nextupdate_I;
        VectorXi glmb_nextupdate_n = VectorXi::Zero(filter.H_upd * 2);
        int nbirths = tt_birth.size();
        double temp = log_sum_exp(0.5 * glmb_update_w);
        VectorXd hypoth_numd = (log(filter.H_upd) + 0.5 * glmb_update_w.array() - temp).exp();
        VectorXi hypoth_num = hypoth_numd.array().round().cast<int>();
        // use to normalize assign_prob using glmb_nextupdate_w, first raw for "measurement" missed detection
        vector<MatrixXd> assign_meas(model.N_sensors);
        for (int s = 0; s < model.N_sensors; s++) {
            assign_meas[s] = MatrixXd::Zero(m[s] + 1, filter.H_upd * 2);
        }

        vector<int> tt_update_parent;
        MatrixXi tt_update_currah(0, model.N_sensors);
        vector<int64_t> tt_update_linidx; //int
        for (int pidx = 0; pidx < glmb_update_w.size(); pidx++) {
            // calculate best updated hypotheses/components
            int nexists = glmb_update_I[pidx].size();
            int ntracks = nbirths + nexists;
            // indices of all births and existing tracks  for current component
            VectorXi tindices(ntracks);
            tindices << VectorXi::LinSpaced(nbirths, 0, nbirths - 1), glmb_update_I[pidx].array() + nbirths;
            vector<Target> tindices_tt(ntracks);
            for (int i = 0; i < ntracks; i++) {
                tindices_tt[i] = glmb_predict_tt[tindices[i]];
            }
            vector<VectorXi> mindices(model.N_sensors); // mindices, sorted -1, 0, 1, 2
            MatrixXd avpd(tindices.size(), model.N_sensors);
            for (int s = 0; s < model.N_sensors; s++) {
                // union indices of gated measurements for corresponding tracks
                MatrixXi gate_tindices = gatemeasidxs[s](tindices, all);
                std::set<int> mindices_set{gate_tindices.data(), gate_tindices.data() + gate_tindices.size()};
                std::vector<int> temp1(mindices_set.begin(), mindices_set.end()); // convert set to vector std
                VectorXi temp2 = VectorXi::Map(temp1.data(), temp1.size()); // convert vector std to eigen
                if (temp2.size() > 0 && temp2(0) == -1) {
                    mindices[s] = VectorXi::Zero(temp2.size());
                    mindices[s](seq(1, temp2.size() - 1)) = 1 + temp2(seq(1, temp2.size() - 1)).array();
                } else {
                    mindices[s] = VectorXi::Zero(temp2.size() + 1);
                    mindices[s](seq(1, temp2.size())) = 1 + temp2.array();
                }
                avpd.col(s) = detection_aka_occlusion_model_v2(tindices_tt, model, s, hypoth_num[pidx]);
            }
            MatrixXd avqd = (1 - avpd.array()).log();
            avpd = avpd.array().log();
            vector<MatrixXd> costc(model.N_sensors);
            MatrixXd avpp = MatrixXd::Zero(tindices.size(), model.N_sensors);
            for (int s = 0; s < model.N_sensors; s++) {
                MatrixXd take_rows = jointcostc[s](tindices, all);
                MatrixXd jointcostc_pidx = take_rows(all, mindices[s]);
                jointcostc_pidx.col(0) = avqd.col(s) + jointcostc_pidx.col(0);
                int endTmp = jointcostc_pidx.cols() - 1;
                jointcostc_pidx(all, seq(1, endTmp)) = jointcostc_pidx(all, seq(1, endTmp)).colwise() + avpd.col(s);
                for (int i = 0; i < jointcostc_pidx.rows(); i++) {
                    avpp(i, s) = log_sum_exp(jointcostc_pidx.row(i));
                }
                costc[s] = jointcostc_pidx.array().exp();
            }
            VectorXd edcost = avqs(tindices).array().exp(); // death cost
            vector<MatrixXi> uasses = gibbs_multisensor_approx_dprobsample(edcost, costc, hypoth_num[pidx]);
            // vector<MatrixXi> uasses = multisensor_lapjv(costc); // single hypothesis

            MatrixXd local_avpdm = avpd;
            MatrixXd local_avqdm = avqd;
            MatrixXi aug_idx(tindices.size(), model.N_sensors + 1);
            aug_idx.col(0) = tindices;
            VectorXi sizeIdx(m.size() + 1);
            sizeIdx(0) = cpreds;
            sizeIdx(seq(1, m.size())) = m.array() + 1;
            for (int hidx = 0; hidx < uasses.size(); hidx++) {
                MatrixXi update_hypcmp_tmp = uasses[hidx]; // ntracks x N_sensors
                vector<int> off_vec, not_off_vec;
                double stemp = (model.lambda_c + model.pdf_c) * m.array().sum();
                for (int ivec = 0; ivec < update_hypcmp_tmp.rows(); ivec++) {
                    if (update_hypcmp_tmp(ivec, 0) < 0) { // check death target in only one sensor
                        off_vec.push_back(ivec);
                        stemp += avqs(tindices(ivec));
                    } else if (update_hypcmp_tmp(ivec, 0) == 0) {
                        not_off_vec.push_back(ivec);
                        stemp += avps(tindices(ivec));
                    } else { // >0
                        not_off_vec.push_back(ivec);
                        stemp += avps(tindices(ivec));
                    }
                    for (int s = 0; s < update_hypcmp_tmp.cols(); s++) {
                        int mIndex = update_hypcmp_tmp(ivec, s); // measurement index
                        if (mIndex > 0) {
                            stemp += local_avpdm(ivec, s) - (model.lambda_c + model.pdf_c);
                            //Get measurement index from uasses
                            //Setting index of measurements associate with a track, 0: missed, 1-|Z| measurement index
                            assign_meas[s](mIndex, runidx) = 1;
                        }
                        if (mIndex == 0) {
                            stemp += local_avqdm(ivec, s);
                            assign_meas[s](mIndex, runidx) = 1;
                        }
                    }
                }
                aug_idx(all, seq(1, model.N_sensors)) = update_hypcmp_tmp;
                VectorXi off_idx = VectorXi::Map(off_vec.data(), off_vec.size());
                VectorXi not_offidx = VectorXi::Map(not_off_vec.data(), not_off_vec.size());
                VectorXi64 update_hypcmp_idx = VectorXi64::Zero(update_hypcmp_tmp.rows()); // VectorXi
                update_hypcmp_idx(off_idx).array() = -1;
                update_hypcmp_idx(not_offidx) = ndsub2ind64(sizeIdx, aug_idx(not_offidx, all)); // ndsub2ind
                int num_trk = (update_hypcmp_idx.array() >= 0).cast<int>().sum();
                //
                glmb_nextupdate_w(runidx) = stemp + glmb_update_w(pidx); // hypothesis/component weight

                if (num_trk > 0) {
                    // hypothesis/component tracks (via indices to track table)
                    int parent_size = tt_update_parent.size();
                    VectorXi range_temp = VectorXi::LinSpaced(num_trk, parent_size, parent_size + num_trk - 1);
                    glmb_nextupdate_I.push_back(range_temp);
                } else {
                    glmb_nextupdate_I.push_back(VectorXi(0));
                }
                glmb_nextupdate_n[runidx] = num_trk; // hypothesis / component cardinality
                for (int tt_idx = 0; tt_idx < not_offidx.size(); tt_idx++) {
                    tt_update_parent.push_back(tindices(not_offidx(tt_idx)));
                    tt_update_linidx.push_back(update_hypcmp_idx(not_offidx(tt_idx)));
                }
                int currah = tt_update_currah.rows(); // Resizes the matrix, while leaving old values untouched.
                tt_update_currah.conservativeResize(currah + not_offidx.size(), NoChange);
                tt_update_currah(seq(currah, currah + not_offidx.size() - 1), all) = update_hypcmp_tmp(not_offidx, all);
                runidx = runidx + 1;
            }
        }

        // component updates via posterior weight correction (including generation of track table)
        unordered_map<int, int> umap; // sorted, order
        int unique_idx = 0;
        vector<double> tt_update_msqz;
        vector<Target> tt_update;
        VectorXi ttU_newidx(tt_update_linidx.size());
        for (int tt_idx = 0; tt_idx < tt_update_linidx.size(); tt_idx++) {
            if (umap.find(tt_update_linidx[tt_idx]) == umap.end()) {
                umap[tt_update_linidx[tt_idx]] = unique_idx;

                int preidx = tt_update_parent[tt_idx];
                VectorXi meascomb = tt_update_currah.row(tt_idx);
                // kalman update for this track and all joint measurements
                double qz_temp;
                Target tt;
                std::tie(qz_temp, tt) = glmb_predict_tt[preidx].ukf_update(measZ, meascomb, k);
                tt_update_msqz.push_back(qz_temp);
                tt_update.push_back(tt);
                ttU_newidx(tt_idx) = unique_idx;

                unique_idx += 1;
            } else {
                ttU_newidx(tt_idx) = umap[tt_update_linidx[tt_idx]];
            }
        }
        VectorXd msqz = VectorXd::Map(tt_update_msqz.data(), tt_update_msqz.size());
        // normalize weights
        VectorXd glmb_w_runidx = glmb_nextupdate_w(seq(0, runidx - 1));
        for (int pidx = 0; pidx < runidx; pidx++) {
            glmb_nextupdate_I[pidx] = ttU_newidx(glmb_nextupdate_I[pidx]);
            glmb_w_runidx[pidx] = glmb_w_runidx[pidx] + msqz(glmb_nextupdate_I[pidx]).array().sum();
        }
        glmb_update_w = glmb_w_runidx.array() - log_sum_exp(glmb_w_runidx); // 2

        // Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking
        vector<VectorXd> meas_ru(model.N_sensors);
        vector<MatrixXd> meas_Z(model.N_sensors);
        VectorXd exp_glmb_w = glmb_update_w.array().exp();
        for (int s = 0; s < model.N_sensors; s++) {
            // adaptive birth weight for each measurement
            VectorXd rA = assign_meas[s](all, seq(0, runidx - 1)) * exp_glmb_w;
            rA = rA.array().min(1 - std::nexttoward(0.0, 1.0L)).max(std::nexttoward(0.0, 1.0L));
            // By notation, let rA,+(0) = 0 which is intuitive as it suggests that a missed detection
            // did not associate with any tracks in the existing hypotheses.
            rA[0] = 0;
            MatrixXd bbox = measZ[s](seq(0, 3), all);
            meas_Z[s] = bbox;
            meas_ru[s] = 1 - rA.array();  // rU
        }
        generate_birth(meas_ru, meas_Z, measZ, k);

        // extract cardinality distribution
        glmb_update_n = glmb_nextupdate_n(seq(0, runidx - 1)); // 4
        VectorXd glmb_nextupdate_cdn = NINF * VectorXd::Ones(glmb_update_n.maxCoeff() + 1);
        for (int card = 0; card < glmb_nextupdate_cdn.size(); card++) {
            // extract probability of n targets
            VectorXi card_check = (glmb_update_n.array() == card).cast<int>();
            if (card_check.sum()) {
                VectorXd vlog_card = card_check.select(glmb_update_w, NINF);
                glmb_nextupdate_cdn[card] = log_sum_exp(vlog_card);
            }
        }
        glmb_nextupdate_cdn = glmb_nextupdate_cdn.array().exp();
        // copy glmb update to the next time step
        glmb_update_tt = tt_update;             // 1
        glmb_update_I = glmb_nextupdate_I;      // 3
        glmb_update_cdn = glmb_nextupdate_cdn;  // 5
        // remove duplicate entries and clean track table
        clean_predict();
        clean_update(k);
        for (Target &tt: glmb_update_tt) {  // pruning, merging, capping Gaussian mixture components
            tt.cleanupTarget();
        }
    }

    tuple<vector<MatrixXd>, vector<double>, vector<vector<MatrixXd>>, vector<vector<int>>>
    mc_adaptive_birth(vector<VectorXd> meas_ru, vector<MatrixXd> meas_z) {
        // Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking
        vector<VectorXd> m_birth;
        vector<double> r_birth;
        vector<vector<int>> sols_birth;
        vector<MatrixXd> P_b;
        tie(m_birth, r_birth, sols_birth) = mMCAB.sample_adaptive_birth(meas_ru, meas_z);
        vector<MatrixXd> m_b_final;
        vector<vector<MatrixXd>> P_b_final;
        for (int idx = 0; idx < r_birth.size(); idx++) {
            MatrixXd m_mode = mModel.m_birth;
            vector<MatrixXd> P_mode = mModel.P_birth;
            for (int imode = 0; imode < mModel.mode.size(); imode++) {
                if (imode == 0) {
                    m_mode.col(imode) = m_birth[idx];
                } else {  // imode=1
                    m_mode(seq(0, 2, 2), imode) = m_birth[idx](seq(0, 2, 2));
                }
            }
            m_b_final.emplace_back(m_mode);
            P_b_final.emplace_back(P_mode);
        }
        return {m_b_final, r_birth, P_b_final, sols_birth};
    }

    tuple<vector<MatrixXd>, vector<double>, vector<vector<MatrixXd>>, vector<vector<int>>> mc_adaptive_birth_efficient(
            vector<VectorXd> meas_ru, vector<MatrixXd> meas_z) {
        // Only birth measurements with low association probability, e.g. rU > 0.9
        vector<VectorXd> meas_ru_keep;
        vector<MatrixXd> meas_z_keep;
        int num_meas = 0;
        for (int sdx = 0; sdx < meas_ru.size(); sdx++) {
            vector<double> meas_ru_s;
            vector<VectorXd> meas_z_s;
            for (int j = 0; j < meas_ru[sdx].size(); j++) {
                if (meas_ru[sdx](j) > mModel.taurU) {
                    meas_ru_s.emplace_back(meas_ru[sdx](j));
                    if (j > 0) {
                        meas_z_s.emplace_back(meas_z[sdx].col(j - 1));
                    }
                    num_meas += 1;
                }
            }
            meas_ru_keep.emplace_back(Eigen::Map<Eigen::VectorXd>(meas_ru_s.data(), meas_ru_s.size()));
            MatrixXd zMat(4, meas_z_s.size());
            for (int j = 0; j < meas_z_s.size(); j++) {
                zMat.col(j) = meas_z_s[j];
            }
            meas_z_keep.emplace_back(zMat);
        }
        meas_ru = meas_ru_keep;
        meas_z = meas_z_keep;
        if (num_meas <= mModel.N_sensors) {  // No measurement
            return {vector<MatrixXd>(0), vector<double>(0), vector<vector<MatrixXd>>(0), vector<vector<int>>(0)};
        }
        vector<vector<int>> sols;
        vector<vector<float>> centroids;
        tie(sols, centroids) = mMCAB.sample_mc_sols(meas_ru, meas_z);
        vector<double> r_b;
        vector<MatrixXd> m_b;
        vector<vector<MatrixXd>> P_b;
        vector<vector<int>> solsKeep;
        for (int idx = 0; idx < sols.size(); idx++) {
            int solSum = 0;
            for (int j = 0; j < sols[idx].size(); j++) {
                if (sols[idx][j] > 0) {
                    solSum += 1;
                }
            }
            if (solSum <= mModel.numDet) {
                continue; // discarded solutions with few detections
            }
            VectorXd m_temp = mModel.m_birth.col(0);
            m_temp(0) = centroids[idx][0]; // x
            m_temp(2) = centroids[idx][1]; // y
            MatrixXd Ptemp = mModel.P_birth[0];
            VectorXd q_z = VectorXd::Zero(mModel.N_sensors);
            vector<int> sol = sols[idx];
            for (int q = 0; q < sol.size(); q++) {
                int jdx = sol[q] - 1;  // restore original measurement index 0-|Z|
                if (jdx < 0) {
                    continue;
                }
                double qt;
                VectorXd mt;
                MatrixXd Pt;
                tie(qt, mt, Pt) = ukf_update_per_sensor(meas_z[q].col(jdx), m_temp, Ptemp, q, 0, mModel);
                q_z(q) = qt + log(meas_ru[q](jdx + 1));
                m_temp = mt;
                Ptemp = Pt;
            }
            r_b.emplace_back(q_z.sum());
            MatrixXd m_mode = mModel.m_birth;
            vector<MatrixXd> P_mode = mModel.P_birth;
            for (int imode = 0; imode < mModel.mode.size(); imode++) {
                if (imode == 0) {
                    m_mode.col(imode) = m_temp;
                } else {  // imode=1
                    m_mode(seq(0, 2, 2), imode) = m_temp(seq(0, 2, 2));
                }
                P_mode[imode] = Ptemp;
            }
            m_b.emplace_back(m_mode);
            P_b.emplace_back(P_mode);
            solsKeep.emplace_back(sols[idx]);
        }
        vector<MatrixXd> m_b_final;
        vector<vector<MatrixXd>> P_b_final;
        vector<vector<int>> sols_final;
        vector<double> v_rB;
        if (r_b.size() > 0) {
            VectorXd rBTemp = VectorXd::Map(r_b.data(), r_b.size());
            rBTemp = (rBTemp.array() - log_sum_exp(rBTemp)).exp();
            for (int i = 0; i < rBTemp.size(); i++) { // prune low weight birth
                if (rBTemp(i) > mModel.rBMin) {
                    m_b_final.emplace_back(m_b[i]);
                    P_b_final.emplace_back(P_b[i]);
                    sols_final.emplace_back(solsKeep[i]);
                    v_rB.emplace_back(std::min(rBTemp(i), mModel.rBMax));
                }
            }
        }
        return {m_b_final, v_rB, P_b_final, sols_final};
    }

    tuple<vector<MatrixXd>, vector<double>, vector<vector<MatrixXd>>, vector<vector<int>>> kmeans_adaptive_birth(
            vector<VectorXd> meas_ru, vector<MatrixXd> meas_z) {
        // Only birth measurements with low association probability, e.g. rU > 0.9
        vector<VectorXd> meas_ru_keep;
        vector<MatrixXd> meas_z_keep;
        int num_meas = 0;
        for (int sdx = 0; sdx < meas_ru.size(); sdx++) {
            vector<double> meas_ru_s;
            vector<VectorXd> meas_z_s;
            for (int j = 0; j < meas_ru[sdx].size(); j++) {
                if (meas_ru[sdx](j) > mModel.taurU) {
                    meas_ru_s.emplace_back(meas_ru[sdx](j));
                    if (j > 0) {
                        meas_z_s.emplace_back(meas_z[sdx].col(j - 1));
                    }
                    num_meas += 1;
                }
            }
            meas_ru_keep.emplace_back(Eigen::Map<Eigen::VectorXd>(meas_ru_s.data(), meas_ru_s.size()));
            MatrixXd zMat(4, meas_z_s.size());
            for (int j = 0; j < meas_z_s.size(); j++) {
                zMat.col(j) = meas_z_s[j];
            }
            meas_z_keep.emplace_back(zMat);
        }
        meas_ru = meas_ru_keep;
        meas_z = meas_z_keep;
        if (num_meas <= mModel.N_sensors) {  // No measurement
            return {vector<MatrixXd>(0), vector<double>(0), vector<vector<MatrixXd>>(0), vector<vector<int>>(0)};
        }
        // idxs ~ [[sensor index, measurement index, cluster index]]
        vector<VectorXi> idxs;
        vector<VectorXd> clusters;
        tie(idxs, clusters) = KMeans(1000).run(mModel.camera_mat, meas_z);
        vector<double> r_b;
        vector<MatrixXd> m_b;
        vector<vector<MatrixXd>> P_b;
        vector<vector<int>> sols;
        double lambda_b = 4; // The Labeled Multi-Bernoulli Filter, eq (75)
        for (int idx = 0; idx < clusters.size(); idx++) {
            VectorXd cluster = clusters[idx];
            vector<int> sol(mModel.N_sensors, 0);
            VectorXd rB_temp = VectorXd::Zero(mModel.N_sensors);
            vector<double> rU_temp(mModel.N_sensors, 0);
            // idx_map ~ [sensor index, measurement index, cluster index]
            for (int jdx = 0; jdx < idxs.size(); jdx++) {
                if (idxs[jdx](2) != idx) {
                    continue;
                }
                VectorXi idx_map = idxs[jdx];
                int s = idx_map(0);
                int meas_idx = idx_map[1] + 1;
                if (rU_temp[s] < meas_ru_keep[s](meas_idx)) {  // if same sensor, replace with higher rU
                    rU_temp[s] = meas_ru_keep[s](meas_idx);
                    double not_assigned_sum = meas_ru_keep[s].array().sum() + std::nexttoward(0.0, 1.0L);
                    rB_temp(s) = min(mModel.rBMax, (meas_ru_keep[s](meas_idx)) / not_assigned_sum * lambda_b);
                    sol[s] = meas_idx;
                }
            }
            MatrixXd m_mode = mModel.m_birth;
            vector<MatrixXd> P_mode = mModel.P_birth;
            for (int imode = 0; imode < mModel.mode.size(); imode++) {
                m_mode(seq(0, 2, 2), imode) = cluster;
            }
            r_b.emplace_back(rB_temp.maxCoeff());
            m_b.emplace_back(m_mode);
            P_b.emplace_back(P_mode);
            sols.emplace_back(sol);
        }
        return {m_b, r_b, P_b, sols};
    }

    tuple<vector<MatrixXd>, vector<double>, vector<vector<MatrixXd>>, vector<vector<int>>> meanshift_adaptive_birth(
            vector<VectorXd> meas_ru, vector<MatrixXd> meas_z) {
        // Only birth measurements with low association probability, e.g. rU > 0.9
        vector<VectorXd> meas_ru_keep;
        vector<MatrixXd> meas_z_keep;
        int num_meas = 0;
        for (int sdx = 0; sdx < meas_ru.size(); sdx++) {
            vector<double> meas_ru_s;
            vector<VectorXd> meas_z_s;
            for (int j = 0; j < meas_ru[sdx].size(); j++) {
                if (meas_ru[sdx](j) > mModel.taurU) {
                    meas_ru_s.emplace_back(meas_ru[sdx](j));
                    if (j > 0) {
                        meas_z_s.emplace_back(meas_z[sdx].col(j - 1));
                    }
                    num_meas += 1;
                }
            }
            meas_ru_keep.emplace_back(Eigen::Map<Eigen::VectorXd>(meas_ru_s.data(), meas_ru_s.size()));
            MatrixXd zMat(4, meas_z_s.size());
            for (int j = 0; j < meas_z_s.size(); j++) {
                zMat.col(j) = meas_z_s[j];
            }
            meas_z_keep.emplace_back(zMat);
        }
        meas_ru = meas_ru_keep;
        meas_z = meas_z_keep;
        if (num_meas <= mModel.N_sensors) {  // No measurement
            return {vector<MatrixXd>(0), vector<double>(0), vector<vector<MatrixXd>>(0), vector<vector<int>>(0)};
        }
        vector<vector<int>> sols;
        vector<vector<float>> centroids;
        tie(sols, centroids) = meanShift(mModel.camera_mat, meas_z, 0.6);
        vector<double> r_b;
        vector<MatrixXd> m_b;
        vector<vector<MatrixXd>> P_b;
        vector<vector<int>> solsKeep;
        for (int idx = 0; idx < sols.size(); idx++) {
            int solSum = 0;
            for (int j = 0; j < sols[idx].size(); j++) {
                if (sols[idx][j] > 0) {
                    solSum += 1;
                }
            }
            if (solSum <= mModel.numDet) {
                continue; // discarded solutions with few detections
            }
            VectorXd m_temp = mModel.m_birth.col(0);
            m_temp(0) = centroids[idx][0]; // x
            m_temp(2) = centroids[idx][1]; // y
            MatrixXd Ptemp = mModel.P_birth[0];
            VectorXd q_z = VectorXd::Zero(mModel.N_sensors);
            vector<int> sol = sols[idx];
            for (int q = 0; q < sol.size(); q++) {
                int jdx = sol[q] - 1;  // restore original measurement index 0-|Z|
                if (jdx < 0) {
                    continue;
                }
                double qt;
                VectorXd mt;
                MatrixXd Pt;
                tie(qt, mt, Pt) = ukf_update_per_sensor(meas_z[q].col(jdx), m_temp, Ptemp, q, 0, mModel);
                q_z(q) = qt + log(meas_ru[q](jdx + 1));
                m_temp = mt;
                Ptemp = Pt;
            }
            r_b.emplace_back(q_z.sum());
            MatrixXd m_mode = mModel.m_birth;
            vector<MatrixXd> P_mode = mModel.P_birth;
            for (int imode = 0; imode < mModel.mode.size(); imode++) {
                if (imode == 0) {
                    m_mode.col(imode) = m_temp;
                } else {  // imode=1
                    m_mode(seq(0, 2, 2), imode) = m_temp(seq(0, 2, 2));
                }
                P_mode[imode] = Ptemp;
            }
            m_b.emplace_back(m_mode);
            P_b.emplace_back(P_mode);
            solsKeep.emplace_back(sols[idx]);
        }
        vector<MatrixXd> m_b_final;
        vector<vector<MatrixXd>> P_b_final;
        vector<vector<int>> sols_final;
        vector<double> v_rB;
        if (r_b.size() > 0) {
            VectorXd rBTemp = VectorXd::Map(r_b.data(), r_b.size());
            rBTemp = (rBTemp.array() - log_sum_exp(rBTemp)).exp();
            for (int i = 0; i < rBTemp.size(); i++) { // prune low weight birth
                if (rBTemp(i) > mModel.rBMin) {
                    m_b_final.emplace_back(m_b[i]);
                    P_b_final.emplace_back(P_b[i]);
                    sols_final.emplace_back(solsKeep[i]);
                    v_rB.emplace_back(std::min(rBTemp(i), mModel.rBMax));
                }
            }
        }
        return {m_b_final, v_rB, P_b_final, sols_final};
    }

    // (adaptive_birth = 0[Fix birth], 1[Monte Carlo], 2[Kmeans], 3[MeanShift])
    void generate_birth(vector<VectorXd> meas_ru, vector<MatrixXd> meas_z, vector<MatrixXd> meas, int k) {
        tt_birth.resize(0);

        // (1) fix birth
        if (mAdaptiveBirth == 0) {
            for (int tabbidx = 0; tabbidx < mModel.r_birth.size(); tabbidx++) {
                MatrixXd m_temp = mModel.m_birth;
                vector<MatrixXd> Pb = mModel.P_birth;
                double r_b = mModel.r_birth[tabbidx];
                Target tt(m_temp, Pb, r_b, mId, MatrixXd::Zero(0, 0), mModel, k, false);
                mId += 1;
                tt_birth.push_back(tt);
            }
            return;
        }

        // (2) adaptive birth
        vector<MatrixXd> m_birth;
        vector<double> r_birth;
        vector<vector<int>> sols_birth;
        vector<vector<MatrixXd>> P_b;
        if (mAdaptiveBirth == 1) {
            tie(m_birth, r_birth, P_b, sols_birth) = mc_adaptive_birth_efficient(meas_ru, meas_z);
        } else if (mAdaptiveBirth == 2) {
            tie(m_birth, r_birth, P_b, sols_birth) = kmeans_adaptive_birth(meas_ru, meas_z);
        } else /*(mAdaptiveBirth == 3)*/ {
            tie(m_birth, r_birth, P_b, sols_birth) = meanshift_adaptive_birth(meas_ru, meas_z);
        }
        if (k == 0) {  // no information of birth (at the first time step), increase birth probability
            vector<double> v_ones(r_birth.size(), 0.7);
            r_birth = v_ones;
        }
        if (sols_birth.empty()) {
            return;
        }
        if (mPrunedTargets.size() == 0) {
            for (int idx = 0; idx < r_birth.size(); idx++) {
                double rbb = r_birth[idx];
                MatrixXd feat = MatrixXd::Zero(mModel.N_sensors, mFeatDim);
                for (int idx_s = 0; idx_s < mModel.N_sensors; idx_s++) {
                    int m_idx = sols_birth[idx][idx_s];
                    if (m_idx) {
                        feat.row(idx_s) = meas[idx_s](seq(5, mFeatDim + 4), m_idx - 1);
                    }
                }
                Target tt(m_birth[idx], P_b[idx], rbb, mId, feat, mModel, k, mUseFeat);
                mId += 1;
                tt_birth.push_back(tt);
            }
            return;
        }

        // (3) Perform re-appearing tracks and adding new birth targets
        MatrixXd allcost = MatrixXd::Ones(mPrunedTargets.size(), sols_birth.size());
        for (int idxsol = 0; idxsol < sols_birth.size(); idxsol++) {
            MatrixXd cdist_mat = MatrixXd::Ones(mPrunedTargets.size(), mModel.N_sensors);
            for (int s = 0; s < mModel.N_sensors; s++) {
                int m_idx = sols_birth[idxsol][s];
                if (m_idx) {
                    VectorXd z_feat = norm_feat01(meas[s](seq(5, mFeatDim + 4), m_idx - 1));
                    for (int pidx = 0; pidx < mPrunedTargets.size(); pidx++) {
                        VectorXd subtract = mPrunedTargets[pidx].mFeat.row(s).transpose() - z_feat;
                        cdist_mat(pidx, s) = subtract.norm(); // euclidean distance
                    }
                }
            }
            allcost.col(idxsol) = cdist_mat.rowwise().minCoeff(); // rowwise().prod()
        }
        VectorXd cost_len(mPrunedTargets.size());
        for (int pidx = 0; pidx < mPrunedTargets.size(); pidx++) {
            Target tt = mPrunedTargets[pidx];
            cost_len[pidx] = (k - tt.mBirthTime) / tt.mAh.size(); // survival_len / association_len
        }
        allcost = allcost.array().colwise() * cost_len.array();
        set<int> u_label{mPrunedTargetsLabel.data(), mPrunedTargetsLabel.data() + mPrunedTargetsLabel.size()};
        std::set<int>::iterator setIt = u_label.begin();
        MatrixXi u_mapping = MatrixXi::Zero(u_label.size(), sols_birth.size());
        MatrixXd u_allcost = MatrixXd::Zero(u_label.size(), sols_birth.size());
        for (int idx = 0; idx < u_label.size(); idx++) {
            MatrixXd label_cost = allcost;
            int label = *setIt;
            for (int jdx = 0; jdx < mPrunedTargetsLabel.size(); jdx++) {
                if (*setIt != mPrunedTargetsLabel[jdx]) {
                    label_cost.row(jdx).array() = INF;
                }
            }
            for (int jdx = 0; jdx < sols_birth.size(); jdx++) {
                u_allcost(idx, jdx) = label_cost.col(jdx).minCoeff();
                label_cost.col(jdx).minCoeff(&u_mapping(idx, jdx));
            }
            setIt++;
        }
        // LapJV (minimization), take negative logarithm to find a solution that maximizing likelihood
        VectorXi x_assignments;
        VectorXi y_assignments;
        lapjv(u_allcost, x_assignments, y_assignments, 0.3);
        double uc_sum = u_allcost.sum();
        vector<int> l_reappear;
        for (int idx = 0; idx < sols_birth.size(); idx++) {
            MatrixXd feat = MatrixXd::Zero(mModel.N_sensors, mFeatDim);
            for (int idx_s = 0; idx_s < mModel.N_sensors; idx_s++) {
                int m_idx = sols_birth[idx][idx_s];
                if (m_idx) {
                    feat.row(idx_s) = meas[idx_s](seq(5, mFeatDim + 4), m_idx - 1);
                }
            }
            if (y_assignments[idx] >= 0) {
                int tt_idx = u_mapping(y_assignments[idx], idx);
                Target target = mPrunedTargets[tt_idx];
                double r_b = std::min(mModel.r_birth[0], (uc_sum - u_allcost(y_assignments[idx], idx)) / uc_sum);
                target.re_activate(m_birth[idx], P_b[idx], r_b, feat, k);
                l_reappear.push_back(target.mL);
                tt_birth.push_back(target);
            } else { // re-id feature distance (a pruned_tt & a new birth) is not close => initiate new birth
                Target tt(m_birth[idx], P_b[idx], r_birth[idx], mId, feat, mModel, k, mUseFeat);
                mId += 1;
                tt_birth.push_back(tt);
            }
        }
        for (int i = mPrunedTargetsLabel.size() - 1; i >= 0; i--) {
            if (std::find(l_reappear.begin(), l_reappear.end(), mPrunedTargetsLabel[i]) != l_reappear.end()) {
                mPrunedTargets.erase(mPrunedTargets.begin() + i);
                mPrunedTargetsFeat.erase(mPrunedTargetsFeat.begin() + i);
                mPrunedTargetsLabel.erase(mPrunedTargetsLabel.begin() + i);
            }
        }
    }

    void clean_predict() {
        // hash label sets, find unique ones, merge all duplicates
        unordered_map<int, int> umap;
        VectorXd glmb_temp_w = NINF * VectorXd::Ones(glmb_update_w.size());
        vector<VectorXi> glmb_temp_I(glmb_update_I);
        VectorXi glmb_temp_n(glmb_update_n.size());
        int unique_idx = 0;
        std::size_t seed;
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            VectorXi glmb_I = glmb_update_I[hidx];
            std::sort(glmb_I.data(), glmb_I.data() + glmb_I.size());
            seed = glmb_I.size();
            for (auto &i : glmb_I) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            // If not present, then put it in unordered_set
            if (umap.find(seed) == umap.end()) {
                umap[seed] = unique_idx;
                glmb_temp_w[unique_idx] = glmb_update_w[hidx];
                glmb_temp_I[unique_idx] = glmb_update_I[hidx];
                glmb_temp_n[unique_idx] = glmb_update_n[hidx];
                unique_idx += 1;
            } else {
                Vector2d wlog_exp_temp;
                wlog_exp_temp << glmb_temp_w[umap[seed]], glmb_update_w[hidx];
                glmb_temp_w[umap[seed]] = log_sum_exp(wlog_exp_temp);
            }
        }
        glmb_update_w = glmb_temp_w(seq(0, unique_idx - 1));  // 2
        glmb_temp_I.erase(glmb_temp_I.begin() + unique_idx, glmb_temp_I.end());
        glmb_update_I = glmb_temp_I;  // 3
        glmb_update_n = glmb_temp_n(seq(0, unique_idx - 1));  // 4
    }

    void clean_update(int time_step) {
        // flag used tracks
        VectorXi usedindicator = VectorXi::Zero(glmb_update_tt.size());
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            usedindicator(glmb_update_I[hidx]).array() += 1;
        }
        // remove unused tracks and reindex existing hypotheses/components
        VectorXi newindices = VectorXi::Zero(glmb_update_tt.size());
        int new_idx = 0;
        vector<Target> glmb_clean_tt;
        for (int i = 0; i < newindices.size(); i++) {
            if (usedindicator(i) > 0) {
                newindices(i) = new_idx;
                new_idx += 1;
                glmb_clean_tt.push_back(glmb_update_tt[i]);
            }
        }
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            glmb_update_I[hidx] = newindices(glmb_update_I[hidx]);
        }
        glmb_update_tt = glmb_clean_tt;

        if (!mUseFeat) {
            return;
        }
        // remove pruned targets that are kept for 50 frames
        for (int i = mPrunedTargets.size() - 1; i >= 0; i--) { // remove from back to front
            if (time_step - mPrunedTargets[i].mLastActive > 50) {
                mPrunedTargets.erase(mPrunedTargets.begin() + i);
                mPrunedTargetsFeat.erase(mPrunedTargetsFeat.begin() + i);
                mPrunedTargetsLabel.erase(mPrunedTargetsLabel.begin() + i);
            }
        }
        // find pruned targets
        std::set<int> curr_tt_labels; // elements in a set are unique
        for (Target tt: glmb_clean_tt) {
            curr_tt_labels.insert(tt.mL);
        }
        std::set<int> pruned_labels;
        std::set_difference(mPreviousTargetLabel.begin(), mPreviousTargetLabel.end(), curr_tt_labels.begin(),
                            curr_tt_labels.end(), std::inserter(pruned_labels, pruned_labels.begin()));
        if (pruned_labels.size()) {
            for (Target tt: mPrevGlmbUpdateTt) {
                bool check = tt.mFeatFlag.sum() == 0; // check if target has feature from all sensors
                check = check && (tt.mAh.size() > 3); // check association history length
                check = check && (pruned_labels.find(tt.mL) != pruned_labels.end()); // check if existing
                if (check) {
                    mPrunedTargets.push_back(tt);
                    mPrunedTargetsFeat.push_back(tt.mFeat);
                    mPrunedTargetsLabel.push_back(tt.mL);
                }
            }
        }

        mPrevGlmbUpdateTt = glmb_clean_tt;
        mPreviousTargetLabel = curr_tt_labels;
    }

    void prune(Filter filter) {
        // prune components with weights lower than specified threshold
        vector<int> idxkeep;
        vector<VectorXi> glmb_out_I;
        for (int i = 0; i < glmb_update_I.size(); i++) {
            if (glmb_update_w(i) > filter.hyp_threshold) {
                idxkeep.push_back(i);
                glmb_out_I.push_back(glmb_update_I[i]);
            }
        }
        VectorXi idxkeep_eigen = VectorXi::Map(idxkeep.data(), idxkeep.size());
        VectorXd glmb_out_w = glmb_update_w(idxkeep_eigen);
        glmb_out_w = glmb_out_w.array() - log_sum_exp(glmb_out_w);
        VectorXi glmb_out_n = glmb_update_n(idxkeep_eigen);
        VectorXd glmb_out_cdn = NINF * VectorXd::Ones(glmb_out_n.maxCoeff() + 1);
        for (int card = 0; card < glmb_out_cdn.size(); card++) {
            VectorXi card_check = (glmb_out_n.array() == card).cast<int>();
            if (card_check.sum()) {
                VectorXd vlog_card = card_check.select(glmb_out_w, NINF);
                glmb_out_cdn[card] = log_sum_exp(vlog_card);
            }
        }
        glmb_out_cdn = glmb_out_cdn.array().exp();

        glmb_update_w = glmb_out_w;  // 2
        glmb_update_I = glmb_out_I;  // 3
        glmb_update_n = glmb_out_n;  // 4
        glmb_update_cdn = glmb_out_cdn;  // 5
    }

    void cap(Filter filter) {
        // cap total number of components to specified maximum
        if (glmb_update_w.size() > filter.H_max) {
            // initialize original index locations
            vector<double> v_glmb_w(glmb_update_w.size());
            VectorXd::Map(&v_glmb_w[0], glmb_update_w.size()) = glmb_update_w;
            vector<int> idx(glmb_update_w.size());
            std::iota(idx.begin(), idx.end(), 0);
            stable_sort(idx.begin(), idx.end(),
                        [&v_glmb_w](int i1, int i2) { return v_glmb_w[i1] > v_glmb_w[i2]; });
            VectorXi idx_eigen = VectorXi::Map(idx.data(), idx.size());
            VectorXi idxkeep_eigen = idx_eigen(seq(0, filter.H_max - 1));

            VectorXd glmb_out_w = glmb_update_w(idxkeep_eigen);
            glmb_out_w = glmb_out_w.array() - log_sum_exp(glmb_out_w);
            VectorXi glmb_out_n = glmb_update_n(idxkeep_eigen);
            vector<VectorXi> glmb_out_I;
            for (int i: idxkeep_eigen) {
                glmb_out_I.push_back(glmb_update_I[i]);
            }

            VectorXd glmb_out_cdn = NINF * VectorXd::Ones(glmb_out_n.maxCoeff() + 1);
            for (int card = 0; card < glmb_out_cdn.size(); card++) {
                VectorXd vlog_card = (glmb_out_n.array() == card).select(glmb_out_w, NINF);
                glmb_out_cdn[card] = log_sum_exp(vlog_card);
            }
            glmb_out_cdn = glmb_out_cdn.array().exp();

            glmb_update_w = glmb_out_w;  // 2
            glmb_update_I = glmb_out_I;  // 3
            glmb_update_n = glmb_out_n;  // 4
            glmb_update_cdn = glmb_out_cdn;  // 5
        }
    }

    std::tuple<MatrixXd, int, MatrixXi, vector<int>> extract_estimates(Model model) {
        // extract estimates via best cardinality, then
        // best component/hypothesis given best cardinality, then
        // best means of tracks given best component/hypothesis and cardinality
        int N;
        glmb_update_cdn.maxCoeff(&N);
        MatrixXd X(model.x_dim, N);
        MatrixXi L(2, N);
        vector<int> S(N);
        int idxcmp; // Logarithm form of glmb_update_w is negative
        (glmb_update_w.array() * (glmb_update_n.array() == N).cast<double>()).minCoeff(&idxcmp);
        for (int n = 0; n < N; n++) {
            int idxptr = glmb_update_I[idxcmp](n);
            Target tt = glmb_update_tt[idxptr];
            int ind;
            tt.mMode.maxCoeff(&ind);
            int start = tt.mGmLen(seq(0, ind - 1)).sum();
            int end = start + tt.mGmLen[ind];
            int indx;
            tt.mW(seq(start, end - 1)).maxCoeff(&indx);
            X.col(n) = tt.mM.col(start + indx);
            L.col(n) << tt.mBirthTime, tt.mL;
            S[n] = ind; // model.mode_type[ind];
        }
        return {X, N, L, S};
    }

public:
    // Choose the following options (adaptive_birth = 0 [Fix birth], 1 [Monte Carlo], 2 [MeanShift])
    // (0) Fix birth uses re-id feature => [adaptive_birth=0, use_feat=False] => ONLY for CMC dataset
    // (1.1) Monte Carlo Adaptive birth uses re-id feature [adaptive_birth=1, use_feat=True]
    // (1.2) Monte Carlo Adaptive birth does NOT use re-id feature [adaptive_birth=1, use_feat=False]
    // (2.1) KMeans Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
    // (2.2) KMeans Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
    // (3.1) MeanShift Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
    // (3.2) MeanShift Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
    MSGLMB(vector<MatrixXd> camera_mat, string dataset = "CMC1", int adaptiveBirth = 1, bool useFeat = true) {
        glmb_update_w = VectorXd::Zero(1);
        glmb_update_I.push_back(VectorXi(0));
        glmb_update_n = VectorXi::Zero(1);
        glmb_update_cdn = VectorXd::Ones(1);
        filter = Filter();
        mModel = Model(camera_mat, dataset);
        double mc_pdf_c = log(1.0 / ((mModel.XMAX[1] - mModel.XMAX[0]) * (mModel.YMAX[1] - mModel.YMAX[0])));
        mMCAB = MCAdaptiveBirth(mModel.camera_mat, 100, 500, mModel.P_D, mModel.lambda_c, mc_pdf_c);
        mMCAB.setMeasureNoise(mModel.R[0], mModel.meas_n_mu.col(0));
        mMCAB.setBirthProb(mModel.rBMin, mModel.rBMax);
        mMCAB.setNumSensorDetect(mModel.numDet);

        mUseFeat = useFeat;
        mAdaptiveBirth = adaptiveBirth;
        mId = 0;
    }

    std::tuple<MatrixXd, int, MatrixXi, vector<int>> run_msglmb_ukf(vector<MatrixXd> measZ, int kt) {

        msjointpredictupdate(mModel, filter, measZ, kt);
        int H_posterior = glmb_update_w.size();

        // pruning and truncation
        prune(filter);
        int H_prune = glmb_update_w.size();
        cap(filter);
        int H_cap = glmb_update_w.size();
        clean_update(kt);

        VectorXd rangetmp = VectorXd::LinSpaced(glmb_update_cdn.size(), 0, glmb_update_cdn.size() - 1);
        cout << "Time " << kt << " #eap cdn=" << rangetmp.transpose() * glmb_update_cdn;
        int temp1 = ((VectorXd) rangetmp.array().pow(2)).transpose() * glmb_update_cdn;
        int temp2 = (rangetmp.transpose() * glmb_update_cdn);
        cout << " #var cdn=" << temp1 - temp2 * temp2;
        cout << " #comp pred=" << H_posterior;
        cout << " #comp post=" << H_posterior;
        cout << " #comp updt=" << H_cap;
        cout << " #trax updt=" << glmb_update_tt.size() << endl;

        // state estimation and display diagnostics
        return extract_estimates(mModel);
    }
};


#endif //UKF_TARGET_MS_GLMB_UKF_HPP
