#include <Eigen/SparseCore>

using namespace std;

Eigen::MatrixXd bboxes_ioi_xyah_back2front_all(Eigen::MatrixXd tt_ltlogwh) {
    /*
    Parameters
    ----------
    tt_lmb (N, 4) ndarray of double, [Left, Top, Log Width, Log Height]
    Returns
    -------
    overlaps: (N, N) ndarray of mutual overlap between boxes from back to front
    */
    unsigned int N = tt_ltlogwh.rows();
    double iw, ih, ioi, ioi_temp;
    unsigned int k, n;
    Eigen::VectorXd query_area;

    Eigen::MatrixXd overlaps(N, N);
    overlaps.setZero();
    if (N == 0) {
        return overlaps;
    }

    tt_ltlogwh(Eigen::all, Eigen::seq(2, 3)) = tt_ltlogwh(Eigen::all, Eigen::seq(2, 3)).array().exp();
    query_area = (tt_ltlogwh.col(2).array() + 1) * (tt_ltlogwh.col(3).array() + 1);

    tt_ltlogwh.col(3) += tt_ltlogwh.col(1);      // bottom
    tt_ltlogwh.col(2) += tt_ltlogwh.col(0);      // right
    for (k = 0; k < N; k++) {
        ioi = 0;
        for (n = 0; n < N; n++) {
            if ((tt_ltlogwh(n, 3) < tt_ltlogwh(k, 3)) || (n == k)) {
                continue; // ignore objects stand behind or itself
            }
            iw = (
                    std::min(tt_ltlogwh(n, 2), tt_ltlogwh(k, 2)) -
                    std::max(tt_ltlogwh(n, 0), tt_ltlogwh(k, 0)) + 1
            );
            if (iw > 0) {
                ih = (
                        std::min(tt_ltlogwh(n, 3), tt_ltlogwh(k, 3)) -
                        std::max(tt_ltlogwh(n, 1), tt_ltlogwh(k, 1)) + 1
                );
                if (ih > 0) {
                    overlaps(k, n) = iw * ih / query_area[k];
                }
            }
        }
    }
    return overlaps;
}

Eigen::VectorXd bboxes_ioi_xyah_back2front_all_v2(Eigen::MatrixXd tt_ltlogwh) {
    /*
    Parameters
    ----------
    tt_lmb (N, 4) ndarray of double, [Left, Top, Log Width, Log Height]
    Returns
    -------
    overlaps: (N) ndarray of non-overlap intersection over the area of bboxes from back to front (Output is PD)
    */
    unsigned int N = tt_ltlogwh.rows();
    int iw, ih, min_x, max_x, min_y, max_y;
    unsigned int k, n;
    Eigen::VectorXd query_area;

    Eigen::VectorXd overlaps = Eigen::VectorXd::Zero(N);
    if (N == 0) {
        return overlaps;
    }

    tt_ltlogwh(Eigen::all, Eigen::seq(2, 3)) = tt_ltlogwh(Eigen::all, Eigen::seq(2, 3)).array().exp();
    Eigen::MatrixXi tt_wh = tt_ltlogwh(Eigen::all, Eigen::seq(2, 3)).array().ceil().cast<int>();

    tt_ltlogwh.col(3) += tt_ltlogwh.col(1);      // bottom
    tt_ltlogwh.col(2) += tt_ltlogwh.col(0);      // right

    Eigen::MatrixXi tt_ltrb = tt_ltlogwh.cast<int>();

    for (k = 0; k < N; k++) {
        Eigen::MatrixXi rectangle = Eigen::MatrixXi::Ones(tt_wh(k, 0), tt_wh(k, 1));
        for (n = 0; n < N; n++) {
            if ((tt_ltrb(n, 3) < tt_ltrb(k, 3)) || (n == k)) {
                continue; // ignore objects stand behind or itself
            }
            min_x = std::min(tt_ltrb(n, 2), tt_ltrb(k, 2));
            max_x = std::max(tt_ltrb(n, 0), tt_ltrb(k, 0));
            min_y = std::min(tt_ltrb(n, 3), tt_ltrb(k, 3));
            max_y = std::max(tt_ltrb(n, 1), tt_ltrb(k, 1));
            iw = min_x - max_x; // + 1;
            if (iw > 0) {
                ih = min_y - max_y; // + 1;
                if (ih > 0) {
                    int start_x = max_x - tt_ltrb(k, 0);
                    int end_x = min_x - tt_ltrb(k, 0) - 2;
                    int start_y = max_y - tt_ltrb(k, 1);
                    int end_y = min_y - tt_ltrb(k, 1) - 2;
                    rectangle(Eigen::seq(start_x, end_x), Eigen::seq(start_y, end_y)).array() = 0;
                }
            }
        }
        overlaps(k) = rectangle.sum() * 1.0 / (tt_wh(k, 0) * tt_wh(k, 1));
    }
    return overlaps;
}

Eigen::VectorXd bbox_giou_ltlwh(Eigen::VectorXd box, Eigen::MatrixXd query_boxes, bool giou = true) {
    /*
    Parameters
    ----------
    box: (4) vector of double [Left, Top, Log Width, Log Height]
    query_boxes: (4, K) array of double [Left, Top, Log Width, Log Height]
    Returns
    -------
    overlaps (GIoU/IoU): (K) array of overlap between box and query_boxes
    */
    query_boxes(Eigen::seq(2, 3), Eigen::all) = query_boxes(Eigen::seq(2, 3), Eigen::all).array().exp();
    box(Eigen::seq(2, 3)) = box(Eigen::seq(2, 3)).array().exp();

    unsigned int K = query_boxes.cols();
    Eigen::VectorXd overlaps = Eigen::VectorXd::Zero(K);
    double iw, ih, box_area, query_area, ua;
    double query_t, query_l, query_b, query_r;
    double box_t, box_l, box_b, box_r;
    double xMax, yMax, xMin, yMin, uc;
    box_t = box(1);  // 0
    box_l = box(0);   // 1
    box_b = box(1) + box(3);  // 2
    box_r = box(0) + box(2);   // 3
    box_area = (box(2) + 1) * (box(3) + 1);
    for (unsigned int k = 0; k < K; k++) {
        query_t = query_boxes(1, k);  // 0
        query_l = query_boxes(0, k);  // 1
        query_b = query_boxes(1, k) + query_boxes(3, k);  // 2
        query_r = query_boxes(0, k) + query_boxes(2, k);  // 3
        query_area = (query_boxes(2, k) + 1) * (query_boxes(3, k) + 1);

        xMax = std::max(box_r, query_r);
        xMin = std::min(box_l, query_l);
        yMax = std::max(box_b, query_b);
        yMin = std::min(box_t, query_t);

        ih = (std::min(box_b, query_b) - std::max(box_t, query_t) + 1);
        ua = box_area + query_area;
        if (ih > 0) {
            iw = (std::min(box_r, query_r) - std::max(box_l, query_l) + 1);
            if (iw > 0) {
                ua = ua - iw * ih;
                overlaps(k) = iw * ih / ua;
            }
        }
        if (giou) {
            uc = (xMax - xMin) * (yMax - yMin);
            overlaps(k) = overlaps(k) - (uc - ua) / uc;
            // The farther two objects are, the smaller this value is and vice versa
            overlaps(k) = 0.5 + 0.5 * overlaps(k);
        }
    }
    return overlaps;
}