import lap
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt


# X,Y -  3-D matrices RxTxN (R: states of 3D rectangles (two corners representation, e.g. [x1,y1,x2,y2,z1,z2])
def ospa_track_metrics(X, Y, wl, flagGIoU, with_extent=True, threshold=1):
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        print('Dimensions of X and Y are inconsistent')

    eval_idx = np.arange(0, X.shape[1])
    win_off = np.arange(-wl + 1, 1)

    num_x = X.shape[2]
    num_y = Y.shape[2]
    num_step = X.shape[1]

    distances = np.zeros((num_x, num_y, num_step))
    x_exists = np.zeros((num_x, num_step), dtype=bool)
    y_exists = np.zeros((num_y, num_step), dtype=bool)

    for i in range(num_step):
        # Compute distance between every pair of points
        x = X[:, i, :]
        y = Y[:, i, :]
        xx = np.tile(x, (1, num_y))
        yy = np.repeat(y, num_x).reshape((y.shape[0], num_x * num_y))

        # calculate distance between 3D bounding boxes
        if with_extent:
            ax = np.prod(xx[2:4, :] - xx[0:2, :], axis=0)  # The rectangle areas in X, Ground Truth
            ay = np.prod(yy[2:4, :] - yy[0:2, :], axis=0)  # The rectangle areas in Y, Estimated Tracks
            VX = ax * (xx[5, :] - xx[4, :])  # Volume of Ground Truth
            VY = ay * (yy[5, :] - yy[4, :])  # Volume of Estimated Tracks
            xym = np.fmin(xx, yy)
            xyM = np.fmax(xx, yy)
            V_Int = np.zeros(xx.shape[1])
            ind = np.all(xyM[[0, 1, 4], :] < xym[[2, 3, 5], :], axis=0)
            intersect = np.prod(xym[2:4, ind] - xyM[0:2, ind], axis=0)
            V_Int[ind] = intersect * (xym[5, ind] - xyM[4, ind])
            V_Unn = VX + VY - V_Int
            V_IoU = V_Int / V_Unn
            if flagGIoU:
                V_Cc = np.prod(xyM[2:4, :] - xym[0:2, :], axis=0) * (xyM[5, :] - xym[4, :])
                V_GIoU = V_IoU - ((V_Cc - V_Unn) / V_Cc)
                d = 0.5 * (1 - V_GIoU).reshape((num_y, num_x)).T
            else:
                d = (1 - V_IoU).reshape((num_y, num_x)).T
        # euclidean distance used for Wildtrack dataset, 3D Center Position (restricted to the ground plane)
        else:
            d = np.sqrt(np.square(xx[0, :] - yy[0, :]) + np.square(xx[1, :] - yy[1, :]))
            d = d.reshape((num_y, num_x)).T

        # Compute track existence flags
        x_exists[:, i] = np.invert(np.isnan(x[0, :]))
        y_exists[:, i] = np.invert(np.isnan(y[0, :]))

        # Distance between an empty and non-empty state
        one_exists = np.logical_xor(x_exists[:, i].reshape(-1, 1), y_exists[:, i].reshape(1, -1))
        d[one_exists] = threshold

        # Find times when neither object exists
        neither_exists = np.logical_and((1 - x_exists[:, i]).reshape(-1, 1), 1 - y_exists[:, i].reshape(1, -1))

        # Full window, distance between empty states is 'nan'
        # A Solution for Large-Scale Multi-Object Tracking, eq(33)
        d[neither_exists] = np.nan

        # Store the distance matrix for this step
        distances[:, :, i] = d

    # Cap all inter-point distances
    # Full window
    distances = np.clip(distances, None, threshold)  # Cut-off threshold

    # Window indices
    win_idx = eval_idx[wl - 1] + win_off
    idx_val = np.logical_and((win_idx >= 0), (win_idx < num_step))
    win_idx = win_idx[idx_val]

    # Compute the matrix of weighted time-averaged
    # OSPA distances between tracks
    trk_dist = np.nanmean(distances[:, :, win_idx], axis=2)

    # Get the number of objects in X and Y that exist
    # at any time inside the current window
    valid_rows = np.any(x_exists[:, win_idx], axis=1)
    valid_cols = np.any(y_exists[:, win_idx], axis=1)
    m = sum(valid_rows)
    n = sum(valid_cols)

    # Solve the optimal assignment problem
    trk_dist = trk_dist[valid_rows, :]
    trk_dist = trk_dist[:, valid_cols]
    if np.prod(trk_dist.shape) == 0:
        cost = 0
    else:
        if m > n:
            trk_dist = trk_dist.T
        cost, _ = lap.lapjv(trk_dist, extend_cost=True)[0:2]

    # Compute the OSPA track distances
    if max(m, n) == 0:
        ospa2 = 0
    else:
        ospa2 = (abs(m - n) + cost) / max(m, n)

    return ospa2


def build_distance(np_array, K, with_extent):
    IDs_X = np.unique(np_array[:, 1]).astype(int)
    distances = np.empty((6, K, len(IDs_X)))
    distances[:] = np.NaN
    for idx_id, id in enumerate(IDs_X):
        identity_idx = (np_array[:, 1] == id)
        frame_idx = np_array[identity_idx, 0].astype(int) - 1
        if len(np.unique(frame_idx)) != len(frame_idx):
            print('An ID appears more than one in a time step.')
            exit(1)
        id_data = np_array[identity_idx, 7:].T
        # Change from [xc, yc, zc, xh, yh, zh] to [x1,y1,x2,y2,z1,z2 ,x2>x1 & y2>y1 && z2>z1]
        id_data_format = np.copy(id_data)
        if with_extent:
            id_data_format[:2, :] = id_data[:2, :] - id_data[3:5, :]
            id_data_format[2:4, :] = id_data[:2, :] + id_data[3:5, :]
            id_data_format[4, :] = id_data[2, :] - id_data[5, :]
            id_data_format[5, :] = id_data[2, :] + id_data[5, :]
        distances[:, frame_idx, idx_id] = id_data_format
    return distances


def ospa2_datasets(gt_list, est_list, dataset_list, flagGIoU=True, with_extent=True):
    ospa2 = np.zeros(len(dataset_list))
    for idx, (gt, est) in enumerate(zip(gt_list, est_list)):
        num_frame = int(max(gt[:, 0]))
        X = build_distance(gt, num_frame, with_extent)
        Y = build_distance(est, num_frame, with_extent)
        ospa2_temp = ospa_track_metrics(X, Y, Y.shape[1] - 1, flagGIoU, with_extent)
        ospa2[idx] = ospa2_temp
    results = ""
    for idx, seq in enumerate(dataset_list):
        results += seq + "\t" + str(ospa2[idx]) + "\n"
    results += "Overall\t" + str(round(np.average(ospa2), 2)) + "(" + str(round(np.std(ospa2), 2)) + ")"
    return results


def ospa2_single_dataset(est, dataset="CMC1", gt_data_dir="../../data/images/"):
    gt_file = os.path.join(gt_data_dir, dataset, "GT_" + dataset + "_WORLD_CENTROID.txt")
    np_gt = pd.read_csv(gt_file, delimiter=' ', header=None).to_numpy()
    # est_file = os.path.join(gt_dataDir, dataset, "EST_" + dataset + "_WORLD_CENTROID.txt")
    # np_est = pd.read_csv(est_file, delimiter=' ', header=None).to_numpy()

    # Ground Truth
    K = int(max(np_gt[:, 0]))
    IDs_X = np.unique(np_gt[:, 1]).astype(int)
    X = np.empty((6, K, len(IDs_X)))
    X[:] = np.NaN
    for idx_id, id in enumerate(IDs_X):
        identity_idx = (np_gt[:, 1] == id)
        frame_idx = np_gt[identity_idx, 0].astype(int) - 1
        if len(np.unique(frame_idx)) != len(frame_idx):
            print('An ID appears more than one in a time step.')
            exit(1)
        id_data = np_gt[identity_idx, 7:].T
        # Change from [xc, yc, zc, xh, yh, zh] to [x1,y1,x2,y2,z1,z2 ,x2>x1 & y2>y1 && z2>z1]
        id_data_format = np.copy(id_data)
        id_data_format[:2, :] = id_data[:2, :] - id_data[3:5, :]
        id_data_format[2:4, :] = id_data[:2, :] + id_data[3:5, :]
        id_data_format[4, :] = id_data[2, :] - id_data[5, :]
        id_data_format[5, :] = id_data[2, :] + id_data[5, :]
        X[:, frame_idx, idx_id] = id_data_format

    # Estimated Tracks
    np_est = est[:, :13]  # do not use {modes}
    IDs_Y = np.unique(np_est[:, 1])
    Y = np.empty((6, K, len(IDs_Y)))
    Y[:] = np.NaN
    for idx_id, id in enumerate(IDs_Y):
        identity_idx = (np_est[:, 1] == id)
        frame_idx = np_est[identity_idx, 0].astype(int) - 1
        if len(np.unique(frame_idx)) != len(frame_idx):
            print('An ID appears more than one in a time step.')
            exit(1)
        id_data = np_est[identity_idx, 7:].T
        # Change from [xc, yc, zc, xh, yh, zh] to [x1,y1,x2,y2,z1,z2) ,x2>x1 & y2>y1 && z2>z1]
        id_data_format = np.copy(id_data)
        id_data_format[0:2, :] = id_data[0:2, :] - id_data[3:5, :]
        id_data_format[2:4, :] = id_data[0:2, :] + id_data[3:5, :]
        id_data_format[4, :] = id_data[2, :] - id_data[5, :]
        id_data_format[5, :] = id_data[2, :] + id_data[5, :]
        Y[:, frame_idx, idx_id] = id_data_format

    # GIoU/IoU OSPA(2)
    ospa_win = np.zeros(K)
    for k in range(1, K + 1):
        ospa_win[k - 1] = ospa_track_metrics(X, Y, k, flagGIoU=True)
    print("OSPA2 for " + dataset, ospa_win[K - 1])
    plt.plot(np.arange(K), ospa_win)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, K])
    ax.plot(np.arange(K), 0.5 * np.ones(K), linestyle='--')
    plt.title(dataset + " OSPA" + r"$^{(2)}$ :: " + "gIOU")
    plt.show()


if __name__ == '__main__':
    # ospa2_single_dataset(None)

    results_root = "../experiments/matlab_release_jonah_fairmotcmc"
    gt_data_dir = "../../data/images/"
    dataset = "CMC5"
    np_gt_list = []
    np_est_list = []
    dataset_list = []
    with_extent = True
    for i in range(25):
        gt_file = os.path.join(gt_data_dir, dataset, "GT_" + dataset + "_WORLD_CENTROID.txt")
        np_gt = pd.read_csv(gt_file, delimiter=' ', header=None).to_numpy()
        est_file = os.path.join(results_root, "EST_" + dataset + str(i + 1) + "_WORLD_CENTROID.txt")
        est = pd.read_csv(est_file, delimiter=',', header=None).to_numpy()
        est[:, [10, 11, 12]] = np.exp(est[:, [10, 11, 12]])  # CMC dataset
        np_gt_list.append(np_gt)
        np_est_list.append(est)
        dataset_list.append(dataset + str(i + 1))
    ospa2_datasets(np_gt_list, np_est_list, dataset_list, True, with_extent)
