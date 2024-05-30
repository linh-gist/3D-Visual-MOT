import os
import pandas as pd
import motmetrics as mm
import numpy as np


# input: x center [x y z wx/2 wy/2 h/2]
# output: bbox_3d [x1,y1,x2,y2,z1,z2 ,x2>x1 & y2>y1 && z2>z1]
def centerxyz_3dbbox(x):
    bbox_3d = np.copy(x)
    bbox_3d[0:2] = x[0:2] - x[3:5]
    bbox_3d[2:4] = x[0:2] + x[3:5]
    bbox_3d[4] = x[2] - x[5]
    bbox_3d[5] = x[2] + x[5]
    return bbox_3d


def bbox3d_distance(box3dA, box3dB, compute_giou=True):
    # # vectorization similar to calculation in OSPA2
    # aA = np.prod(box3dA[2:4] - box3dA[0:2])  # Area of box3dA
    # aB = np.prod(box3dB[2:4] - box3dB[0:2])  # Area of box3dB
    # vA = aA * (box3dA[5] - box3dA[4])  # Volume of box3dA
    # vB = aB * (box3dB[5] - box3dB[4])  # Volume of box3dB
    # xym = np.fmin(box3dA, box3dB)
    # xyM = np.fmax(box3dA, box3dB)
    # ind = np.all(xyM[[0, 1, 4]] < xym[[2, 3, 5]])
    # intersect = 0
    # if ind:  # determine whether can be intersected or not
    #     intersect = np.prod(xym[2:4] - xyM[0:2])
    # V_Int = intersect * (xym[5] - xyM[4])
    # V_Unn = vA + vB - V_Int
    # V_IoU = V_Int / V_Unn
    # if compute_giou:
    #     V_Cc = np.prod(xyM[2:4] - xym[0:2]) * (xyM[5] - xym[4])
    #     V_GIoU = V_IoU - ((V_Cc - V_Unn) / V_Cc)
    #     d = 0.5 * (1 - V_GIoU)
    #     return d
    #     # return the intersection over union value
    # return 1 - V_IoU

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box3dA[0], box3dB[0])
    yA = max(box3dA[1], box3dB[1])
    zA = max(box3dA[4], box3dB[4])
    xB = min(box3dA[2], box3dB[2])
    yB = min(box3dA[3], box3dB[3])
    zB = min(box3dA[5], box3dB[5])
    # compute the area of intersection rectangle
    interVolume = max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAVolume = (box3dA[2] - box3dA[0]) * (box3dA[3] - box3dA[1]) * (box3dA[5] - box3dA[4])
    boxBVolume = (box3dB[2] - box3dB[0]) * (box3dB[3] - box3dB[1]) * (box3dB[5] - box3dB[4])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    volUnion = boxAVolume + boxBVolume - interVolume
    volIoU = interVolume / volUnion
    if compute_giou:
        xMaxC = max(box3dA[2], box3dB[2])
        yMaxC = max(box3dA[3], box3dB[3])
        zMaxC = max(box3dA[5], box3dB[5])
        xminC = min(box3dA[0], box3dB[0])
        yminC = min(box3dA[1], box3dB[1])
        zminC = min(box3dA[4], box3dB[4])
        volC = (xMaxC - xminC) * (yMaxC - yminC) * (zMaxC - zminC)
        volGIoU = volIoU - (volC - volUnion) / volC
        # distance (lower is better), see motmetrics.iou_matrix()
        volGIoU = 0.5 * (1 - volGIoU)
        return volGIoU
    # return the intersection over union value
    return 1 - volIoU


def euclid_distance(x, y):  # use to evaluate Wildtrack dataset
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


# Compute distance between every pair of points
def clear_mot(gt_list, est_list, dataset_list, max_iou=0.5, with_extent=True):
    acc = [None for i in gt_list]
    for idx, (gt, est) in enumerate(zip(gt_list, est_list)):
        num_frame = int(max(gt[:, 0]))
        acc[idx] = mm.MOTAccumulator(auto_id=True)

        for frame_idx in range(num_frame):
            targets_gt = gt[gt[:, 0] == (frame_idx + 1)]
            gt_ids = targets_gt[:, 1]
            targets_est = est[est[:, 0] == (frame_idx + 1)]
            trk_ids = targets_est[:, 1]
            iou = np.zeros((len(targets_gt), len(targets_est)))
            for gt_idx, tt_gt in enumerate(targets_gt):
                if with_extent:
                    bbox3d_gt = centerxyz_3dbbox(tt_gt[7:])  # CMC
                for est_idx, tt_est in enumerate(targets_est):
                    if with_extent:
                        bbox3d_est = centerxyz_3dbbox(tt_est[7:])  # CMC
                        iou[gt_idx, est_idx] = bbox3d_distance(bbox3d_gt, bbox3d_est)  # CMC
                    else:
                        iou[gt_idx, est_idx] = euclid_distance(tt_gt[7:9], tt_est[7:9])  # Wildtrack
            # Object / hypothesis points with larger distance are set to np.nan signalling do-not-pair.
            iou = np.where(iou > max_iou, np.nan, iou)
            acc[idx].update(gt_ids, trk_ids, iou)  # Wildtrack dataset, iou / np.amax(iou)

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc,
        metrics=metrics,
        names=dataset_list,
        generate_overall=True
    )
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    return strsummary


def clearmot_single_dataset(est, dataset="CMC1", gt_data_dir="../../data/images/"):
    # save_format = '{frame},{id}, -1, -1, -1, -1, -1,{x},{y},{z},{wx},{wy},{h}, {modes}\n'
    est = est[:, :13]  # do not use {modes}
    gt_file = os.path.join(gt_data_dir, dataset, "GT_" + dataset + "_WORLD_CENTROID.txt")
    np_gt = pd.read_csv(gt_file, delimiter=' ', header=None).to_numpy()
    # est_file = os.path.join(gt_dataDir, dataset, "EST_" + dataset + "_WORLD_CENTROID.txt")
    # est = pd.read_csv(est_file, delimiter=' ', header=None).to_numpy()
    # <x_half y_half z_half> is the half-lengths of the ellipsoid
    # est[:, [10, 11, 12]] = np.exp(est[:, [10, 11, 12]])  # CMC dataset
    strsummary = clear_mot([np_gt], [est], [dataset])
    print("CLEAR MOT for ", dataset)
    print(strsummary)
    root = "./results"
    np.savetxt(os.path.join(root, 'EST_{}_WORLD_CENTROID.txt'.format(dataset)), est)
    with open(os.path.join(root, 'summary_clearmot_{}.txt'.format(dataset)), 'w') as f:
        f.write(strsummary)
    #


if __name__ == '__main__':
    # clearmot_single_dataset(None, "CMC1")

    results_root = "../experiments/matlab_release_jonah_fairmotcmc"
    gt_data_dir = "../../data/images/"
    dataset = "CMC5"
    np_gt_list = []
    np_est_list = []
    dataset_list = []
    for i in range(25):
        gt_file = os.path.join(gt_data_dir, dataset, "GT_" + dataset + "_WORLD_CENTROID.txt")
        np_gt = pd.read_csv(gt_file, delimiter=' ', header=None).to_numpy()
        est_file = os.path.join(results_root, "EST_" + dataset + str(i + 1) + "_WORLD_CENTROID.txt")
        est = pd.read_csv(est_file, delimiter=',', header=None).to_numpy()
        est[:, [10, 11, 12]] = np.exp(est[:, [10, 11, 12]])  # CMC dataset
        np_gt_list.append(np_gt)
        np_est_list.append(est)
        dataset_list.append(dataset + str(i + 1))
    strsummary = clear_mot(np_gt_list, np_est_list, dataset_list, 1, False)
    print("CLEAR MOT for ", dataset)
    print(strsummary)
