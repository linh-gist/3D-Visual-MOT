import glob
import random
import cv2
import numpy as np
from gen_model import model
from cppmsglmb import bboxes_ioi_xyah_back2front_all


def homtrans(T, p):
    if T.shape[0] == p.shape[0]:
        if len(p.shape) == 3:
            pt = []
            for i in range(p.shape[2]):
                pt.append(np.dot(T, p[:, :, i]))
            pt = np.stack(pt, axis=2)
        else:
            pt = np.dot(T, p)
    else:
        if T.shape[0] - p.shape[0] == 1:
            e2h = np.row_stack((p, np.ones(p.shape[1])))  # E2H Euclidean to homogeneous
            temp = np.dot(T, e2h)
            # H2E Homogeneous to Euclidean
            numrows = temp.shape[0]
            pt = temp[:numrows - 1, :] / np.tile(temp[numrows - 1, :], (numrows - 1, 1))
        else:
            print("matrices and point data do not conform")
    return pt


def visual_ioa(img0, bboxes):
    # img0 = cv2.imread(imgdirs[s][i])
    # visual_ioa(img0, bboxes)
    bboxes_visual = np.copy(bboxes)
    bboxes_visual[:, 2:4] += bboxes_visual[:, 0:2]  # convert ltwh to ltrb
    bboxes_ioa = np.copy(bboxes)
    bboxes_ioa[:, 2:4] = np.log(bboxes_ioa[:, 2:4])  # convert ltwh to l,t,log(w),log(h)
    ioa = bboxes_ioi_xyah_back2front_all(bboxes_ioa)
    ioa = 1 - np.max(ioa, axis=1)
    for ibbox, bbox in enumerate(bboxes_visual):
        r = np.random.randint(50, 255)
        g = np.random.randint(50, 255)
        b = np.random.randint(50, 255)
        rand_color = (r, g, b)
        l, t = bbox[0], bbox[1]
        r, b = bbox[2], bbox[3]
        # draw bbox
        # img0 = cv2.circle(img0, (l, t), radius=8, color=(255, 255, 255), thickness=-1)
        img0 = cv2.rectangle(img0, (l, t), (r, b), color=rand_color, thickness=3)
        img0 = cv2.putText(img0, str(ibbox) + " IoA " + str(round(ioa[ibbox], 2)), org=(int(l + 5), int(b - 10)),
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=rand_color, thickness=2)
    cv2.imshow("Intersection over Area", img0)
    cv2.waitKey(0)


def load_detection(model, dataset="CMC1", reconfig=False, images_root="../../data/images/"):
    imgdirs = []
    dets = []
    timestep = 0
    conf = 0.1  # Detection confidence score
    # w, h = 1920, 1024  # image width, height (suppose same value for all images)
    for s in range(1, model.N_sensors + 1):
        det = np.load("../detection/fairmot/" + dataset + "/Cam_" + str(s) + ".npz")
        dets.append(det)
        timestep = len(det.files)  # make sure all cameras collect the same number of frames
        # imgs = sorted(glob.glob(images_root + dataset + "/Cam_" + str(s) + "/*.png"))
        # imgdirs.append(imgs)
    meas = []
    meas_soff = []
    divide_conf = 5
    cam_reconfig_ts = []
    cam_reconfig = []
    if reconfig:
        cam_reconfig_ts = np.array_split(np.arange(timestep), divide_conf)
        cam_reconfig = [[0, 1, 2, 3], [1, 2, 3], random.sample([0, 1, 2, 3], 3), [0, 2], [1, 3]]
    for i in range(timestep):
        meas_cam = []
        meas_cam_idxoff = []
        # imshow = np.zeros((h * 2, w * 2, 3), dtype="uint8")
        for s, det in enumerate(dets):
            conf_indices = det[str(i)][:, 4] > conf
            bboxes = det[str(i)][conf_indices, :4]
            feats = det[str(i)][conf_indices, 4:]  # confidence (4) & features (5:end)

            # img0 = cv2.imread(imgdirs[s][i])
            # for ibbox, bbox in enumerate(bboxes):
            #     l, t = bbox[0], bbox[1]
            #     r, b = bbox[2], bbox[3]
            #     # draw bbox
            #     img0 = cv2.circle(img0, (l, t), radius=8, color=(255, 255, 255), thickness=-1)
            #     img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
            #     img0 = cv2.putText(img0, str(ibbox), org=(l, t), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            #                        color=(0, 255, 255), thickness=2)
            #     img0 = cv2.putText(img0, "CAM " + str(s) + " Frame " + str(i), org=(100, 100),
            #                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)
            # start_h, end_h, start_w, end_w = int(s / 2) * h, int(s / 2 + 1) * h, (s % 2) * w, (s % 2 + 1) * w
            # imshow[start_h: end_h, start_w: end_w, :] = img0

            bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # convert ltrb to ltwh
            feet_loc = np.copy(bboxes[:, 0:2])
            feet_loc[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
            feet_loc[:, 1] = bboxes[:, 1] + bboxes[:, 3]
            feet_loc_gp = homtrans(np.linalg.inv(model.cam_mat[s][:, [0, 1, 3]]), feet_loc.T)
            ind_y = np.logical_and(feet_loc_gp[1, :] <= model.YMAX[1], feet_loc_gp[1, :] >= model.YMAX[0])
            ind_x = np.logical_and(feet_loc_gp[0, :] <= model.XMAX[1], feet_loc_gp[0, :] >= model.XMAX[0])
            indices = np.nonzero(np.logical_and(ind_x, ind_y))[0]
            soff = False
            if reconfig:
                conf_idx = 0
                for idxconf in range(divide_conf):
                    if i in cam_reconfig_ts[idxconf]:
                        conf_idx = idxconf
                        break
                if not (s in cam_reconfig[conf_idx]):
                    indices = np.array([], dtype=bool)
                    soff = True
            bboxes = bboxes[indices, :]
            bboxes[:, [2, 3]] = np.log(bboxes[:, [2, 3]])  # log extent
            meas_cam.append(np.column_stack((bboxes, feats[indices, :])).T)
            meas_cam_idxoff.append(soff)
        meas.append(meas_cam)
        meas_soff.append(meas_cam_idxoff)
        # scale_percent = 0.6  # percent of original size
        # dim = (int(imshow.shape[1] * scale_percent), int(imshow.shape[0] * scale_percent))
        # resized = cv2.resize(imshow, dim, interpolation=cv2.INTER_AREA)  # resize image
        # cv2.imshow('Image', resized)
        # cv2.moveWindow('Image', 100, 100)
        # cv2.waitKey(100)
    # List of measurement: [0] detection for all cameras, [1] detection for all cameras, etc.
    if reconfig:
        return meas, meas_soff, imgdirs
    return meas, imgdirs
    # End


if __name__ == '__main__':
    dataset = "CMC5"
    model_params = model(dataset)
    load_detection(model_params, dataset)
