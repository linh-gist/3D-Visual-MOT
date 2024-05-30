import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import os


def read_intrinsic_extrinsic(intr, extr):
    intr_CVLab = cv2.FileStorage(intr, cv2.FILE_STORAGE_READ)
    matrix_intr = intr_CVLab.getNode("camera_matrix").mat()

    extr_CVLab = cv2.FileStorage(extr, cv2.FILE_STORAGE_READ)
    rvec_seq = extr_CVLab.getNode("rvec")
    tvec_seq = extr_CVLab.getNode("tvec")
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    for i in range(rvec_seq.size()):
        rvec[i] = rvec_seq.at(i).real()  # rvec
        tvec[i] = tvec_seq.at(i).real()  # tvec
    # Converts a rotation matrix to a rotation vector or vice versa.
    rmat = cv2.Rodrigues(np.array(rvec))[0]
    extrinsic_cam = np.column_stack((rmat, tvec / 100))  # tvec/100 from meter to millimeter

    # A way of projection image pixel at location [302.04425, 242.62845] to z=0 world coordinate
    # Similar to gen_meas.py > homtrans
    # img_pixel = np.array([302.04425, 242.62845, 1])
    # r_intri_inv = np.matmul(np.linalg.inv(rmat), np.linalg.inv(matrix_intr))
    # cam_coor = np.matmul(r_intri_inv, img_pixel)
    # rinv_tvec = np.matmul(np.linalg.inv(rmat), tvec)
    # world_loc = cam_coor * rinv_tvec[2] / cam_coor[2] - rinv_tvec

    # The camera (projection 3D to image) matrix is computed as follows:
    # camMatrix = K * [rotationMatrix; translationVector], where K is the intrinsic matrix.
    cam_mat = np.dot(matrix_intr, extrinsic_cam)
    sensor_pos = -np.dot(rmat.T, tvec[:, np.newaxis] / 100).flatten()  # C = -R^T * T, note R^T = inv(R)
    return sensor_pos, cam_mat


def compute_wildtrack_cam_mat(data_root="./"):
    int_dir = os.path.join(data_root, "./calibrations/intrinsic_zero/")
    ext_dir = os.path.join(data_root, "./calibrations/extrinsic/")
    pos1, cam_mat1 = read_intrinsic_extrinsic(int_dir + "intr_CVLab1.xml", ext_dir + "extr_CVLab1.xml")
    pos2, cam_mat2 = read_intrinsic_extrinsic(int_dir + "intr_CVLab2.xml", ext_dir + "extr_CVLab2.xml")
    pos3, cam_mat3 = read_intrinsic_extrinsic(int_dir + "intr_CVLab3.xml", ext_dir + "extr_CVLab3.xml")
    pos4, cam_mat4 = read_intrinsic_extrinsic(int_dir + "intr_CVLab4.xml", ext_dir + "extr_CVLab4.xml")
    pos5, cam_mat5 = read_intrinsic_extrinsic(int_dir + "intr_IDIAP1.xml", ext_dir + "extr_IDIAP1.xml")
    pos6, cam_mat6 = read_intrinsic_extrinsic(int_dir + "intr_IDIAP2.xml", ext_dir + "extr_IDIAP2.xml")
    pos7, cam_mat7 = read_intrinsic_extrinsic(int_dir + "intr_IDIAP3.xml", ext_dir + "extr_IDIAP3.xml")
    list_cammat = [cam_mat1, cam_mat2, cam_mat3, cam_mat4, cam_mat5, cam_mat6, cam_mat7]
    list_pos = [pos1, pos2, pos3, pos4, pos5, pos6, pos7]
    return list_pos, list_cammat


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color[0] / 255, color[1] / 255, color[2] / 255


def load_gt():
    json_root = "../../../data/images/wildtrack/annotations_positions"
    files = sorted(glob.glob(json_root + '/*.json'))
    fig, ax = plt.subplots()
    # save_format = '{frame},{id}, -1, -1, -1, -1, -1,{x},{y},{z},{wx},{wy},{h}\n'
    gt_list = []
    gt_views_bboxes = [[] for i in range(7)]  # number of sensors is seven
    card = []  # cardinality
    for i, path in enumerate(files):
        ax.set_xlim(-3.0, 9)  # [-3.0, 8.975000000000001]
        ax.set_ylim(-9, 26)  # [-8.994114583333333, 25.737604166666664]
        with open(path, 'r') as f:
            data = json.load(f)
            card.append([i, len(data)])
            for person in data:
                positionID = person['positionID']
                xi = -3.0 + 0.025 * (positionID % 480)
                yi = -9.0 + 0.025 * (positionID // 480)
                personID = person['personID']
                # if personID == 358 and i > 189:
                #     personID = 35822
                # if personID == 370 and (i * 5 in [950, 955]):
                #     continue
                ax.plot(xi, yi, ls="", marker="o", color=get_color(personID))
                ax.annotate(str(personID), xy=(xi, yi))
                line = [i + 1, personID, -1, -1, -1, -1, -1, xi, yi, 00, 00, 00, 00]  # restricted to the ground plane
                gt_list.append(line)
                # MOT <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                views = person['views']
                for v_s in views:
                    s = v_s['viewNum']
                    xmax = v_s['xmax']
                    xmin = v_s['xmin']
                    ymax = v_s['ymax']
                    ymin = v_s['ymin']
                    if xmax < 0 or ymax < 0:  # object does not appear in this sensor
                        continue
                    line_bboxes = [i + 1, personID, xmin, ymin, xmax - xmin, ymax - ymin, -1, -1, -1, -1]
                    gt_views_bboxes[s].append(line_bboxes)
        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        str_show = 'Frame {}'.format(i * 5)
        img = cv2.putText(img, str_show, org=(img.shape[1] - 180, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.8, color=(255, 0, 0), thickness=2)
        cv2.imshow("Visual", img)
        cv2.imwrite("./results/" + path.split('\\')[-1].split('.')[0] + '.jpg', img)
        cv2.waitKey(1)
        ax.cla()
    x_frame, y_card = np.array(card).T
    plt.ylim(0, 40)  # maximum number of object is 40, at frame 00000045, 00000940, 00000945, 00000950.png
    plt.plot(x_frame, y_card, '-g')
    plt.ylabel("Number of Objects")
    plt.xlabel("Frame")
    plt.savefig('./results/GT_cardinality.pdf')
    # plt.show()
    plt.close()
    for i, gt_bboxes in enumerate(gt_views_bboxes):
        np.savetxt("./results/GT_WILDTRACK_Cam" + str(i + 1) + ".txt", gt_bboxes, fmt='%i', delimiter=' ')
    gt = np.array(gt_list)
    np.savetxt('./results/GT_WILDTRACK_WORLD_CENTROID.txt', gt, delimiter=' ')  # motchallenge format


def read_calibration():
    mtx_list = []
    data_root = "./calibrations/intrinsic_original/"

    def read_data(filename):
        intr_CVLab = cv2.FileStorage(data_root + filename, cv2.FILE_STORAGE_READ)
        mtx = intr_CVLab.getNode("camera_matrix").mat()
        dist = intr_CVLab.getNode("distortion_coefficients").mat()
        return mtx, dist

    mtx_list.append(read_data("intr_CVLab1.xml"))
    mtx_list.append(read_data("intr_CVLab2.xml"))
    mtx_list.append(read_data("intr_CVLab3.xml"))
    mtx_list.append(read_data("intr_CVLab4.xml"))
    mtx_list.append(read_data("intr_IDIAP1.xml"))
    mtx_list.append(read_data("intr_IDIAP2.xml"))
    mtx_list.append(read_data("intr_IDIAP3.xml"))

    return mtx_list


def undistort(video_name, camera_matrix, save_image=False):
    data_root = "../../../data/images/wildtrack/"
    mtx, dist = camera_matrix

    file_subsets = sorted(glob.glob(os.path.join(data_root, "Image_subsets", video_name, '*.png')))
    index_gt = 1  # ignore the first GT (not matched found in video)
    img_gt = cv2.imread(file_subsets[index_gt])  # image provided in Ground Truth (2 FPS)

    vidcap = cv2.VideoCapture(os.path.join(data_root, "video", video_name + ".mp4"))
    success, image = vidcap.read()
    count = 0
    video_fps = 60  # WILDTRACK provides video 60 FPS
    det_fps = 6  # video_fps/det_fps can be any divisor of video_fps/gt_fps -> (60, 30, 20, 12, 10, 6, 4, 2)
    gt_fps = 2  # Ground Truth provided by WILDTRACK
    while success:
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        img_subtract = cv2.subtract(dst, img_gt)
        if np.amax(img_subtract[:, 0, 0]) < 10:  # consider two images are similar
            break
        success, image = vidcap.read()
        count += 1
    gt_indices = np.arange(1, len(file_subsets))
    raw_indices = np.arange(count, (len(file_subsets) - 1) * (video_fps / gt_fps), video_fps / gt_fps)
    det_indices = np.arange(count, max(raw_indices) + 1, video_fps / det_fps)
    save_indices = os.path.join(data_root, "video", video_name + "_indices.npz")
    dict_indices = dict()
    dict_indices["gt_indices"] = gt_indices
    dict_indices["det_indices"] = det_indices
    dict_indices["raw_indices"] = raw_indices
    # we only take subset of det_indices (where it is in raw_indices GT) in the final result
    dict_indices["detraw_indices"] = np.nonzero(np.isin(det_indices, raw_indices))
    np.savez(save_indices, **dict_indices)

    if not save_image:
        return

    files = glob.glob(data_root + "/video/images/*.png")
    for i in files:
        os.remove(i)

    # Visualization
    vidcap = cv2.VideoCapture(os.path.join(data_root, "video", video_name + ".mp4"))
    success, image = vidcap.read()
    count = 0
    det_index = 0
    while success:
        # h, w = image.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        # from skimage.metrics import structural_similarity
        # first_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # second_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        # score, diff = structural_similarity(first_gray, second_gray, full=True)
        # print("Similarity Score: {:.3f}%".format(score * 100))
        img_subtract = cv2.subtract(dst, img_gt)
        if np.amax(img_subtract[:, 0, 0]) < 10:  # consider two images are similar
            index_gt += 1
            if index_gt >= len(file_subsets):
                break
            img_gt = cv2.imread(file_subsets[index_gt])

        if det_indices[det_index] == count and save_image:
            cv2.imwrite(data_root + "/video/images/frame%d.png" % count, dst)  # save frame as JPEG file
            det_index += 1

        success, image = vidcap.read()
        count += 1
        cv2.imshow("Image Subtract", img_subtract)
        cv2.waitKey(1)
    # END


if __name__ == '__main__':
    # cam_mat = read_calibration()
    # for s in range(7):
    #     print("Undistorted sensor index", s)
    #     undistort("C" + str(s + 1), cam_mat[s])
    load_gt()
