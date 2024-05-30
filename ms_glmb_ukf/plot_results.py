import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gen_model import model
from gen_meas import load_detection
import matplotlib.colors as pltcolors


def plot_3d_video(model, est):
    # est, numpy array of MOTChallenge format
    # save_format = '{frame},{id}, -1, -1, -1, -1, -1,{x},{y},{z},{wx},{wy},{h}, {modes}\n'
    size = (1920, 1024)
    out = cv2.VideoWriter('./results/ellipsoid_plot.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
    print("Saving tracking result in video 'ellipsoid_plot.mp4'...")
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    fig.set_size_inches(size[0] / fig.dpi, size[1] / fig.dpi)
    ax = fig.add_subplot(111, projection='3d')
    offset = 0.5
    plt.tight_layout(pad=0)
    for k in range(int(max(est[:, 0]))):
        print("Saved Frame", k)
        targets = est[est[:, 0] == k + 1, :]
        ax.plot([model.XMAX[0], model.XMAX[1]], [model.YMAX[0], model.YMAX[0]])
        ax.plot([model.XMAX[0], model.XMAX[0]], [model.YMAX[0], model.YMAX[1]])
        ax.plot([model.XMAX[1], model.XMAX[1]], [model.YMAX[0], model.YMAX[1]])
        ax.plot([model.XMAX[0], model.XMAX[1]], [model.YMAX[1], model.YMAX[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(model.XMAX[0] - offset, model.XMAX[1] + offset)
        ax.set_ylim3d(model.YMAX[0] - offset, model.YMAX[1] + offset)
        ax.set_zlim3d(model.ZMAX[0], model.ZMAX[1])
        for eidx, tt in enumerate(targets):
            ellipsoid = tt[[7, 8, 9, 10, 11, 12]]
            # Swapping the first and third elements of the BGR tuple.
            # OpenCV uses the BGR color order, while Matplotlib uses the RGB color order.
            color = get_color(tt[1])
            color = tuple(item / 255 for item in color[::-1])
            color = pltcolors.to_hex(color)
            label = str(tt[1]) + ' [' + model.mode_type[int(tt[13])] + ']'
            # index = assigncolor(est.L[k][:, eidx], colorarray)
            # color = colorarray.rgb[:, index]
            # label = str(np.array2string(est.L[k][:, eidx], separator="."))[1:-1] + '[' + est.S[k][eidx][0] + ']'
            # Plotting an ellipsoid
            cx, cy, cz, rx, ry, rz = ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = rx * np.outer(np.cos(u), np.sin(v)) + cx
            y = ry * np.outer(np.sin(u), np.sin(v)) + cy
            z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
            ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color=color)  # input color
            ax.text(cx, cy, cz + rz, label, size=12, color='green')
        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        str_show = 'Frame {}'.format(k)
        img = cv2.putText(img, str_show, org=(img.shape[1] - 150, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.8, color=(255, 0, 0), thickness=2)
        cv2.imshow("Visual", img)
        cv2.waitKey(1)
        out.write(img)
        # plt.show()
        ax.cla()
    plt.close(fig)
    del fig
    cv2.destroyAllWindows()
    out.release()


def get_color(idx):
    idx = (idx + 1) * 50
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def draw_on_images(model, meas, est, imgdirs, k, s, is_bbox3d=False):
    bboxes = np.copy(meas[k][s][:4, :]).T
    bboxes[:, [2, 3]] = np.exp(bboxes[:, [2, 3]])  # exp extent
    bboxes[:, 2:4] = bboxes[:, 2:4] + bboxes[:, 0:2]  # convert ltwh to ltrb
    img0 = cv2.imread(imgdirs[s][k])
    state_3d = est[est[:, 0] == k + 1, :][:, [1, 7, 8, 9, 10, 11, 12]]  # id, x, y, z, wx, wy, h
    for ibbox, bbox in enumerate(bboxes):
        l, t = int(bbox[0]), int(bbox[1])
        r, b = int(bbox[2]), int(bbox[3])
        # draw bbox
        img0 = cv2.circle(img0, (l, t), radius=8, color=(255, 255, 255), thickness=-1)
        img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
        # img0 = cv2.putText(img0, str(ibbox), org=(l, t), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
        #                    color=(0, 255, 255), thickness=2)
    img0 = cv2.putText(img0, "CAM " + str(s) + " Frame " + str(k), org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(0, 0, 255), thickness=2)
    for ibbox, bbox in enumerate(state_3d):
        obj_id = int(state_3d[ibbox, 0])
        cx, cy, cz, rx, ry, rz = state_3d[ibbox, 1:]
        if is_bbox3d:
            vet = np.ones((4, 8))
            vet[:, 0] = [cx - rx, cy - ry, cz - rz, 1]
            vet[:, 1] = [cx + rx, cy + ry, cz - rz, 1]
            vet[:, 2] = [cx - rx, cy + ry, cz - rz, 1]
            vet[:, 3] = [cx + rx, cy - ry, cz - rz, 1]
            vet[:, 4] = [cx - rx, cy - ry, cz + rz, 1]
            vet[:, 5] = [cx + rx, cy + ry, cz + rz, 1]
            vet[:, 6] = [cx - rx, cy + ry, cz + rz, 1]
            vet[:, 7] = [cx + rx, cy - ry, cz + rz, 1]
            temp = np.dot(model.cam_mat[s], vet)
            x_p = (temp[[0, 1], :] / temp[2, :]).astype("int")
            # Define the indices of the points to draw lines between
            indices = [[0, 2], [1, 3], [0, 3], [1, 2],
                       [4, 6], [5, 7], [4, 7], [5, 6],
                       [0, 4], [3, 7], [1, 5], [2, 6]]
            # Draw the lines
            for i, j in indices:
                cv2.line(img0, (x_p[0, i], x_p[1, i]), (x_p[0, j], x_p[1, j]), get_color(obj_id), 2, 2, 0)
            continue
        # generates three (N)-by-(N) matrices so that SURF(X,Y,Z) produces an
        # ellipsoid with center (XC,YC,ZC) and radii XR, YR, ZR
        N = 150
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x = rx * np.outer(np.cos(u), np.sin(v)) + cx
        y = ry * np.outer(np.sin(u), np.sin(v)) + cy
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
        # projection 3d ellipsoid points to image plane
        vet = np.ones((4, N ** 2))  # (x, y, z, 1)
        for idx in range(N):
            tempp = np.row_stack((x[:, idx], y[:, idx], z[:, idx]))
            vet[:3, idx * N:(idx + 1) * N] = tempp
        temp = np.dot(model.cam_mat[s], vet)
        img1_vert = (temp[[0, 1], :] / temp[2, :]).astype("int")

        i_ty = np.argmin(img1_vert[1, :])
        tx, ty = (img1_vert[0, i_ty], img1_vert[1, i_ty])
        img0 = cv2.putText(img0, str(obj_id), org=(tx, ty), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.8, color=(0, 255, 255), thickness=2)

        # Get the indices of the pixels to be modified
        row_indices, col_indices = img1_vert[1, :], img1_vert[0, :]
        # Use boolean indexing to select the pixels that are within the bounds of the image
        valid_rows = (row_indices >= 0) & (row_indices < img0.shape[0])
        valid_cols = (col_indices >= 0) & (col_indices < img0.shape[1])
        valid_pixels = valid_rows & valid_cols
        # Set the values of the valid pixels to 255
        img0[row_indices[valid_pixels], col_indices[valid_pixels], :] = get_color(obj_id)
    # cv2.imwrite("./results/" + str(k) + "_" + str(s) + ".jpg", img0)
    cv2.imshow("Image", img0)
    cv2.waitKey(0)


def build_3d_border_points(model):
    # Projection from 3D line to an image is not linear, we must draw a set of points as a line instead
    # four corner points, (x, y, z=0)
    # [model.XMAX[0], model.YMAX[0]], [model.XMAX[1], model.YMAX[0]]
    # [model.XMAX[0], model.YMAX[1]], [model.XMAX[1], model.YMAX[1]]
    _offset = 0.025
    temp_y = np.arange(model.YMAX[0], model.YMAX[1], _offset)
    line02 = np.ones((4, len(temp_y)))
    line02[0, :] = model.XMAX[0]
    line02[1, :] = temp_y
    line02[2, :] = 0
    temp_x = np.arange(model.XMAX[0], model.XMAX[1], _offset)
    line23 = np.ones((4, len(temp_x)))
    line23[0, :] = temp_x
    line23[1, :] = model.YMAX[1]
    line23[2, :] = 0
    line31 = np.ones((4, len(temp_y)))
    line31[0, :] = model.XMAX[1]
    line31[1, :] = temp_y
    line31[2, :] = 0
    line10 = np.ones((4, len(temp_x)))
    line10[0, :] = temp_x
    line10[1, :] = model.YMAX[0]
    line10[2, :] = 0
    border_line_points = np.column_stack((line02, line23, line31, line10))
    return border_line_points


def making_demo_video(model, dataset, est, method="MV-GLMB-AB", images_root="../../data/images/"):
    show_pose = False  # Make sure Estimated files (CMC4&5) have Upright/Fallen (0/1)
    save_visual_img = False
    store_data_dir = "./results/"
    reconfig = False  # Re-configuration cameras, turn on/off
    reconfig_off30f = False  # Turn off all cameras for 30f from middle of the scene

    # est, numpy array of MOTChallenge format
    # save_format = '{frame},{id}, -1, -1, -1, -1, -1,{x},{y},{z},{wx},{wy},{h}, {modes}\n'
    imgdirs = []
    b_proj_points = []
    border_line_points = build_3d_border_points(model)
    for s in range(1, model.N_sensors + 1):
        imgs = sorted(glob.glob(images_root + dataset + "/Cam_" + str(s) + "/*.png"))
        imgdirs.append(imgs)
        #
        b_points = np.dot(model.cam_mat[s - 1], border_line_points)
        b_points = (b_points[[0, 1], :] / b_points[2, :]).astype("int").T
        b_proj_points.append(b_points)

    cam_reconfig = [[0, 1, 2, 3], [1, 2, 3], [0, 1, 3], [0, 2], [1, 3]]  # known in advance when drawing
    cam_reconfig_ts = np.array_split(np.arange(len(imgdirs[0])), len(cam_reconfig))

    size = (1920, 1024)  # (1920, 1080)
    scale_percent = 0.35  # percent of original size
    video_size = (size[0] * 2 + 1050, size[1] * 2)  # (size[0] * 3 + 1080, size[1] * 2)
    scaled_size = (int(video_size[0] * scale_percent), int(video_size[1] * scale_percent))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # openh264-1.7.0-win64.dll
    out = cv2.VideoWriter(store_data_dir + dataset + "_" + method + ".avi", fourcc, 4, scaled_size)

    print("Saving tracking result in video 'ellipsoid_plot.mp4'...")
    fig = plt.figure(figsize=(15, 9))
    fig.set_size_inches(size[0] / fig.dpi, size[1] / fig.dpi)
    ax = fig.add_subplot(111, projection='3d')
    offset = 0.5
    plt.tight_layout(pad=0)
    for k in range(int(max(est[:, 0]))):
        print("Saved Frame", k)
        targets = est[est[:, 0] == k + 1, :]
        ax.plot([model.XMAX[0], model.XMAX[1]], [model.YMAX[0], model.YMAX[0]])
        ax.plot([model.XMAX[0], model.XMAX[0]], [model.YMAX[0], model.YMAX[1]])
        ax.plot([model.XMAX[1], model.XMAX[1]], [model.YMAX[0], model.YMAX[1]])
        ax.plot([model.XMAX[0], model.XMAX[1]], [model.YMAX[1], model.YMAX[1]])
        ax.set_xlabel('X', fontsize=25)
        ax.set_ylabel('Y', fontsize=25)
        ax.set_zlabel('Z', fontsize=25)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(axis="z", labelsize=18)
        ax.set_xlim3d(model.XMAX[0] - offset, model.XMAX[1] + offset)
        ax.set_ylim3d(model.YMAX[0] - offset, model.YMAX[1] + offset)
        ax.set_zlim3d(model.ZMAX[0], model.ZMAX[1])  # adjust ZMAX[1] to scale down ellipsoid height
        cam_imgs = []
        img_frame_k = []
        for s in range(model.N_sensors):
            img0 = cv2.imread(imgdirs[s][k])
            # for point in b_proj_points[s]:
            #     cv2.circle(img0, tuple(point), 5, (0, 0, 255))
            img_frame_k.append(img0)
        start_off = int(int(max(est[:, 0])) / 2)
        if start_off < k < start_off + 30 and reconfig_off30f:
            for s in range(model.N_sensors):
                img0 = img_frame_k[s] * 0
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "CAM " + str(s) + " OFF"
                textsize = cv2.getTextSize(text, font, 6, 4)[0]
                # get coords based on boundary
                textX = int((img0.shape[1] - textsize[0]) / 2)
                textY = int((img0.shape[0] + textsize[1]) / 2)
                cv2.putText(img0, text, (textX, textY), font, 6, (255, 255, 255), 4)
                cam_imgs.append(img0)
        if len(targets) == 0:
            for s in range(model.N_sensors):
                cam_imgs.append(img_frame_k[s] * 0)
        for eidx, tt in enumerate(targets):
            ellipsoid = tt[[7, 8, 9, 10, 11, 12]]
            # Swapping the first and third elements of the BGR tuple.
            # OpenCV uses the BGR color order, while Matplotlib uses the RGB color order.
            color = get_color(tt[1])
            color = tuple(item / 255 for item in color[::-1])
            color = pltcolors.to_hex(color)
            label = str(int(tt[1]))  # + ' [' + model.mode_type[int(tt[13])] + ']'
            if show_pose:
                label = str(int(tt[1])) + '[' + model.mode_type[int(tt[13])] + ']'
            # index = assigncolor(est.L[k][:, eidx], colorarray)
            # color = colorarray.rgb[:, index]
            # label = str(np.array2string(est.L[k][:, eidx], separator="."))[1:-1] + '[' + est.S[k][eidx][0] + ']'
            # Plotting an ellipsoid
            cx, cy, cz, rx, ry, rz = ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = rx * np.outer(np.cos(u), np.sin(v)) + cx
            y = ry * np.outer(np.sin(u), np.sin(v)) + cy
            z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
            ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color=color)  # input color
            ax.text(cx, cy, cz + rz, label, size=25, color='green')

            current_config = 0
            for cindex, config_index in enumerate(cam_reconfig_ts):
                if k in config_index:
                    current_config = cindex
                    break
            for s in range(model.N_sensors):
                img0 = img_frame_k[s]
                img0 = cv2.putText(img0, "CAM " + str(s) + " Frame " + str(k), org=(100, 100),
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=2, color=(0, 0, 255), thickness=3)
                # generates three (N)-by-(N) matrices so that SURF(X,Y,Z) produces an
                # ellipsoid with center (XC,YC,ZC) and radii XR, YR, ZR
                obj_id = int(tt[1])
                N = 150
                u = np.linspace(0, 2 * np.pi, N)
                v = np.linspace(0, np.pi, N)
                x = rx * np.outer(np.cos(u), np.sin(v)) + cx
                y = ry * np.outer(np.sin(u), np.sin(v)) + cy
                z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
                # projection 3d ellipsoid points to image plane
                vet = np.ones((4, N ** 2))  # (x, y, z, 1)
                for idx in range(N):
                    tempp = np.row_stack((x[:, idx], y[:, idx], z[:, idx]))
                    vet[:3, idx * N:(idx + 1) * N] = tempp
                temp = np.dot(model.cam_mat[s], vet)
                img1_vert = (temp[[0, 1], :] / temp[2, :]).astype("int")
                i_ty = np.argmin(img1_vert[1, :])
                tx, ty = (img1_vert[0, i_ty], img1_vert[1, i_ty])
                obj_id_str = str(obj_id)
                if show_pose and int(tt[13]) == 1:  # Only show Fallen in images
                    obj_id_str = label
                img0 = cv2.putText(img0, obj_id_str, org=(tx, ty), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=1.5, color=(0, 255, 255), thickness=2)
                # Get the indices of the pixels to be modified
                row_indices, col_indices = img1_vert[1, :], img1_vert[0, :]
                # Use boolean indexing to select the pixels that are within the bounds of the image
                valid_rows = (row_indices >= 0) & (row_indices < img0.shape[0])
                valid_cols = (col_indices >= 0) & (col_indices < img0.shape[1])
                valid_pixels = valid_rows & valid_cols
                # Set the values of the valid pixels to 255
                img0[row_indices[valid_pixels], col_indices[valid_pixels], :] = get_color(obj_id)
                # Save image for visualization
                if (k == start_off or k == start_off + 35) and s == 2 and save_visual_img:
                    cv2.imwrite(store_data_dir + dataset + "_f" + str(k) + "_cam" + str(s) + ".png", img0)
                if reconfig and (s not in cam_reconfig[current_config]):
                    img0 = np.zeros_like(img_frame_k[s], dtype="uint8")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = "CAM " + str(s) + " OFF"
                    textsize = cv2.getTextSize(text, font, 6, 4)[0]
                    # get coords based on boundary
                    textX = int((img0.shape[1] - textsize[0]) / 2)
                    textY = int((img0.shape[0] + textsize[1]) / 2)
                    cv2.putText(img0, text, (textX, textY), font, 6, (255, 255, 255), 4)
                cam_imgs.append(img0)
        ax.view_init(None, None)  # adjust point of view, default (30, -60)
        fig.canvas.draw()  # redraw the canvas
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # convert canvas to image
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = img[:, 450:1500, :]  # 1530 (WT)
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        str_show = '3D View, Frame {}'.format(k)
        img_3d = cv2.putText(img, str_show, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=2, color=(255, 0, 0), thickness=3)
        # Save image for visualization
        if (k == start_off or k == start_off + 35) and save_visual_img:
            cv2.imwrite(store_data_dir + dataset + "_f" + str(k) + "_3d.png", img_3d)
        ax.view_init(90, -90)  # adjust point of view, Bird eye view
        fig.canvas.draw()  # redraw the canvas
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # convert canvas to image
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = img[:, 450:1500, :]  # 1530 (WT)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        str_show = 'Bird Eye View, Frame {}'.format(k)
        img_bird = cv2.putText(img, str_show, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=2, color=(255, 0, 0), thickness=3)
        x, y, w, h = 120, 900, 800, 100  # 120, 950, 850, 100
        img_bird = cv2.rectangle(img_bird, (x, y), (x + w, y + h), (36, 255, 12), 3)
        img_bird = cv2.putText(img_bird, dataset + ": " + method, org=(x + 60, y + 70),  # x + 10
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 250), thickness=3)
        ax.cla()

        img_first_row = np.concatenate((cam_imgs[0], cam_imgs[1], img_3d), axis=1)
        img_second_row = np.concatenate((cam_imgs[2], cam_imgs[3], img_bird), axis=1)
        imshow = np.concatenate((img_first_row, img_second_row), axis=0)
        dim = (int(imshow.shape[1] * scale_percent), int(imshow.shape[0] * scale_percent))
        resized = cv2.resize(imshow, dim, interpolation=cv2.INTER_AREA)  # resize image
        cv2.imshow('Image', resized)
        cv2.waitKey(100)
        out.write(resized)
    plt.close(fig)
    del fig
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    dataset = "CMC1"
    model_params = model(dataset)
    # meas, imgdirs = load_detection(model_params, dataset)
    est = np.loadtxt("../experiments/demo_videos/EST_CMC10_3_WORLD_CENTROID.txt", delimiter=' ')
    making_demo_video(model_params, dataset, est, "MV-GLMB-AB")
