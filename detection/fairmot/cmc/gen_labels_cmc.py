'''
https://github.com/ifzhang/FairMOT
You can train FairMOT on custom dataset by following several steps bellow:
1. Generate one txt label file for one image. Each line of the txt label file represents one object. The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". You can modify src/gen_labels_16.py to generate label files for your custom dataset.
2. Generate files containing image paths. The example files are in src/data/. Some similar code can be found in src/gen_labels_crowd.py
3. Create a json file for your custom dataset in src/lib/cfg/. You need to specify the "root" and "train" keys in the json file. You can find some examples in src/lib/cfg/.
4. Add --data_cfg '../src/lib/cfg/your_dataset.json' when training.
'''
import glob
import os.path as osp
import os
import shutil

import numpy as np
from numpy.random import default_rng


def mkdirs(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    if not osp.exists(d):
        os.makedirs(d)


def labels_with_ids():
    seq_root = 'D:/dataset/tracking/CMC/data/images'
    label_root = 'D:/dataset/tracking/CMC/data/labels_with_ids/'
    seqs = [s for s in os.listdir(seq_root)]
    mkdirs(label_root)

    for idx, seq in enumerate(seqs):
        for cam_num in [1, 2, 3, 4]:
            # CMC dataset is a four-camera 1920x1024 resolution dataset
            seq_width = 1920
            seq_height = 1024

            gt_txt = osp.join(seq_root, seq, "GT_" + seq + "_Cam" + str(cam_num) + ".txt")
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=' ')
            gt = gt[np.lexsort((gt[:, 0], gt[:, 1]))]  # convert to MOT GT format

            seq_label_root = osp.join(label_root, seq, "Cam_" + str(cam_num))
            mkdirs(seq_label_root)

            for fid, tid, x, y, w, h, _, _, _, _ in gt:
                fid = int(fid)
                tid_curr = int(tid) + 10 ** idx  # use to distinguish IDs among dataset CMC1, CMC2, CMC3, CMC4, CMC5
                x += w / 2
                y += h / 2
                label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)
            # End
        # End
    # End


def gen_image_paths():
    seq_root = '../../data/images/'
    seqs = [s for s in os.listdir(seq_root)]
    if os.path.exists("cmc.train"):
        os.remove("cmc.train")

    for idx, seq in enumerate(seqs):
        for cam_num in [1, 2, 3, 4]:
            dataset_dir = osp.join(seq_root, seq, "Cam_" + str(cam_num) + "/*.png")
            imgs = glob.glob(dataset_dir)
            for img_dir in imgs:
                with open("cmc.train", 'a') as f:
                    f.write(img_dir[len(seq_root):] + '\n')


def gen_image_paths_half():
    seq_root = '../../data/images/'
    seqs = [s for s in os.listdir(seq_root)]
    if os.path.exists("cmc_half.train"):
        os.remove("cmc_half.train")
    rng = default_rng()

    for idx, seq in enumerate(seqs):
        for cam_num in [1, 2, 3, 4]:
            dataset_dir = osp.join(seq_root, seq, "Cam_" + str(cam_num) + "/*.png")
            imgs = glob.glob(dataset_dir)
            half_ind = rng.choice(len(imgs), size=int(len(imgs) / 2), replace=False)
            half_ind = np.sort(half_ind)
            for index in half_ind:
                img_dir = imgs[index]
                with open("cmc_half.train", 'a') as f:
                    f.write(img_dir[len(seq_root):] + '\n')
    #


def gen_train_on_dataset(dataset="CMC4"):
    seq_root = '../../data/images/'
    filename = "cmc_"+dataset+".train"
    if os.path.exists(filename):
        os.remove(filename)
    for cam_num in [1, 2, 3, 4]:
        dataset_dir = osp.join(seq_root, dataset, "Cam_" + str(cam_num) + "/*.png")
        imgs = glob.glob(dataset_dir)
        for img_dir in imgs:
            with open(filename, 'a') as f:
                f.write(img_dir[len(seq_root):] + '\n')


if __name__ == '__main__':
    gen_image_paths()
    gen_image_paths_half()
    # labels_with_ids()
    gen_train_on_dataset()
