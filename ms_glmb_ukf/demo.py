import os
import pandas as pd

from gen_model import model
# from gen_truth import load_gt
from gen_meas import load_detection
from plot_results import plot_3d_video, making_demo_video
from run_filter import GLMB
from ospa2 import ospa2_single_dataset, ospa2_datasets
from clearmot import clearmot_single_dataset, clear_mot
import multiprocessing


def dataset_eval(dataset, adaptive_birth, use_feat, exp_idx="", root_dir="./results"):
    model_params = model(dataset)
    meas, img_dirs = load_detection(model_params, dataset)
    glmb = GLMB(model_params, adaptive_birth, use_feat)
    glmb.run(model_params, dataset, meas)
    # glmb.runcpp(model_params, dataset, meas, adaptive_birth, use_feat)
    glmb.save_est_motformat(root_dir, dataset + str(exp_idx) + "_" + str(adaptive_birth) + "_" + str(use_feat))


def run_tests(dataset, start, end, adaptive_birth, use_feat):
    root_results_dir = "./results"
    gt_data_dir = "../../data/images/"
    processes = []
    for exp_idx in range(start, end):
        p = multiprocessing.Process(target=dataset_eval, args=(dataset, adaptive_birth, use_feat, exp_idx))
        processes.append(p)
        p.start()
    print("Waiting all processes to be finished................")
    for process in processes:
        process.join()
    gt_list = []
    est_list = []
    dataset_list = []
    for exp_idx in range(start, end):
        exp_result = dataset + str(exp_idx) + "_" + str(adaptive_birth) + "_" + str(use_feat)
        gt_file = os.path.join(gt_data_dir, dataset, "GT_" + dataset + "_WORLD_CENTROID.txt")
        np_gt = pd.read_csv(gt_file, delimiter=' ', header=None).to_numpy()
        est_file = os.path.join(root_results_dir, "EST_" + exp_result + "_WORLD_CENTROID.txt")
        est = pd.read_csv(est_file, delimiter=' ', header=None).to_numpy()
        gt_list.append(np_gt)
        est_list.append(est)
        dataset_list.append(exp_result)
    ospa2_strsummary = ospa2_datasets(gt_list, est_list, dataset_list)
    print(ospa2_strsummary)
    strsummary = clear_mot(gt_list, est_list, dataset_list)
    print(strsummary)
    with open(os.path.join(root_results_dir, 'summary_clearmot.txt'), 'w') as f:
        f.write(strsummary)


if __name__ == '__main__':
    dataset = "CMC1"  # CMC1, CMC2, CMC3, CMC4, CMC5, WILDTRACK
    birth_opt = 3  # (0[Fix birth], 1[Monte Carlo (AB)], 2[KMeans (AB)], 3[MeanShift (AB)])
    model_params = model(dataset)
    meas, img_dirs = load_detection(model_params, dataset)

    # Choose the following options (adaptive_birth = 0 [Fix birth], 1 [Monte Carlo], 2 [MeanShift])
    # (0) Fix birth uses re-id feature => [adaptive_birth=0, use_feat=False] => ONLY for CMC dataset
    # (1.1) Monte Carlo Adaptive birth uses re-id feature [adaptive_birth=1, use_feat=True]
    # (1.2) Monte Carlo Adaptive birth does NOT use re-id feature [adaptive_birth=1, use_feat=False]
    # (2.1) KMeans Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
    # (2.2) KMeans Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
    # (3.1) MeanShift Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
    # (3.2) MeanShift Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
    glmb = GLMB(model_params, adaptive_birth=birth_opt, use_feat=True)
    est_params = glmb.run(model_params, dataset, meas)
    # est_params = glmb.runcpp(model_params, dataset, meas, adaptive_birth=birth_opt, use_feat=True)
    clearmot_single_dataset(est_params, dataset)
    ospa2_single_dataset(est_params, dataset)
    # plot_3d_video(model_params, est_params)
    making_demo_video(model_params, dataset, est_params)

    # Running repeated experiments parallelly
    # datasets = ["CMC1", "CMC2", "CMC3", "CMC4", "CMC5", "WILDTRACK"]
    # start, end = 0, 25
    # for dataset in datasets:
    #     birth_opt = 1
    #     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=False)
    #     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=True)
    #
    #     birth_opt = 2
    #     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=False)
    #     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=True)
    #
    #     birth_opt = 3
    #     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=False)
    #     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=True)
