import os
if True:
    os.environ['OMP_NUM_THREADS'] = "70"
from datetime import datetime
import sys
if True:
    sys.path.append("..")
import random
if True:
    random.seed(10)
from multiprocessing import Pool
from itertools import combinations, islice


import torch
import numpy as np
import matplotlib.pyplot as plt
if True:
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = (20.0, 4.0)
import seaborn as sns
if True:
    sns.set_theme()
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from datasets import create_datasets
from plot import plot_PCA, plot_CM


# dist = 'norm2'
dist = 'wasserstein'
label = 'pseudotime'
model_name = "supervised_histogramme_synthetic_{}_{}".format(dist, label)
data_type = 'synthetic'
test = False
task_type = 'supervised_hist'
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
model_path = "/data/biocomp/qin/results/baseline_{}/{}/".format(data_type,
                                                                model_name)
result_path = model_path + model_name + "_"


# FUNCTIONS
def general_result(y_true, y_pred, n=5):
    k = 13/n
    _y_true = ((y_true-2)/k).astype(int)
    _y_pred = ((y_pred-2)/k).astype(int)
    return _y_true, _y_pred


# GET DATASET  (this part is different for different data type)
print(" ====== LOAD DATA ====== ")
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
if not os.path.exists(model_path):
    os.mkdir(model_path)

with open('./settings/synthetic_val_paths.txt') as f:
    all_files_paths = f.read().splitlines()
if test:
    all_files_paths = all_files_paths[:3]
print('all_files_paths', len(all_files_paths))


frames, y_true = create_datasets(all_files_paths, data_type, label)
print('frames are of shape', frames.shape)
print('labels are of shape', y_true.shape)


print(" ====== CALCULATION ====== ")
y_pred = []

norm_hists = frames.pow(2).sum(1)
# dis 2
if dist == 'norm2':
    distance = norm_hists.unsqueeze(1) + norm_hists.unsqueeze(0) - \
               2. * torch.mm(frames, frames.permute(1, 0))
# dis Wasserstein
else:
    n_frames = frames.shape[0]
    distance = np.zeros((n_frames, n_frames))

    def calc_dist(args):
        dist = []
        for arg in args:
            dist.append([wasserstein_distance(arg[0][0], arg[1][0]),
                         arg[0][1], arg[1][1]])
        return dist

    # def calc_dist(args):
    #     frame_i, i, frame_j, j = args
    #     dist = wasserstein_distance(frame_i, frame_j)
    #     return dist, i, j

    def split_every(n, iterable):
        i = iter(iterable)
        piece = list(islice(i, n))
        # piece = [(frames[k[0]], k[0], frames[k[1]], k[1]) for k in piece]
        while piece:
            yield piece
            piece = list(islice(i, n))
            # piece = [(frames[k[0]], k[0], frames[k[1]], k[1]) for k in piece]

    print('create iteration', frames.shape[0])
    r = iter(combinations(zip(frames, range(frames.shape[0])), 2))
    n = 10000
    print('create iteration list')
    r_lst = split_every(n, r)
    print('begin calculation')
    
    jobs = 1
    with Pool(jobs) as p:
        res = list(tqdm(p.imap(calc_dist, r_lst),
                        total=int(n_frames*(n_frames-1)/2/n)))
    all_list = np.concatenate([[res_[j][0] for j in range(len(res_))] for res_ in res])
    all_i = np.concatenate([[res_[j][1] for j in range(len(res_))] for res_ in res])
    all_j = np.concatenate([[res_[j][2] for j in range(len(res_))] for res_ in res])
    for k, (i, j) in enumerate(zip(all_i, all_j)):
        distance[i, j] = distance[j, i] = all_list[k]

    # for r_ in r_lst:
    #     r_ = [(frames[k[0]], k[0], frames[k[1]], k[1]) for k in r_]
    #     
    #     jobs = 128
    #     with Pool(jobs) as p:
    #         res = list(tqdm(p.imap(calc_dist, r_), total=len(r_)))
    #     all_list = np.stack([res[k][0] for k in range(len(res))])
    #     all_i = np.stack([res[k][1] for k in range(len(res))])
    #     all_j = np.stack([res[k][2] for k in range(len(res))])
    #     for k, (i, j) in enumerate(zip(all_i, all_j)):
    #         distance[i, j] = distance[j, i] = all_list[k]


    # for i in tqdm(range(n_frames), total=n_frames):
    #     frame_i = frames[i]
    # def distance_i(frame_i, i):
    #     def calc_dist(args):
    #         frame_j, j = args
    #         dist = wasserstein_distance(frame_i, frame_j)
    #         return dist, j
    #     jobs = 128
    #     with Pool(jobs) as p:
    #         res = list(p.imap(calc_dist, zip(frames, 
    #                                          range(len(frames)))))
    #     all_list = np.stack(res[k][0] for k in range(len(res)))
    #     all_i = np.stack(res[k][1] for k in range(len(res)))
    #     distance[i] = all_list[all_i]
    # new_jobs = 4
    # with Pool(new_jobs) as p:
    #     res = list(tqdm(p.imap(distance_i, zip(frames, 
    #                                       range(len(frames)))),
    #                     total=n_frames)

    # for i in tqdm(range(n_frames), total=n_frames):
    #     for j in range(n_frames):
    #         distance[i, j] = wasserstein_distance(frames[i], frames[j])


for i in range(distance.shape[0]//51):
    distance[i*51:(i+1)*51, i*51:(i+1)*51] = np.inf
y_pred = distance.argmin(1)
y_true = np.array(y_true)
y_pred = y_true[y_pred]

y_true_g, y_pred_g = general_result(y_true=np.array(y_true),
                                    y_pred=np.array(y_pred), n=5)

np.savez(result_path+'calculation.npz', hists=frames,
         distance=distance, y_true=y_true, y_pred=y_pred,
         y_true_g=y_true_g, y_pred_g=y_pred_g,
         val_paths=[path.split('/')[-1].split('.')[0]
                    for path in all_files_paths])

print(" ====== CONFUSION MATRIX ======")
# CONFUSION MATRIX

plot_CM(model_name)

print(" ====== t-SNE and PCA ======")

plot_PCA(model_name, tsne=True)
