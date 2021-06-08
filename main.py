import os
if True:
    os.environ['OMP_NUM_THREADS'] = "70"
from datetime import datetime
import sys
if True:
    sys.path.append("..")
import glob
import random
if True:
    random.seed(10)
from multiprocessing import Pool

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
    all_files_paths = all_files_paths[:10]
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

    def calc_dist(i):
        dist = []
        for j in range(n_frames):
            dist.append(wasserstein_distance(frames[i], frames[j]))
        return dist, i

    jobs = 64
    with Pool(jobs) as p:
        res = list(tqdm(p.imap(calc_dist, range(n_frames)),
                   desc='num frames', total=n_frames))
    all_list = np.stack(res[k][0] for k in range(len(res)))
    all_i = np.stack(res[k][1] for k in range(len(res)))
    distance[all_i] = all_list

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
# sample_weight = Counter(y_true)
# sample_weight = np.array([1./float(sample_weight[p]) for p in y_true])
# sample_weight_g = Counter(y_true_g)
# sample_weight_g = np.array([1./float(sample_weight_g[p]) for p in y_true_g])

plot_CM(model_name)

print(" ====== t-SNE and PCA ======")

plot_PCA(model_name, tsne=True)
