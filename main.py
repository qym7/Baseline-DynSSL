# # dependence
# from IPython.lib.deepreload import reload
# %load_ext autoreload
# %autoreload 2

import os
if True:
    os.environ['OMP_NUM_THREADS'] = "70"
from datetime import datetime
import sys
sys.path.append("..")
import glob
import random
from collections import Counter
random.seed(10)
from multiprocessing import Pool

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# %matplotlib inline
plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = (20.0, 4.0) 
import seaborn as sns; sns.set_theme()
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from scipy.stats import wasserstein_distance
from torchvision import transforms
from PIL import Image

from datasets import create_datasets
from plot import plot_PCA, plot_CM



dist = 'norm2'
# dist = 'wasserstein' 
label = 'pseudotime'
model_name = "supervised_histogramme_dataset05_{}_{}".format(dist, label)
data_type = 'synthetic05'
# inverse = 'inv_False' not in model_name and 'hist' not in model_name and 'mean' not in model_name
test = False
data_size = 11 if test else '*'
task_type = 'supervised_hist'
print('data_size is', data_size)
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
result_path = "/data/biocomp/qin/results/baseline_{}/{}/".format(data_type, model_name) + model_name + "_"


# GET DATASET  (this part is different for different data type)
print(" ====== LOAD DATA ====== ")
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
try:
    os.mkdir('/data/biocomp/qin/results/baseline_{}/'.format(data_type)+model_name)
except:
    ()

dataset_paths = '/data/biocomp/qin/datasets/synthetic/Dataset_05/*.npz'
print(dataset_paths)
all_files_paths = glob.glob(dataset_paths)
if data_size != '*':
    all_files_paths = random.choices(all_files_paths, k=data_size) # 221
else:
    val_indices = np.load('/data/biocomp/qin/datasets/synthetic/val_indices_{}.npy'.format(data_type))
    all_files_paths = list(np.array(all_files_paths)[val_indices])
print('all_files_paths', len(all_files_paths))

frames, y_true = create_datasets(all_files_paths, data_type, label)
print('frames are of shape', frames.shape)
print('labels are of shape', y_true.shape)


# comparison = comparison.astype(int)

print(" ====== CALCULATION ====== ")

y_pred = []

norm_hists= frames.pow(2).sum(1)
# dis 2
if dist == 'norm2':
    distance = norm_hists.unsqueeze(1) + norm_hists.unsqueeze(0) - 2. * torch.mm(frames, frames.permute(1,0))
# dis W
else:
    n_frames = frames.shape[0]
    distance = np.zeros((n_frames, n_frames))
#     for i in range(n_frames):
#         for j in range(n_frames):
#             distance[i,j] = wasserstein_distance(frames[i], frames[j])
    def calc_dist(i):
        dist = []
        for j in range(n_frames):
            dist.append(wasserstein_distance(frames[i], frames[j]))
        return dist, i
    
    #res = []
    #for i in range(n_frames):
    #    dist, _i = calc_dist(i)
    #    res.append((dist, _i))
    #all_list = np.stack(res[k][0] for k in range(len(res)))
    #all_i = np.stack(res[k][1] for k in range(len(res)))
    #distance[all_i] = all_list
    jobs = 64
    with Pool(jobs) as p:
        res = list(tqdm(p.imap(calc_dist, range(n_frames)),
                   desc='num frames', total=n_frames))
    all_list = np.stack(res[k][0] for k in range(len(res)))
    all_i = np.stack(res[k][1] for k in range(len(res)))
    distance[all_i] = all_list
# for i in range(n_frames):
#     print(i)
#     def calc_dist(j):
#         distance[i,j] = wasserstein_distance(frames[i], frames[j])
#    jobs = 64
#    with Pool(jobs) as p:
#        res = list(tqdm(p.imap(calc_dist, range(n_frames)),
#            desc='num frames', total=n_frames))
# for j in range(n_frames):
#     distance[i,j] = wasserstein_distance(frames[i], frames[j])

for i in range(distance.shape[0]//51):
    distance[i*51:(i+1)*51, i*51:(i+1)*51] = np.inf
y_pred = distance.argmin(1)
y_true = np.array(y_true)
y_pred = y_true[y_pred]

def general_result(y_true, y_pred, n=5):
    k = 13/n
    _y_true = ((y_true-2)/k).astype(int)
    _y_pred = ((y_pred-2)/k).astype(int)
    return _y_true, _y_pred

y_true_g, y_pred_g = general_result(y_true=np.array(y_true), y_pred=np.array(y_pred), n=5)

np.savez(result_path+'calculation.npz', hists=frames, distance=distance, y_true=y_true, y_pred=y_pred, y_true_g=y_true_g, y_pred_g=y_pred_g, val_paths=[path.split('/')[-1].split('.')[0] for path in all_files_paths])

print(" ====== CONFUSION MATRIX ======")
# CONFUSION MATRIX
# sample_weight = Counter(y_true)
# sample_weight = np.array([1./float(sample_weight[p]) for p in y_true])
# sample_weight_g = Counter(y_true_g)
# sample_weight_g = np.array([1./float(sample_weight_g[p]) for p in y_true_g])

plot_CM(model_name)

print(" ====== t-SNE and PCA ======")

plot_PCA(model_name, tsne=True)
