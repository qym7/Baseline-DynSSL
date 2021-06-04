# # dependence
# from IPython.lib.deepreload import reload
# %load_ext autoreload
# %autoreload 2

import os
from datetime import datetime
import socket
import sys
sys.path.append("..")
import glob
import random
from collections import Counter
random.seed(10)
from multiprocessing import Pool

import pickle
from PIL import Image
import cv2
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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



# dist = 'norm2'
dist = 'wasserstein' 
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
y_pred = y_true[y_pred]

def general_result(y_true, y_pred, n=5):
    k = 13/n
    _y_true = ((y_true-2)/k).astype(int)
    _y_pred = ((y_pred-2)/k).astype(int)
    return _y_true, _y_pred

y_true_g, y_pred_g = general_result(y_true=np.array(y_true), y_pred=np.array(y_pred), n=5)

print(" ====== F1 and KENDALL ======")


print('begin')
f = open(result_path + "score.txt", 'w')
f.write("%s"%(datetime.now()))
def evaluate_score(y1, y2, file, title, times=None):
    f1 = f1_score(y1, y2, average=None)
    f1_macro = f1_score(y1, y2, average='macro')
    f1_micro = f1_score(y1, y2, average='micro')
    tau, p_value = stats.kendalltau(y1, y2)
    file.write(" ===== " + title + " ======\n")
    file.write(f"f1 score per class = {f1}\n")
    file.write(f"macro mean f1 score = {f1_macro}\n")
    file.write(f"micro mean f1 score = {f1_micro}\n")
    file.write(f"kendall tau = {tau}\n")
    file.write(f"p value = {p_value}\n")
    file.write(f"R squared measure = {r2_score(y1, y2)}\n")
    if times is not None:
        y1 = np.array(y1)
        y1_sorted = sorted(y1)
        y1_time = y1[(-times).argsort()]
    return tau

print(y_true.shape, y_pred.shape)
tau = evaluate_score(y_true, y_pred, f, "ALL")
tau_g = evaluate_score(y_true_g, y_pred_g, f, "GENERAL")
f.close()

print(" ====== CONFUSION MATRIX ======")
# CONFUSION MATRIX
y_true = np.array(y_true)
# sample_weight = Counter(y_true)
# sample_weight = np.array([1./float(sample_weight[p]) for p in y_true])
# sample_weight_g = Counter(y_true_g)
# sample_weight_g = np.array([1./float(sample_weight_g[p]) for p in y_true_g])


fig, ax = plt.subplots(figsize=(7,7))
cm = confusion_matrix(y_true, y_pred, normalize='true')
ax = sns.heatmap(cm, square=True)
ax.set_title("all result, tau = {:.4f}, acc = {:.4f}".format(tau, cm.diagonal().mean()))
plt.savefig(result_path + "confusion_matrix_true.png")

fig, ax = plt.subplots(figsize=(7,7))
cm = confusion_matrix(y_true, y_pred, normalize='pred')
ax = sns.heatmap(cm, square=True)
ax.set_title("all result, tau = {:.4f}, acc = {:.4f}".format(tau, cm.diagonal().mean()))
plt.savefig(result_path + "confusion_matrix_pred.png")

fig, ax = plt.subplots(figsize=(7,7))
cm = confusion_matrix(y_true, y_pred, normalize='all')
ax = sns.heatmap(cm, square=True)
ax.set_title("all result, tau = {:.4f}, acc = {:.4f}".format(tau, cm.diagonal().sum()))
plt.savefig(result_path + "confusion_matrix_all.png")

fig, ax = plt.subplots(figsize=(7,7))
cm = confusion_matrix(y_true_g, y_pred_g, normalize='all')
ax = sns.heatmap(cm, square=True)
ax.set_title("all result, tau = {:.4f}, acc = {:.4f}".format(tau_g, cm.diagonal().sum()))
plt.savefig(result_path + "general_confusion_matrix_pred.png")

print(" ====== t-SNE and PCA ======")


tsne_2 = TSNE(n_components=2,random_state=33).fit_transform(frames)
tsne_3 = TSNE(n_components=3,random_state=33).fit_transform(frames)

plt.figure(figsize=(5, 5))
plt.scatter(tsne_2[:, 0], tsne_2[:, 1], c=y_true,label="t-SNE", cmap='viridis')
plt.colorbar()
plt.legend()
plt.savefig(result_path+'tsne_outputs_2.png', dpi=120)

plt.figure(figsize = (5, 5))
plt.axes(projection ="3d")
plt.scatter(tsne_3[:, 0], tsne_3[:, 1], tsne_3[:, 2], c=y_true,label="t-SNE", cmap='viridis')
plt.colorbar()
plt.legend()
plt.savefig(result_path+'tsne_outputs_3.png', dpi=120)

pca_2 = PCA(n_components=2)
pca_2_res = pca_2.fit_transform(frames)
pca_3 = PCA(n_components=3)
pca_3_res = pca_3.fit_transform(frames)

plt.figure(figsize=(5, 5))
plt.scatter(pca_2_res[:, 0], pca_2_res[:, 1], c=y_true,label="PCA", cmap='viridis')
plt.colorbar()
plt.xlabel(pca_2.explained_variance_ratio_[0])
plt.ylabel(pca_2.explained_variance_ratio_[1])
plt.legend()
plt.savefig(result_path+'pca_outputs_2.png', dpi=120)

ax = plt.axes(projection ="3d")
ax.scatter(pca_3_res[:, 0], pca_3_res[:, 1], pca_3_res[:, 2], c=y_true,label="PCA", cmap='viridis')
ax.set_xlabel(pca_3.explained_variance_ratio_[0])
ax.set_ylabel(pca_3.explained_variance_ratio_[1])
ax.set_zlabel(pca_3.explained_variance_ratio_[2])
plt.colorbar()
plt.legend()
plt.savefig(result_path + 'pca_outputs_3.png', dpi=120)

