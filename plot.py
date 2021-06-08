import sys
import os
if True:
    os.environ['OMP_NUM_THREADS'] = "70"
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
if True:
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = (20.0, 4.0)
import seaborn as sns
if True:
    sns.set_theme()
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, f1_score, confusion_matrix


def plot_PCA(folder_name, tsne=False):
    result_path = '/data/biocomp/qin/results/baseline_synthetic/' + \
                  folder_name + '/' + folder_name + '_'
    data = np.load(result_path + 'calculation.npz')
    frames = data['hists']
    y_true = data['y_true']

    if tsne:
        tsne_2 = TSNE(n_components=2, random_state=33).fit_transform(frames)
        tsne_3 = TSNE(n_components=3, random_state=33).fit_transform(frames)

        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_2[:, 0], tsne_2[:, 1], c=y_true,
                    label="t-SNE", cmap='viridis')
        plt.colorbar()
        plt.legend()
        plt.savefig(result_path+'tsne_outputs_2.png', dpi=120)

        plt.figure(figsize=(5, 5))
        plt.axes(projection="3d")
        plt.scatter(tsne_3[:, 0], tsne_3[:, 1], tsne_3[:, 2],
                    c=y_true, label="t-SNE", cmap='viridis')
        plt.colorbar()
        plt.legend()
        plt.savefig(result_path+'tsne_outputs_3.png', dpi=120)

    pca_2 = PCA(n_components=2)
    pca_2_res = pca_2.fit_transform(frames)
    pca_3 = PCA(n_components=3)
    pca_3_res = pca_3.fit_transform(frames)

    plt.figure(figsize=(5, 5))
    plt.scatter(pca_2_res[:, 0], pca_2_res[:, 1], c=y_true,
                label="PCA", cmap='viridis')
    plt.colorbar()
    plt.xlabel(pca_2.explained_variance_ratio_[0])
    plt.ylabel(pca_2.explained_variance_ratio_[1])
    plt.legend()
    plt.savefig(result_path+'pca_outputs_2.png', dpi=120)

    ax = plt.axes(projection="3d")
    ax.scatter(pca_3_res[:, 0], pca_3_res[:, 1], pca_3_res[:, 2],
               c=y_true, label="PCA", cmap='viridis')
    ax.set_xlabel(pca_3.explained_variance_ratio_[0])
    ax.set_ylabel(pca_3.explained_variance_ratio_[1])
    ax.set_zlabel(pca_3.explained_variance_ratio_[2])
    plt.colorbar()
    plt.legend()
    plt.savefig(result_path + 'pca_outputs_3.png', dpi=120)


def plot_CM(folder_name):
    print(" ====== F1 and KENDALL ======")
    result_path = '/data/biocomp/qin/results/baseline_synthetic/' + \
                  folder_name+'/' + folder_name + '_'
    data = np.load(result_path + 'calculation.npz')
    y_pred = data['y_pred']
    y_true = data['y_true']
    y_true_g = data['y_true_g']
    y_pred_g = data['y_pred_g']

    f = open(result_path + "score.txt", 'w')
    f.write("%s" % (datetime.now()))

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
            tau_t, _ = stats.kendalltau(y1_sorted, y1_time)
            file.write(f"kendall tau for time = {tau_t}\n")
        return tau

    print(y_true.shape, y_pred.shape)
    tau = evaluate_score(y_true, y_pred, f, "ALL")
    tau_g = evaluate_score(y_true_g, y_pred_g, f, "GENERAL")
    f.close()

    # sample_weight = Counter(y_true)
    # sample_weight = np.array([1./float(sample_weight[p]) for p in y_true])
    # sample_weight_g = Counter(y_true_g)
    # sample_weight_g = np.array([1./float(sample_weight_g[p]) 
    #                             for p in y_true_g])

    fig, ax = plt.subplots(figsize=(7, 7))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    ax = sns.heatmap(cm, square=True)
    ax.set_title("all result, tau = {:.4f}, acc = {:.4f}".
                 format(tau, cm.diagonal().mean()))
    plt.savefig(result_path + "confusion_matrix_true.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    cm = confusion_matrix(y_true, y_pred, normalize='pred')
    ax = sns.heatmap(cm, square=True)
    ax.set_title("all result, tau = {:.4f}, acc = {:.4f}"
                 .format(tau, cm.diagonal().mean()))
    plt.savefig(result_path + "confusion_matrix_pred.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    cm = confusion_matrix(y_true, y_pred, normalize='all')
    ax = sns.heatmap(cm, square=True)
    ax.set_title("all result, tau = {:.4f}, acc = {:.4f}"
                 .format(tau, cm.diagonal().sum()))
    plt.savefig(result_path + "confusion_matrix_all.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    cm = confusion_matrix(y_true_g, y_pred_g, normalize='all')
    ax = sns.heatmap(cm, square=True)
    ax.set_title("all result, tau = {:.4f}, acc = {:.4f}"
                 .format(tau_g, cm.diagonal().sum()))
    plt.savefig(result_path + "general_confusion_matrix_pred.png")


if __name__ == '__main__':
    model_name = sys.argv[1]
    plot_PCA(model_name, tsne=True)
    plot_CM(model_name)
