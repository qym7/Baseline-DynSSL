import sys

import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = (20.0, 4.0) 
import seaborn as sns; sns.set_theme()
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def plot_PCA(folder_name, TSNE=False):
    result_path = '/data/biocomp/qin/results/supervision_dataset05/'+folder_name+'/evaluation/'
    data = np.load(result_path + 'calculation.npz')
    frames = data.hists
    y_true = data.y_true

    if TSNE:
        tsne_2 = TSNE(n_components=2,random_state=33).fit_transform(frames)
        tsne_3 = TSNE(n_components=3,random_state=33).fit_transform(frames)

        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_2[:, 0], tsne_2[:, 1], c=y_true,label="t-SNE", cmap='viridis')
        plt.colorbar()
        plt.legend()
        plt.savefig(result_path+folder_name+'tsne_outputs_2.png', dpi=120)

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
    elt.colorbar()
    plt.xlabel(pca_2.explained_variance_ratio_[0])
    plt.ylabel(pca_2.explained_variance_ratio_[1])
    plt.legend()
    plt.savefig(result_path+folder_name+'pca_outputs_2.png', dpi=120)

    ax = plt.axes(projection ="3d")
    ax.scatter(pca_3_res[:, 0], pca_3_res[:, 1], pca_3_res[:, 2], c=y_true,label="PCA", cmap='viridis')
    ax.set_xlabel(pca_3.explained_variance_ratio_[0])
    ax.set_ylabel(pca_3.explained_variance_ratio_[1])
    ax.set_zlabel(pca_3.explained_variance_ratio_[2])
    plt.colorbar()
    plt.legend()
    plt.savefig(result_path+folder_name + 'pca_outputs_3.png', dpi=120)


def plot_CM(folder_name):
    # CONFUSION MATRIX
    result_path = '/data/biocomp/qin/results/supervision_dataset05/'+folder_name+'/evaluation/'
    data = np.load(result_path + 'calculation.npz')
    y_pred = data.y_pred
    y_true = data.y_true
    y_true_g = y_true_g
    y_pred_g = y_pred_g
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


if __name__ == '__main__':
    model_name = sys.argv[1]
    plot_PCA(model_name, TSNE=True)
    plot_CM(model_name)
