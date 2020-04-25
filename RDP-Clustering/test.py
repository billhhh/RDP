"""
Author: Bill Wang
"""

import torch
import pandas as pd
import numpy as np
import platform
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups_vectorized, load_digits, fetch_olivetti_faces, fetch_rcv1
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats import mode
from util import dataLoading, get_data_from_svmlight_file, tic_time
import sys
from model import RDP_Model

data_path = "r8"

save_path = "save_model/model_latest.h5"

log_path = "logs/log.log"
out_c = 512
USE_GPU = True
LR = 1e-1
dropout_r = 0.1

# Set mode
dev_flag = platform.node()
if dev_flag == "bill-XPS-8930":
    print("Running in DEV_MODE!")
else:
    # running on servers
    print("Running in SERVER_MODE!")
    if not torch.cuda.is_available():
        USE_GPU = False
    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    out_c = int(sys.argv[3])
    save_path = "save_model/" + data_path + "_b" + str(batch_size) + "_[" + str(out_c) + "]_" + "/" + \
                "model_latest.h5"
    log_path = "logs/test_" + data_path + "_b" + str(batch_size) + "_[" + str(out_c) + "]_" + ".log"

logfile = open(log_path, 'w')


def main():

    is_tsne = False

    if data_path == '20newsgroups':
        newsgroups_data = fetch_20newsgroups_vectorized(subset='all')
        x = newsgroups_data.data.toarray()
        labels = newsgroups_data.target
        n_clusters = 20
    elif data_path == 'r8':
        df = pd.read_csv('data/r8-all-stemmed.txt')
        labels_idx = ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']
        labels = df['class'].values
        labels = [labels_idx.index(ele) for ele in labels]
        labels = np.asarray(labels, dtype=np.int64)
        x_df = df.drop(['class'], axis=1)
        corpus = np.squeeze(x_df.values)

        is_TfidfVectorizer = True
        if is_TfidfVectorizer:
            vectorizer = TfidfVectorizer()
            x = vectorizer.fit_transform(corpus).toarray()
        else:
            vectorizer = CountVectorizer()
            x = vectorizer.fit_transform(corpus).toarray()
        n_clusters = 8
    elif data_path == 'olivetti_faces':
        data = fetch_olivetti_faces()
        x = data.data
        labels = data.target
        n_clusters = 40
    elif data_path == 'rcv1':
        # data = fetch_rcv1()
        # x = data.data.toarray()
        # labels = data.target.toarray()
        # n_clusters = 103

        x, labels = get_data_from_svmlight_file('data/rcv1_train.binary')
        x = x.toarray()
        n_clusters = 2
    elif data_path == 'sector':
        x, labels = get_data_from_svmlight_file('data/sector.scale.all')
        x = x.toarray()
        n_clusters = 105
    else:
        raise Exception("Invalid data path!")
    print("Data shape: (%d, %d)" % x.shape)

    model = RDP_Model(in_c=x.shape[1], out_c=out_c, USE_GPU=USE_GPU,
                    LR=LR, logfile=logfile, dropout_r=dropout_r)
    model.load_model(save_path)

    print("Projection tic time.")
    tic_time()
    gap_dims = model.eval_model(x)
    tic_time()
    print("Projection tic time end.")

    # evaluations
    kmeans_rounds = 30
    for round in range(kmeans_rounds):
        print("round:", round)
        if logfile:
            logfile.write("round: %d\n" % round)

        print("KMeans tic time.")
        tic_time()
        kmeans_results = KMeans(n_clusters=n_clusters, random_state=round).fit(gap_dims)
        print("KMeans tic time end.")
        tic_time()

        # Match each learned cluster label with the true labels found in them
        y_pred = kmeans_results.labels_
        labels_pred = np.zeros_like(y_pred)
        for i in range(n_clusters):
            mask = (y_pred == i)
            labels_pred[mask] = mode(labels[mask])[0]

        # evaluations
        nmi_scores = normalized_mutual_info_score(labels, labels_pred)
        print("nmi_scores:", nmi_scores)
        if logfile:
            logfile.write("nmi_scores: %.4f\n" % nmi_scores)

        fscores = f1_score(labels, labels_pred, average='macro')
        print("fscores_macro:", fscores)
        if logfile:
            logfile.write("fscores_macro: %.4f\n" % fscores)

        fscores = f1_score(labels, labels_pred, average='micro')
        print("fscores_micro:", fscores)
        if logfile:
            logfile.write("fscores_micro: %.4f\n" % fscores)

        fscores = f1_score(labels, labels_pred, average='weighted')
        print("fscores_weighted:", fscores)
        if logfile:
            logfile.write("fscores_weighted: %.4f\n" % fscores)

        RI_scores = adjusted_rand_score(labels, labels_pred)
        print("RI_scores:", RI_scores)
        print()
        if logfile:
            logfile.write("RI_scores: %.4f\n\n" % RI_scores)
            logfile.flush()

    if is_tsne:
        tsne_results = TSNE(n_components=2).fit_transform(gap_dims)
        vis_x = tsne_results[:, 0]
        vis_y = tsne_results[:, 1]
        plt.scatter(vis_x, vis_y, c=labels_pred, cmap=plt.cm.get_cmap("jet", 10), marker='.')
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.show()

        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", 10), marker='.')
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.show()

    return


if __name__ == "__main__":
    main()
