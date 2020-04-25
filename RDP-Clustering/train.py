"""
Author: Bill Wang
"""

import platform
import sys

import pandas as pd
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_olivetti_faces, fetch_rcv1

import shutil
import os
from util import dataLoading, get_data_from_svmlight_file, random_list, tic_time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats import mode
import torch
from model import RDP_Model

data_path = "r8"

save_path = "save_model/"
log_path = "logs/log.log"
total_epoch = 1 + 1000  # epoch for a node training
epoch_batch = 30
eval_interval = 25
batch_size = 192
out_c = 512
USE_GPU = True
LR = 1e-1
dropout_r = 0.1
is_eval = False

if not torch.cuda.is_available():
    USE_GPU = False

# Set mode
dev_flag = True
if dev_flag:
    print("Running in DEV_MODE!")
else:
    # running on servers
    print("Running in SERVER_MODE!")
    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    out_c = int(sys.argv[3])
    save_path = "save_model/" + data_path + "_b" + str(batch_size) + "_[" + str(out_c) + "]_" + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = "logs/train_" + data_path + "_b" + str(batch_size) + "_[" + str(out_c) + "]_" + ".log"

logfile = open(log_path, 'w')


def main():

    shutil.rmtree(save_path)
    os.mkdir(save_path)

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
    data_size = labels.size

    # build model
    model = RDP_Model(in_c=x.shape[1], out_c=out_c, USE_GPU=USE_GPU,
                    LR=LR, logfile=logfile, dropout_r=dropout_r)

    best_nmi = best_epoch = 0
    loss = 0

    for epoch in range(0, total_epoch):

        # random sampling with replacement
        for batch_i in range(epoch_batch):
            random_pos = random_list(0, data_size-1, batch_size)
            batch_data = x[random_pos]
            loss = model.train_model(batch_data, epoch)

        if epoch % eval_interval == 0:
            print("epoch ", epoch, "loss:", loss)
            if logfile:
                logfile.write("epoch " + str(epoch) + " loss: " + str(loss) + '\n')

            model.save_model(save_path + 'model_latest.h5')

            # eval
            if is_eval:
                gap_dims = model.eval_model(x)

                kmeans_results = KMeans(n_clusters=n_clusters, random_state=0).fit(gap_dims)
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
                if logfile:
                    logfile.write("RI_scores: %.4f\n" % RI_scores)

                if best_nmi < nmi_scores:
                    best_nmi = nmi_scores
                    best_epoch = epoch

                print("Best NMI: %.4f" % best_nmi)
                print("Best Epoch %d\n" % best_epoch)
                if logfile:
                    logfile.write("Best NMI: %.4f\n" % best_nmi)
                    logfile.write("Best Epoch %d\n\n" % best_epoch)
                    logfile.flush()


if __name__ == "__main__":
    main()
