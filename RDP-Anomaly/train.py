"""
Author: Bill Wang
"""

import datetime
import platform
from rdp_tree import RDPTree
import shutil
import os
import sys
from util import dataLoading, random_list, tic_time
import time

data_path = "data/apascal.csv"

save_path = "save_model/"
log_path = "logs/log.log"
logfile = open(log_path, 'w')
node_batch = 30
node_epoch = 200  # epoch for a node training
eval_interval = 24
batch_size = 192
out_c = 50
USE_GPU = True
LR = 1e-1
tree_depth = 8
forest_Tnum = 30
filter_ratio = 0.05  # filter those with high anomaly scores
dropout_r = 0.1
random_size = 10000  # randomly choose 1024 size of data for training

# Set mode
dev_flag = True
if dev_flag:
    print("Running in DEV_MODE!")
else:
    # running on servers
    print("Running in SERVER_MODE!")
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logfile = None


def main():
    global random_size

    shutil.rmtree(save_path)
    os.mkdir(save_path)

    svm_flag = False
    if 'svm' in data_path:
        svm_flag = True
        from util import get_data_from_svmlight_file
        x_ori, labels_ori = get_data_from_svmlight_file(data_path)
        random_size = 1024
    else:
        x_ori, labels_ori = dataLoading(data_path, logfile)
    data_size = labels_ori.size

    # build forest
    forest = []
    for i in range(forest_Tnum):
        forest.append(RDPTree(t_id=i+1,
                              tree_depth=tree_depth,
                              filter_ratio=filter_ratio,
                              ))

    print("Init tic time.")
    tic_time()

    # training process
    for i in range(forest_Tnum):

        # random sampling with replacement
        random_pos = random_list(0, data_size-1, random_size)
        # random sampling without replacement
        # random_pos = random.sample(range(0, data_size), random_size)

        # to form x and labels
        x = x_ori[random_pos]
        if svm_flag:
            labels = labels_ori[random_pos]
        else:
            labels = labels_ori[random_pos].values

        print("tree id:", i, "tic time.")
        tic_time()

        forest[i].training_process(
            x=x,
            labels=labels,
            batch_size=batch_size,
            node_batch=node_batch,
            node_epoch=node_epoch,
            eval_interval=eval_interval,
            out_c=out_c,
            USE_GPU=USE_GPU,
            LR=LR,
            save_path=save_path,
            logfile=logfile,
            dropout_r=dropout_r,
            svm_flag=svm_flag,
        )

        print("tree id:", i, "tic time end.")
        tic_time()


if __name__ == "__main__":
    main()
