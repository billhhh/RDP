"""
Author: Bill Wang
"""

import datetime
import platform
import time

from rdp_tree import RDPTree
import numpy as np
import sys
from util import dataLoading, aucPerformance, tic_time

data_path = "data/apascal.csv"

load_path = "save_model/"
out_c = 50
USE_GPU = True
tree_depth = 8
forest_Tnum = 30
dropout_r = 0.1

# count from 1
testing_methods_set = ['last_layer', 'first_layer', 'level']
testing_method = 1

# Set mode
dev_flag = True
if dev_flag:
    print("Running in DEV_MODE!")
else:
    # running on servers
    print("Running in SERVER_MODE!")
    data_path = sys.argv[1]
    load_path = sys.argv[2]
    tree_depth = int(sys.argv[3])
    testing_method = int(sys.argv[4])


def main():
    svm_flag = False
    if 'svm' in data_path:
        svm_flag = True
        from util import get_data_from_svmlight_file
        x, labels = get_data_from_svmlight_file(data_path)
    else:
        x, labels = dataLoading(data_path)
    data_size = labels.size

    # build forest
    forest = []
    for i in range(forest_Tnum):
        forest.append(RDPTree(t_id=i+1,
                              tree_depth=tree_depth,
                              ))

    sum_result = np.zeros(data_size, dtype=np.float64)

    print("Init tic time.")
    tic_time()

    # testing process
    for i in range(forest_Tnum):

        print("tree id:", i, "tic time.")
        tic_time()

        x_level, first_level_scores = forest[i].testing_process(
            x=x,
            out_c=out_c,
            USE_GPU=USE_GPU,
            load_path=load_path,
            dropout_r=dropout_r,
            testing_method=testing_methods_set[testing_method - 1],
            svm_flag=svm_flag,
        )

        if testing_methods_set[testing_method - 1] == 'level':
            sum_result += x_level
        else:
            sum_result += first_level_scores

        print("tree id:", i, "tic time.")
        tic_time()

    scores = sum_result / forest_Tnum
    aucPerformance(scores, labels)


if __name__ == "__main__":
    main()
