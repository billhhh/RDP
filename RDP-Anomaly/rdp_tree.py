"""
Author: Bill Wang

RDP Tree class for filter normal samples
"""


from model import RDP_Model
import numpy as np
import random
import os
from util import random_list, aucPerformance

is_batch_replace = True
is_eval = False
test_1l_only = True


class RDPTree():
    def __init__(self,
                 t_id,
                 tree_depth,
                 filter_ratio=0.1):

        self.t_id = t_id
        self.tree_depth = tree_depth
        self.filter_ratio = filter_ratio
        self.thresh = []

    # include train and eval
    def training_process(self,
                         x,
                         labels,
                         batch_size,
                         node_batch,
                         node_epoch,
                         eval_interval,
                         out_c,
                         USE_GPU,
                         LR,
                         save_path,
                         logfile=None,
                         dropout_r=0.1,
                         svm_flag=False,
                         ):
        if svm_flag:
            x_ori = x.toarray()
        else:
            x_ori = x
        labels_ori = labels
        x_level = np.zeros(x_ori.shape[0])
        for level in range(1, self.tree_depth+1):

            # form x and labels
            keep_pos = np.where(x_level == 0)
            x = x_ori[keep_pos]
            labels = labels_ori[keep_pos]
            group_num = int(x.shape[0] / batch_size) + 1
            batch_x = np.array_split(x, group_num)
            model = RDP_Model(in_c=x.shape[1], out_c=out_c, USE_GPU=USE_GPU,
                            LR=LR, logfile=logfile, dropout_r=dropout_r)
            best_auc = best_epoch = 0

            for epoch in range(0, node_epoch):
                if not is_batch_replace:
                    random.shuffle(batch_x)
                    batch_cnt = 0
                    for batch_i in batch_x:
                        gap_loss = model.train_model(batch_i, epoch)
                        # print("epoch ", epoch, "loss: ", loss)
                        batch_cnt += 1
                        if batch_cnt >= node_batch:
                            break

                else:
                    # random sampling with replacement
                    for batch_i in range(node_batch):
                        random_pos = random_list(0, x.shape[0] - 1, batch_size)
                        batch_data = x[random_pos]
                        gap_loss = model.train_model(batch_data, epoch)

                if epoch % eval_interval == 0:
                    # print("epoch ", epoch, "gap_loss:", gap_loss, " recon_loss:", recon_loss)
                    # if logfile:
                    #     logfile.write("epoch " + str(epoch) + " gap_loss: " + str(gap_loss) +
                    #                   " recon_loss: " + str(recon_loss) + '\n')

                    print("tree_id:", self.t_id, "level:", level)
                    print("keep_pos.size ==", keep_pos[0].size)
                    if logfile:
                        logfile.write("tree_id: " + str(self.t_id) + " level: " + str(level)
                                      + "keep_pos.size == " + str(keep_pos[0].size) + '\n')
                    print("epoch ", epoch, "gap_loss:", gap_loss)
                    if logfile:
                        logfile.write("epoch " + str(epoch) + " gap_loss: " + str(gap_loss) + '\n')
                    model.save_model(save_path + 't' + str(self.t_id) + '_l' + str(level) + '_latest.h5')

                    scores = model.eval_model(x)

                    # eval
                    if is_eval:
                        try:
                            roc_auc, ap = aucPerformance(scores, labels, logfile)
                            if roc_auc > best_auc:
                                best_auc = roc_auc
                                best_epoch = epoch

                            print("Best AUC-ROC: %.4f" % best_auc)
                            if logfile:
                                logfile.write("Best AUC-ROC: %.4f\n" % best_auc)
                            print("Best Epoch %d\n" % best_epoch)
                            if logfile:
                                logfile.write("Best Epoch %d\n\n" % best_epoch)
                        except ValueError:
                            print("Only one class present in y_true. ROC AUC score is not defined in that case.")

                    if logfile:
                        logfile.flush()

            # filter anomaly elements. the higher the scores are, the more abnormal
            ranking_scores = scores
            score_ranking_idx = np.argsort(ranking_scores)
            filter_num = int(self.filter_ratio * score_ranking_idx.size)
            filter_idx = score_ranking_idx[score_ranking_idx.size-filter_num:]
            x_level[keep_pos[0][filter_idx]] = self.tree_depth+1 - level
            self.thresh.append(ranking_scores[score_ranking_idx[score_ranking_idx.size-filter_num]])

            # epoch for
        # level for

        # save self.thresh
        filename = save_path + 'threshList_t' + str(self.t_id) + '.txt'
        list_save(self.thresh, filename, 'w')

    def testing_process(self,
                        x,
                        out_c,
                        USE_GPU,
                        load_path,
                        dropout_r,
                        testing_method='last_layer',
                        svm_flag=False,
                        ):

        if svm_flag:
            x_ori = x.toarray()
        else:
            x_ori = x
        x_level = np.zeros(x_ori.shape[0])
        self.thresh = list_read(load_path + 'threshList_t' + str(self.t_id) + '.txt')
        for level in range(1, self.tree_depth + 1):
            # form x
            keep_pos = np.where(x_level == 0)
            x = x_ori[keep_pos]
            model = RDP_Model(in_c=x.shape[1], out_c=out_c, USE_GPU=USE_GPU,
                            dropout_r=dropout_r)
            
            if testing_method == 'last_layer':
                # high --> low load
                model.load_model(
                    load_path + 't' + str(self.t_id) + '_l' + str(self.tree_depth + 1 - level) + '_latest.h5')
            else:
                # low --> high load
                model.load_model(load_path + 't' + str(self.t_id) + '_l' + str(level) + '_latest.h5')

            # eval
            scores = model.eval_model(x)

            if level == 1:
                first_level_scores = scores
                if test_1l_only and testing_method != 'level':
                    return x_level, first_level_scores
            # filter elements
            if testing_method == 'last_layer':
                filter_idx = np.where(scores >= float(self.thresh[self.tree_depth + 1 - level - 1]))
            else:
                filter_idx = np.where(scores >= float(self.thresh[level-1]))
            x_level[keep_pos[0][filter_idx]] = self.tree_depth+1 - level

        return x_level, first_level_scores


def list_save(content, filename, mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()


def list_read(filename):
    # Try to read a txt file and return a list. Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]

    file.close()
    return content
