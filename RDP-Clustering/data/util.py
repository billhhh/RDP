import pandas as pd
import random
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score, roc_auc_score

mem = Memory("./dataset/svm_data")

@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def dataLoading(path, logfile=None):

    # loading data
    df = pd.read_csv(path)
    labels = df['class']
    x_df = df.drop(['class'], axis=1)
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    if logfile:
        logfile.write("Data shape: (%d, %d)\n" % x.shape)

    return x, labels


# random sampling with replacement
def random_list(start, stop, length):
    if length >= 0:
        length = int(length)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))  # including start and stop
    return random_list


def aucPerformance(scores, labels, logfile=None):
    roc_auc = roc_auc_score(labels, scores)
#    print(roc_auc)
    ap = average_precision_score(labels, scores)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    if logfile:
        logfile.write("AUC-ROC: %.4f, AUC-PR: %.4f\n" % (roc_auc, ap))

#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1],[0,1],'r--')
#    plt.xlim([-0.001, 1])
#    plt.ylim([0, 1.001])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show();

    return roc_auc, ap
