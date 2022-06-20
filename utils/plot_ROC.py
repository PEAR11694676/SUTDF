from itertools import cycle

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
label = np.load( '../data/test_data/Mix_target_0.5m_label.npy')
pred_05 = np.load('../data/ROC_data/pred_0.5m.npy')
pred_15 = np.load('../data/ROC_data/pred_1.5m.npy')
pred_20 = np.load('../data/ROC_data/pred_2m.npy')
pred_25 = np.load('../data/ROC_data/pred_2.5m.npy')
pred_30 = np.load('../data/ROC_data/pred_3m.npy')
pred_35 = np.load('../data/ROC_data/pred_3.5m.npy')
pred_40 = np.load('../data/ROC_data/pred_4m.npy')

import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np


from sklearn.preprocessing import label_binarize

def plot_roc(label, pred, nb_classes=5):
    Y_valid = label
    Y_pred = pred
    Y_valid = label_binarize(Y_valid, classes=[i for i in range(nb_classes)])
    Y_pred = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:,i], Y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(nb_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #     # Finally average it and compute AUC
    #     mean_tpr /= nb_classes
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # # Plot all ROC curves
    lw = 2

    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    # plt.plot(fpr["macro"], tpr["macro"],
    #      label='macro-average ROC curve (area = {0:0.2f})'
    #            ''.format(roc_auc["macro"]),
    #      color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of  multi-class 4.0m')
        plt.legend(loc="lower right")
        plt.savefig("../result/roc_4.0m.png")
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))

plot_roc(label, pred_40)







