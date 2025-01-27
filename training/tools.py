import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Confusion matrix
def knockout_confusion_matrix(data, thresh=0.5, score='score', column='knockout_name'):
    #Split by knockout and do majority voting
    tp, fp, tn, fn = [], [], [], []
    for k in data[column].unique():
        k_data = data[data[column] == k]
        mean_pred_score = k_data[score].mean()

        if score == 'pred_proba_0':
            cond = mean_pred_score < thresh
        else:
            cond = mean_pred_score > thresh

        if cond:
            if k_data.true_label.mean() == 1:
                tp.append(k)
            else:
                fp.append(k)
        else:
            if k_data.true_label.mean() == 0:
                tn.append(k)
            else:
                fn.append(k)

    # Confusion matrix
    TP = len(tp)
    FP = len(fp)
    FN = len(fn)
    TN = len(tn)

    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    return confusion_matrix
