import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def calculate_metric(target, pred):

    right_index = (target != -1.)

    right_target = target[right_index]
    right_pred = pred[right_index]

    auc = roc_auc_score(right_target, right_pred)

    right_pred[right_pred > 0.5] = 1.0
    right_pred[right_pred <= 0.5] = 0.0
    acc = accuracy_score(right_target, right_pred)

    return auc, acc 
