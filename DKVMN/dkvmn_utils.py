import numpy as np
import os
import logging

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def calculate_auc(target, pred):
    right_index = (target != -1.)

    right_target = target[right_index]
    right_pred = pred[right_index]

    try:
        return roc_auc_score(right_target, right_pred)
    except:
        return -1

def calculate_acc(target, pred):
    right_index = (target != -1.)

    right_target = target[right_index]
    right_pred = pred[right_index]

    right_pred[right_pred > 0.5] = 1.0
    right_pred[right_pred <= 0.5] = 0.0

    return accuracy_score(right_target, right_pred)


def calculate_metric(target, pred):
    auc = calculate_auc(target, pred) 
    acc = calculate_acc(target, pred)

    return auc, acc 


def calculate_metric_for_each_q(target, pred, q, num_q):
    count = list()
    result = list()

    for q_idx in range(num_q):

        filtered_idx = (q == q_idx+1)
        count.append(np.sum(filtered_idx))
        filtered_target = target[filtered_idx]
        filtered_pred = pred[filtered_idx]
        sub_result = calculate_metric(filtered_target, filtered_pred)
         
        result.append(sub_result)

    return count, result


def set_logger(name, path, logging_level):
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    streamFormatter = logging.Formatter('[%(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(streamFormatter)
    streamHandler.setLevel(logging.DEBUG)

    if not os.path.exists('log'):
        os.mkdir('log')

    fileFormatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(os.path.join('log',path))
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.INFO)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(eval('logging.{}'.format(logging_level)))

    return logger
