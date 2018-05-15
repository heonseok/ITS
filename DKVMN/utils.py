import numpy as np
import os
import logging

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

########## METRIC ##########
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


def calculate_auc_acc(target, pred):
    auc = calculate_auc(target, pred) 
    acc = calculate_acc(target, pred)

    return auc, acc 

########## LOGGER ##########
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


######### utility ############
def float2str_list(input_list):
    return ','.join('{:.4f}'.format(item) for item in input_list)

def int2str_list(input_list):
    return ','.join('{}'.format(item) for item in input_list)
 
def calc_seq_trend(seq):
    zeros = np.zeros(shape=(seq.shape[0], 1))
    seq_right = np.hstack((seq, zeros))
    seq_left = np.hstack((zeros, seq))

    seq_diff = (seq_right-seq_left)[:,1:-1]
    increase_count = len(np.where(seq_diff>0)[0])
    decrease_count = len(np.where(seq_diff<0)[0])

    return increase_count, decrease_count

def calc_rmse_with_ones(seq):
    ones = np.ones_like(seq)
    return np.sqrt(np.average(np.square(seq-ones), axis=0))

def calc_diff_with_ones(seq):
    ones = np.ones_like(seq)
    return np.average((ones-seq), axis=0)
 
    
