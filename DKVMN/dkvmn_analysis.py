import numpy as np
import os, time
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import dkvmn_utils

from model import *

import pickle

class DKVMNAnalyzer():
    def __init__(self, args, sess, dkvmn):
        
        self.args = args
        self.sess = sess
        self.dkvmn = dkvmn
        self.dkvmn.build_step_graph()
        self.num_actions = self.args.n_questions

        self.logger = dkvmn_utils.set_logger('aDKVMN', 'dkvmn_analysis.log', self.args.logging_level)
        self.logger.debug('Initializing DKVMN Analyzer')

    def test1(self):
        '''
        Positive sensitivity of overall questions
        '''
        self.dkvmn.load()

        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix)
        
        answer = self.dkvmn.expand_dims(1)
        neg_counter = 0

        for action_idx in range(self.num_actions):

            action = self.dkvmn.expand_dims(action_idx+1)
            value_matrix = self.dkvmn.update_value_matrix(init_value_matrix, action, answer)
            probs = self.dkvmn.get_prediction_probability(value_matrix)

            probs_diff = probs - init_probs
            probs_diff_action = probs_diff[action_idx]
            probs_diff_avg = np.average(probs_diff)

            if probs_diff_avg < 0:
                neg_counter += 1
            
            self.logger.info('Action: {:>3}, prob: {: .4f}, diff: {: .4f}, overall_diff: {: .4f}'.format(action_idx, init_probs[action_idx], probs_diff_action, probs_diff_avg))

        self.logger.info('Number of negative update skills: {}'.format(neg_counter))

