import numpy as np
import os, time
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import logging

from model import *

import pickle

class DKVMNAnalyzer():
    def __init__(self, args, sess, dkvmn):
        
        self.args = args
        self.sess = sess
        self.dkvmn = dkvmn
        self.dkvmn.init_step()
        self.num_actions = self.args.n_questions
        self.log_file_dir = 'DKVMN_analysis'
        if not os.path.exists(self.log_file_dir):
            os.mkdir(self.log_file_dir)

        self.logger = self.set_logger()
        self.logger.setLevel(eval('logging.{}'.format(self.args.logging_level)))

        self.logger.debug('Initializing DKVMN Analyzer')

        #self.value_matrix = self.get_init_value_matrix()

    def set_logger(self):
        logger = logging.getLogger('aDKVMN')

        streamFormatter = logging.Formatter('[%(levelname)s] %(message)s')
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(streamFormatter)
        streamHandler.setLevel(logging.DEBUG)

        fileFormatter = logging.Formatter('%(message)s')
        fileHandler = logging.FileHandler('./analysis_dkvmn.log')
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(logging.INFO)

        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)

        return logger

    def get_init_value_matrix(self):
        return self.sess.run(self.dkvmn.init_memory_value)

    def get_prediction_probability(self, value_matrix):
        return np.squeeze(self.sess.run(self.dkvmn.total_pred_probs, feed_dict={self.dkvmn.total_value_matrix: value_matrix}))
    
    def update_value_matrix(self, value_matrix, action, answer):
       ops = [self.dkvmn.stepped_value_matrix]
       value_matrix = self.sess.run(ops, feed_dict={self.dkvmn.q: action, self.dkvmn.a: answer, self.dkvmn.value_matrix: value_matrix})
       return np.squeeze(value_matrix)

    def expand_dims(self, val):
        return np.expand_dims(np.expand_dims(val, axis=0), axis=0)


    def test1(self):
        '''
        Positive sensitivity of overall questions
        '''
        self.dkvmn.load()
        init_value_matrix = self.get_init_value_matrix()
        init_probs = self.get_prediction_probability(init_value_matrix)
        
        log_file_name = os.path.join(self.log_file_dir, 'test1.csv')
        log_file = open(log_file_name, 'a') 
        log_file.write(self.dkvmn.model_dir)
        header = 'Action, prob, diff, overall_diff'
        log_file.write(header+'\n')
        
        answer = self.expand_dims(1)
        neg_counter = 0
        for action_idx in range(self.num_actions):
            action = self.expand_dims(action_idx+1)

            #ops = [self.dkvmn.stepped_value_matrix]
            #value_matrix = self.sess.run(ops, feed_dict={self.dkvmn.q: action, self.dkvmn.a: answer, self.dkvmn.value_matrix:init_value_matrix})
            value_matrix = self.update_value_matrix(init_value_matrix, action, answer)
            probs = self.get_prediction_probability(value_matrix)
            probs_diff = probs - init_probs
            probs_diff_action = probs_diff[action_idx]
            probs_diff_avg = np.average(probs_diff)

            if probs_diff_avg < 0:
                neg_counter += 1
            log = '{:>3}, {: .4f}, {: .4f}, {: .4f}'.format(action_idx, init_probs[action_idx], probs_diff_action, probs_diff_avg)
            log_file.write(log+'\n')
            log_file.flush()
            
            #self.logger.debug('Action: {:>3}, prob: {: .4f}, diff: {: .4f}, overall_diff: {: .4f}'.format(action_idx, init_probs[action_idx], probs_diff_action, probs_diff_avg))
        self.logger.debug('Number of negative update skills: {}'.format(neg_counter))

