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

    
    def test1_1(self):
        self.logger.info('#'*120)
        self.logger.info(self.dkvmn.model_dir)
        self.logger.info('Test 1-1')

        self.test1_base(1)
    
    def test1_2(self):
        self.logger.info('#'*120)
        self.logger.info(self.dkvmn.model_dir)
        self.logger.info('Test 1-2')

        self.test1_base(0)
        
    def test1_base(self, answer_type):
        '''
        Positive sensitivity of overall questions
        '''
        self.dkvmn.load()

        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix)
        
        answer = self.dkvmn.expand_dims(answer_type)
        skill_counter = 0
        right_update_skill_counter = 0

        self.logger.info('Action, prob, diff, diff_avg, wrong_response')
        for action_idx in range(self.num_actions):

            action = self.dkvmn.expand_dims(action_idx+1)
            value_matrix = self.dkvmn.update_value_matrix(init_value_matrix, action, answer)
            probs = self.dkvmn.get_prediction_probability(value_matrix)

            probs_diff = probs - init_probs
            probs_diff_action = probs_diff[action_idx]
            probs_diff_avg = np.average(probs_diff)

            if answer_type == 1:
                wrong_response = np.sum(probs_diff < 0)
                '''
                if probs_diff_avg < 0:
                    skill_counter += 1
                '''

            elif answer_type == 0:
                wrong_response = np.sum(probs_diff > 0)
                '''
                if probs_diff_avg > 0:
                    skill_counter += 1
                '''

            if wrong_response == 0:
                right_update_skill_counter += 1


            self.logger.info('{:>3}, {: .4f}, {: .4f}, {: .4f}, {:>3}'.format(action_idx+1, init_probs[action_idx], probs_diff_action, probs_diff_avg, wrong_response))

        #self.logger.info('Total number of wrong update skills: {}'.format(skill_counter))
        self.logger.info('Number of right update skills : {}'.format(right_update_skill_counter))


    def test2(self):
        pass
