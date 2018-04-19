import numpy as np
import os, time
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from utils import *

from model import *

import pickle

class DKVMNAnalyzer():
    def __init__(self, args, sess, dkvmn):
        
        self.args = args
        #self.args.batch_size = 1
        #self.args.seq_len = 1
        self.sess = sess
        self.dkvmn = dkvmn
        self.dkvmn.build_step_graph()
        self.num_actions = self.args.n_questions

        self.logger = set_logger('aDKVMN', self.args.prefix + 'dkvmn_analysis.log', self.args.logging_level)
        self.logger.debug('Initializing DKVMN Analyzer')


    def calc_influence(self, action_idx, answer_type, _value_matrix, _counter, update_value_matrix_flag):

        value_matrix = np.copy(_value_matrix)
        counter = np.copy(_counter)
 
        answer = self.dkvmn.expand_dims(answer_type)

        prev_probs = self.dkvmn.get_prediction_probability(value_matrix, counter)
        prev_mastery_level = self.dkvmn.get_mastery_level(value_matrix, counter)
        action = self.dkvmn.expand_dims(action_idx+1)

        if update_value_matrix_flag == True:
            value_matrix = self.dkvmn.update_value_matrix(value_matrix, action, answer, counter)
        counter[0,action_idx + 1] += 1

        probs = self.dkvmn.get_prediction_probability(value_matrix, counter)
        mastery_level = self.dkvmn.get_mastery_level(value_matrix, counter)

        probs_diff = probs - prev_probs
        mastery_diff = mastery_level - prev_mastery_level
    
        if answer_type == 1:
            wrong_response_count_prob = np.sum(probs_diff < 0)
            wrong_response_count_mastery = np.sum(mastery_diff < 0)
        elif answer_type == 0:
            wrong_response_count_prob = np.sum(probs_diff > 0)
            wrong_response_count_mastery = np.sum(mastery_diff > 0)

        return value_matrix, counter, probs, mastery_level, wrong_response_count_prob, wrong_response_count_mastery


    def test(self):
        self.logger.info('#'*120)
        self.logger.info(self.dkvmn.model_dir)

        #self.dkvmn.load()
        
        self.logger.info('value matrix update enable') 
        self.test1(True)
        self.test2(True)
    
        '''
        self.logger.info('value matrix update disable') 
        self.test1(False)
        self.test2(False)
        '''
        
    def test1(self, update_value_matrix_flag):
        self.logger.info('Test 1')
        #self.logger.info('Answer_type, 0, 0.001, 0.01, 0.05')
        self.test1_base(1, update_value_matrix_flag)
        self.test1_base(0, update_value_matrix_flag)

    def test2(self, update_value_matrix_flag):
        self.logger.info('Test 2')
        #self.logger.info('Answer_type, increase_count, decrease_count, diff, rmse')
        self.test2_base(1, update_value_matrix_flag)
        self.test2_base(0, update_value_matrix_flag)
        

    def test1_base(self, answer_type, update_value_matrix_flag):
        '''
        Reponse of one action 
        '''

        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_counter = self.dkvmn.get_init_counter()
        
        right_updated_skill_counter = 0
        wrong_updated_skill_list = []

        right_updated_mastery_counter = 0
        wrong_updated_mastery_list = []

        for action_idx in range(self.num_actions):
            _, _, _, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, init_value_matrix, init_counter, update_value_matrix_flag)

            if wrong_response_count_prob == 0:
                right_updated_skill_counter += 1
            elif wrong_response_count_prob > 0:
                wrong_updated_skill_list.append(action_idx+1)

            if wrong_response_count_mastery == 0:
                right_updated_mastery_counter += 1
            elif wrong_response_count_mastery > 0:
                wrong_updated_mastery_list.append(action_idx+1)

        self.logger.info('Probs   {}, {}'.format(answer_type, right_updated_skill_counter))
        self.logger.info('{}'.format(int2str_list(wrong_updated_skill_list)))

        self.logger.info('Mastery {}, {}'.format(answer_type, right_updated_mastery_counter))
        self.logger.info('{}'.format(int2str_list(wrong_updated_mastery_list)))


    # TODO : rename it

    def test2_base(self, answer_type, update_value_matrix_flag):
        ''' 
        Response of repeated action 
        '''
        repeat_num = 10

        init_counter = self.dkvmn.get_init_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()

        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter)
   
        prob_mat = np.zeros((self.num_actions, repeat_num+1))

        for action_idx in range(self.num_actions):
            action = self.dkvmn.expand_dims(action_idx+1)
            value_matrix = np.copy(init_value_matrix)
            counter = np.copy(init_counter)

            prob_mat[action_idx][0] = init_probs[action_idx]
            for repeat_idx in range(repeat_num):

                value_matrix, counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, update_value_matrix_flag)
                prob_mat[action_idx][repeat_idx+1] = probs[action_idx]

        increase_count, decrease_count = calc_seq_trend(prob_mat)
        rmse = calc_rmse_with_ones(prob_mat)
        diff = calc_diff_with_ones(prob_mat)
        self.logger.info('{}, {:>3}, {:>3}'.format(answer_type, increase_count, decrease_count))
        self.logger.info(float2str_list(diff))
        self.logger.info(float2str_list(rmse))
