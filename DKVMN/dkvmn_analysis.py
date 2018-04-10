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

        self.logger = dkvmn_utils.set_logger('aDKVMN', self.args.prefix + 'dkvmn_analysis.log', self.args.logging_level)
        self.logger.debug('Initializing DKVMN Analyzer')

    def test(self):
        self.logger.info('#'*120)
        self.logger.info(self.dkvmn.model_dir)

        self.dkvmn.load()

        self.logger.info('Test 1')
        self.logger.info('Answer_type, 0, 0.001, 0.01, 0.05')
        self.test1_base(1)
        self.test1_base(0)
 
        self.logger.info('Test 2')
        self.logger.info('Answer_type, increase_count, decrease_count, diff, rmse')
        self.test2_base(1)
        self.test2_base(0)
    
    '''
    def test1_1(self):
        self.logger.info('Test 1-1')
        self.test1_base(1)
    
    def test1_2(self):
        self.logger.info('Test 1-2')
        self.test1_base(0)
        
    def test2_1(self):
        self.logger.info('Test 2-1')
        self.test2_base(1)
        
    def test2_2(self):
        self.logger.info('Test 2-2')
        self.test2_base(0)
    '''


    def test1_base(self, answer_type):
        '''
        Reponse of one action 
        '''
        th_list = [0, 0.001, 0.01, 0.05]


        init_counter = self.dkvmn.get_init_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter)
        
        answer = self.dkvmn.expand_dims(answer_type)
        skill_counter = 0
        right_updated_skill_counter = 0

        #self.logger.debug('Action, prob, diff, diff_avg, wrong_response')

        wrong_response = np.zeros_like(th_list, dtype=np.int32)
        right_updated_skill_counter = np.zeros_like(th_list, dtype=np.int32)

        for action_idx in range(self.num_actions):

            action = self.dkvmn.expand_dims(action_idx+1)
            value_matrix = self.dkvmn.update_value_matrix(init_value_matrix, action, answer, init_counter)
            
            counter = np.copy(init_counter)
            '''
            print('I')
            print(init_counter)
            print('C')
            print(counter)
            '''
            counter[0,action_idx + 1] += 1
            '''
            print('C_')
            print(counter)
            '''
            # TODO : check init_counter is correct. counter_with_incrase_one? 
            probs = self.dkvmn.get_prediction_probability(value_matrix, counter)
            #probs = self.dkvmn.get_prediction_probability(value_matrix, init_counter)

            probs_diff = probs - init_probs
            probs_diff_action = probs_diff[action_idx]
            probs_diff_avg = np.average(probs_diff)

            for idx, th in enumerate(th_list):
                if answer_type == 1:
                    wrong_response[idx] = np.sum(probs_diff < -th)

                    '''
                    if probs_diff_avg < 0:
                        skill_counter += 1
                    '''

                elif answer_type == 0:
                    wrong_response[idx] = np.sum(probs_diff > th)

                    '''
                    if probs_diff_avg > 0:
                        skill_counter += 1
                    '''

                if wrong_response[idx] == 0:
                    right_updated_skill_counter[idx] += 1


            #self.logger.debug('{:>3}, {: .4f}, {: .4f}, {: .4f}, {:>3d}, {:>3d}, {:>3d}, {:>3d}'.format(action_idx+1, init_probs[action_idx], probs_diff_action, probs_diff_avg, wrong_response[0], wrong_response[1], wrong_response[2], wrong_response[3]))

        #self.logger.info('Total number of wrong updated skills: {}'.format(skill_counter))
        self.logger.info('{}, {:>3}, {:>3}, {:>3}, {:>3}'.format(answer_type, right_updated_skill_counter[0],right_updated_skill_counter[1],right_updated_skill_counter[2],right_updated_skill_counter[3]))


    def test2_base(self, answer_type):
        ''' 
        Response of repeated action 
        '''
        # TODO : implement a counter
        repeat_num = 10

        #self.dkvmn.load()

        init_counter = self.dkvmn.get_init_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter)
        answer = self.dkvmn.expand_dims(answer_type)
   
        prob_mat = np.zeros((self.num_actions, repeat_num+1))

        for action_idx in range(self.num_actions):
            action = self.dkvmn.expand_dims(action_idx+1)
            value_matrix = np.copy(init_value_matrix)
            counter = np.copy(init_counter)

            #prob_list = list()
            #prob_list.append('{:.4f}'.format(init_probs[action_idx]))
            prob_mat[action_idx][0] = init_probs[action_idx]
            for repeat_idx in range(repeat_num):
                value_matrix = self.dkvmn.update_value_matrix(value_matrix, action, answer, counter)
                probs = self.dkvmn.get_prediction_probability(value_matrix, counter)
                #prob_list.append('{:.4f}'.format(probs[action_idx]))
                prob_mat[action_idx][repeat_idx+1] = probs[action_idx]
                counter[0,action_idx + 1] += 1
                #print(np.sum([action_idx]))
            #print(','.join(prob_list))
            #self.logger.info('{:>3}, {}'.format(action_idx+1, ','.join(prob_list)))

        #print(prob_mat) 
        increase_count, decrease_count = self.calc_seq_trend(prob_mat)
        #print(increase_count)
        #print(decrease_count)
        rmse = self.calc_rmse_with_ones(prob_mat)
        #print(rmse)
        diff = self.calc_diff_with_ones(prob_mat)
        #print(diff)
        self.logger.info('{}, {:>3}, {:>3}'.format(answer_type, increase_count, decrease_count))
        self.logger.info(self.str_format(diff))
        self.logger.info(self.str_format(rmse))

    def str_format(self, input_list):
        return ','.join('{:4f}'.format(item) for item in input_list)

    def calc_seq_trend(self, seq):
        zeros = np.zeros(shape=(seq.shape[0], 1))
        seq_right = np.hstack((seq, zeros))
        seq_left = np.hstack((zeros, seq))
 
        seq_diff = (seq_right-seq_left)[:,1:-1]
        increase_count = len(np.where(seq_diff>0)[0])
        decrease_count = len(np.where(seq_diff<0)[0])
 
        return increase_count, decrease_count

    def calc_rmse_with_ones(self, seq):
        ones = np.ones_like(seq)
        return np.sqrt(np.average(np.square(seq-ones), axis=0))
    
    def calc_diff_with_ones(self, seq):
        ones = np.ones_like(seq)
        return np.average((ones-seq), axis=0)
     
        
