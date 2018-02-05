import numpy as np
import os, time
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score

import dkvmn_utils

import logging

from model import *

import pickle


class ClusteredDKVMN():
    def __init__(self, args, sess, baseDKVMN):  
        self.args = args
        self.sess = sess
        self.k = self.args.k 
        self.baseDKVMN = baseDKVMN
        
        self.selected_mastery_index = self.args.target_mastery_index - 1 

        self.logger = dkvmn_utils.set_logger('cDKVMN', 'clustered_dkvmn.log')
        self.logger.setLevel(eval('logging.{}'.format(self.args.logging_level)))

        self.logger.debug('Initializing Clustered DKVMN')

        self.checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.baseDKVMN.model_dir, '{}_masteryIdx_{}_clustered'.format(self.selected_mastery_index+1, self.k))
        self.kmeans_path = os.path.join(self.checkpoint_dir, 'kmeans.pkl')

        self.logger.info('#'*120)
        self.logger.info(self.checkpoint_dir)
        self.logger.info('{:<20s}: {:>6d}'.format('Mastery idx', self.selected_mastery_index+1))
        self.logger.info('{:<20s}: {:>6d}'.format('Number of clusters', self.k))

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def clustering_exists(self):
        return os.path.exists(self.kmeans_path) 

    def save_clusters(self, kmeans):
        pickle.dump(kmeans, open(self.kmeans_path, "wb"))

    def load_clusters(self):
        return pickle.load(open(self.kmeans_path, "rb"))

    def train(self, train_q, train_qa, valid_q, valid_qa, test_q, test_qa):
        self.logger.debug('Start training')

        if not self.clustering_exists():
            self.build_clusters(train_q, train_qa)
        kmeans = self.load_clusters()

        self.logger.info('{:<20s}: {:>6d}'.format('Number of train_q', len(train_q)))
        clustered_train_q, clustered_train_qa = self.clustering_students(train_q, train_qa, kmeans)

        self.logger.info('{:<20s}: {:>6d}'.format('Number of valid_q', len(valid_q)))
        clustered_valid_q, clustered_valid_qa = self.clustering_students(valid_q, valid_qa, kmeans)

        for idx in range(self.k):
            self.train_subDKVMN(idx, clustered_train_q[idx], clustered_train_qa[idx], clustered_valid_q[idx], clustered_valid_qa[idx])

        self.logger.info('\n')

    def test(self, test_q, test_qa):
        self.logger.debug('Start testing')

        kmeans = self.load_clusters()
        self.logger.info('{:<20s}: {:>6d}'.format('Number of test_q', len(test_q)))
        clustered_test_q, clustered_test_qa = self.clustering_students(test_q, test_qa, kmeans)

        b_auc_string = 'Base auc'.ljust(20) + ':'
        b_acc_string = 'Base acc'.ljust(20) + ':'
        c_auc_string = 'Clustred auc'.ljust(20) + ':'
        c_acc_string = 'Clustred acc'.ljust(20) + ':'
  
        for idx in range(self.k):
            b_pred_list, b_target_list, b_auc, b_acc, c_pred_list, c_target_list, c_auc, c_acc = self.test_subDKVMN(idx, clustered_test_q[idx], clustered_test_qa[idx])

            b_auc_string += ' {:.4f},'.format(b_auc)
            b_acc_string += ' {:.4f},'.format(b_acc)
            c_auc_string += ' {:.4f},'.format(c_auc)
            c_acc_string += ' {:.4f},'.format(c_acc)

            if idx == 0:
                b_total_pred_list = b_pred_list
                b_total_target_list = b_target_list

                c_total_pred_list = c_pred_list
                c_total_target_list = c_target_list
            else:
                b_total_pred_list = np.concatenate((b_total_pred_list, b_pred_list), axis=0)
                b_total_target_list = np.concatenate((b_total_target_list, b_target_list), axis=0)

                c_total_pred_list = np.concatenate((c_total_pred_list, c_pred_list), axis=0)
                c_total_target_list = np.concatenate((c_total_target_list, c_target_list), axis=0)

        total_b_auc, total_b_acc = dkvmn_utils.calculate_metric(b_total_target_list, b_total_pred_list)
        total_c_auc, total_c_acc = dkvmn_utils.calculate_metric(c_total_target_list, c_total_pred_list)

        self.logger.info(b_auc_string)
        self.logger.info(b_acc_string)
        self.logger.info(c_auc_string)
        self.logger.info(c_acc_string)

        self.logger.info('{:<20s}: auc {:.4f}, acc {:.4f}'.format('Total Base', total_b_auc, total_b_acc))
        self.logger.info('{:<20s}: auc {:.4f}, acc {:.4f}'.format('Total Clustered', total_c_auc, total_c_acc))

        self.logger.info('\n')
    

    def get_mastery_level(self, q_data, qa_data):
        self.logger.debug('Get mastery level')

        # load base DKVMN
        if self.baseDKVMN.load():
            self.logger.debug('Base DKVMN ckpt loaded')
        else: 
            raise Exception('Base DKVMN ckpt is not available')
        
        # inference mastery_level 
        steps = q_data.shape[0] // self.args.batch_size
        for s in range(steps):
            q_batch = q_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            qa_batch = qa_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]

            #q_batch = q_data[s*self.args.batch_size:min((s+1)*self.args.batch_size, q_data.shape[0]-1), :]
            #qa_batch = qa_data[s*self.args.batch_size:min((s+1)*self.args.batch_size, q_data.shape[0]-1), :]

            feed_dict = {self.baseDKVMN.q_data_seq: q_batch, self.baseDKVMN.qa_data_seq: qa_batch}
            batch_mastery_level_seq = np.squeeze(self.sess.run([self.baseDKVMN.mastery_level_seq], feed_dict=feed_dict))

            # seq_len+1, batch_size, num_concept 
            #print(np.shape(batch_mastery_level_seq))

            if s == 0:
                total_mastery_level_seq = batch_mastery_level_seq
            else:
                total_mastery_level_seq = np.concatenate((total_mastery_level_seq, batch_mastery_level_seq), axis=1) 

        return total_mastery_level_seq

        # seq_len+1, num_student, num_concpet
        #print(np.shape(total_mastery_level_seq))
        #last_mastery_level = total_mastery_level_seq[self.args.seq_len]

        #for i in range(len(last_mastery_level)):
            #print('{}th last_mastery_level: {}'.format(i,last_mastery_level[i]))

        #return last_mastery_level
        
        
    def build_clusters(self, q_data, qa_data):
        self.logger.debug('Building clusters')

        mastery_level_seq = self.get_mastery_level(q_data, qa_data)
        selected_mastery_level = mastery_level_seq[self.selected_mastery_index]

        kmeans = KMeans(n_clusters=self.k).fit(selected_mastery_level)
        self.save_clusters(kmeans)


    def clustering_students(self, q_data, qa_data, kmeans):
        self.logger.debug('Clustering students')

        mastery_level_seq = self.get_mastery_level(q_data, qa_data)
        selected_mastery_level = mastery_level_seq[self.selected_mastery_index]
        self.logger.debug('Selected mastery level shape: {}'.format(np.shape(selected_mastery_level)))

        cluster_labels = kmeans.predict(selected_mastery_level)

        q_data = q_data[0:len(selected_mastery_level), :]
        qa_data = qa_data[0:len(selected_mastery_level), :]

        clustered_q_data = list()
        clustered_qa_data = list()

        cluster_size_string = 'Size of clusters'.ljust(20) + ':'
        for cluster_id in range(self.k):
            target_idx = (cluster_labels == cluster_id)

            clustered_q_data.append(q_data[target_idx])
            clustered_qa_data.append(qa_data[target_idx])

            cluster_size_string += ' {:>6d},'.format(len(clustered_q_data[cluster_id]))

        self.logger.info(cluster_size_string)
        return clustered_q_data, clustered_qa_data


    def train_subDKVMN(self, idx, train_q, train_qa, valid_q, valid_qa):
        self.logger.debug('Train {}-th sub DKVMN'.format(idx))

        sub_checkpoint_dir = os.path.join(self.checkpoint_dir, 'subDKVMN{}'.format(idx))
        self.baseDKVMN.train(train_q, train_qa, valid_q, valid_qa, early_stop=True, checkpoint_dir=sub_checkpoint_dir, selected_mastery_index=self.selected_mastery_index)


    def test_subDKVMN(self, idx, test_q, test_qa):
        self.logger.debug('Test {}-th sub DKVMN'.format(idx))

        sub_checkpoint_dir = os.path.join(self.checkpoint_dir, 'subDKVMN{}'.format(idx))

        b_pred_list, b_target_list, b_auc, b_acc = self.baseDKVMN.test(test_q, test_qa)
        c_pred_list, c_target_list, c_auc, c_acc = self.baseDKVMN.test(test_q, test_qa, sub_checkpoint_dir, self.selected_mastery_index)

        return b_pred_list, b_target_list, b_auc, b_acc, c_pred_list, c_target_list, c_auc, c_acc
