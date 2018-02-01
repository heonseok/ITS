import numpy as np
import os, time
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

#from logger import *
import logging

from model import *

import pickle


class ClusteredDKVMN():
    def __init__(self, args, sess, k, baseDKVMN, name='ClustredDKVMN'):  
        self.args = args
        self.name = name
        self.sess = sess
        self.k = k
        self.baseDKVMN = baseDKVMN
        
        self.selected_mastery_index = self.args.target_mastery_index - 1 

        self.logger = logging.getLogger('cDKVMN')
        self.logger.setLevel(eval('logging.{}'.format(self.args.logging_level)))

        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info('Initializing Clustered DKVMN')

        self.checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.baseDKVMN.model_dir, '{}_masteryIdx_{}_clustered'.format(self.selected_mastery_index+1, self.k))
        #self.checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.baseDKVMN.model_dir, '{}_masteryIdx'.format(self.selected_mastery_index), '{}_clustered'.format(self.k))
        self.kmeans_path = os.path.join(self.checkpoint_dir, 'kmeans.pkl')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def clustering_exists(self):
        return os.path.exists(self.kmeans_path) 

    def save_clusters(self, kmeans):
        pickle.dump(kmeans, open(self.kmeans_path, "wb"))

    def load_clusters(self):
        return pickle.load(open(self.kmeans_path, "rb"))

    def train(self, train_q, train_qa, valid_q, valid_qa, test_q, test_qa):
        self.logger.info('Start training')
        self.logger.debug('Number of train_q: {}'.format(len(train_q)))

        train_mastery_level_seq = self.get_mastery_level(train_q, train_qa)
        train_selected_mastery_level = train_mastery_level_seq[self.selected_mastery_index]

        if not self.clustering_exists():
            self.build_clusters(train_q, train_qa, train_selected_mastery_level)

        kmeans = self.load_clusters()
        #kmeans = pickle.load(open(self.kmeans_path, "rb"))
        clustered_train_q, clustered_train_qa = self.clustering_students(train_q, train_qa, train_selected_mastery_level, kmeans)

        valid_mastery_level_seq = self.get_mastery_level(valid_q, valid_qa)
        valid_selected_mastery_level = valid_mastery_level_seq[self.selected_mastery_index]
        clustered_valid_q, clustered_valid_qa = self.clustering_students(valid_q, valid_qa, valid_selected_mastery_level, kmeans)

        test_mastery_level_seq = self.get_mastery_level(test_q, test_qa)
        test_selected_mastery_level = test_mastery_level_seq[self.selected_mastery_index]
        clustered_test_q, clustered_test_qa = self.clustering_students(test_q, test_qa, test_selected_mastery_level, kmeans)

        for idx in range(self.k):
            #target_q = clustered_q[idx]
            #target_qa = clustered_qa[idx]
            #print(len(target_q))

            self.train_subDKVMN(idx, clustered_train_q[idx], clustered_train_qa[idx], clustered_valid_q[idx], clustered_valid_qa[idx])
            b_pred_list, b_target_list, c_pred_list, c_taget_list = self.test_subDKVMN(idx, clustered_test_q[idx], clustered_test_qa[idx])

            if idx == 0:
                #total_mastery_level_seq = batch_mastery_level_seq
                b_total_pred_list = b_pred_list
                b_total_target_list = b_target_list

                c_total_pred_list = c_pred_list
                c_total_target_list = c_target_list
            else:
                #total_mastery_level_seq = np.concatenate((total_mastery_level_seq, batch_mastery_level_seq), axis=1) 
                b_total_pred_list = np.concatenate((b_total_pred_list, b_pred_list), axis=0)
                b_total_target_list = np.concatenate((b_total_target_list, b_target_list), axis=0)

                c_total_pred_list = np.concatenate((c_total_pred_list, c_pred_list), axis=0)
                c_total_target_list = np.concatenate((c_total_target_list, c_target_list), axis=0)


        b_auc, b_acc = self.calculate_metric(b_total_target_list, b_total_pred_list)
        c_auc, c_acc = self.calculate_metric(c_total_target_list, c_total_pred_list)

        print('Base     : auc {}, acc {}'.format(b_auc, b_acc))
        print('Clustred : auc {}, acc {}'.format(c_auc, c_acc))

    def test(self, test_q, test_qa):
        self.logger.info('Start testing')

        kmeans = self.load_clusters()
        #kmeans = pickle.load(open(self.kmeans_path, "rb"))

        # calculate matsery level for test dataset
        #last_mastery_level = self.get_mastery_level(test_q, test_qa)
        test_mastery_level_seq = self.get_mastery_level(test_q, test_qa)
        test_selected_mastery_level = test_mastery_level_seq[self.selected_mastery_index]

        clustered_test_q, clustered_test_qa = self.clustering_students(test_q, test_qa, test_selected_mastery_level, kmeans)
        
        for idx in range(self.k):
            pred_list, target_list = self.test_subDKVMN(idx, clustered_test_q[idx], clustered_test_qa[idx])
            #total_pred_list.append(pred_list)
            #total_target_list.append(target_list)

            if idx == 0:
                #total_mastery_level_seq = batch_mastery_level_seq
                total_pred_list = pred_list
                total_target_list = target_list
            else:
                #total_mastery_level_seq = np.concatenate((total_mastery_level_seq, batch_mastery_level_seq), axis=1) 
                total_pred_list = np.concatenate((total_pred_list, pred_list), axis=0)
                total_target_list = np.concatenate((total_target_list, target_list), axis=0)


        auc, acc = self.calculate_metric(total_target_list, total_pred_list)
        print(auc)
        print(acc)
      

        '''
        total_pred_list = list()
        total_target_list = list()
        for idx in range(self.k):
           pred_list, target_list = self.test_subDKVMN(idx, test_q, test_qa)
           total_pred_list.append(pred_list)
           total_target_list.append(target_list)
        '''
    
    def calculate_metric(self, target, pred):

        right_index = (target != -1.)

        right_target = target[right_index]
        right_pred = pred[right_index]

        auc = roc_auc_score(right_target, right_pred)

        right_pred[right_pred > 0.5] = 1.0
        right_pred[right_pred <= 0.5] = 0.0
        acc = accuracy_score(right_target, right_pred)

        return auc, acc 

    def get_mastery_level(self, q_data, qa_data):
        self.logger.info('Get mastery level')

        # load base DKVMN
        if self.baseDKVMN.load():
            self.logger.info('Base DKVMN ckpt loaded')
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
        
        
    def build_clusters(self, q_data, qa_data, last_mastery_level):
        self.logger.info('Building clusters')

        kmeans = KMeans(n_clusters=self.k).fit(last_mastery_level)
        self.save_clusters(kmeans)
        #pickle.dump(kmeans, open(self.kmeans_path, "wb"))


    def clustering_students(self, q_data, qa_data, selected_mastery_level, kmeans):
        self.logger.info('Clustering students')

        #kmeans = KMeans(n_clusters=self.k).fit(selected_mastery_level)
        cluster_labels = kmeans.predict(selected_mastery_level)

        q_data = q_data[0:len(selected_mastery_level), :]
        qa_data = qa_data[0:len(selected_mastery_level), :]

        clustered_q_data = list()
        clustered_qa_data = list()

        for cluster_id in range(self.k):
            target_idx = (cluster_labels == cluster_id)

            clustered_q_data.append(q_data[target_idx])
            clustered_qa_data.append(qa_data[target_idx])

            print('{}-th cluster size : {}'.format(cluster_id, len(clustered_q_data[cluster_id])))

        return clustered_q_data, clustered_qa_data


    def train_subDKVMN(self, idx, train_q, train_qa, valid_q, valid_qa):
        self.logger.info('Train {}-th sub DKVMN'.format(idx))

        # split target to train and valid 
        #train_q, valid_q, train_qa, valid_qa = train_valid_split(tain_q, tain_qa, valid_size=0.2)
        # save sub dkvmn as original/k_clusteredDKVMN/subDKVMN_i
        sub_checkpoint_dir = os.path.join(self.checkpoint_dir, 'subDKVMN{}'.format(idx))
        self.baseDKVMN.train(train_q, train_qa, valid_q, valid_qa, early_stop=True, checkpoint_dir=sub_checkpoint_dir, selected_mastery_index=self.selected_mastery_index)

    def test_subDKVMN(self, idx, test_q, test_qa):
        self.logger.info('Test {}-th sub DKVMN'.format(idx))

        #test_q = test_q[:,self.selected_mastery_index+1:-1]
        #test_qa = test_qa[:,self.selected_mastery_index+1:-1]

        sub_checkpoint_dir = os.path.join(self.checkpoint_dir, 'subDKVMN{}'.format(idx))
        b_pred_list, b_target_list = self.baseDKVMN.test(test_q, test_qa)
        c_pred_list, c_target_list = self.baseDKVMN.test(test_q, test_qa, sub_checkpoint_dir, self.selected_mastery_index)

        return b_pred_list, b_target_list, c_pred_list, c_target_list

    '''
    def calculate_result(self, total_pred_list, target_list):
        self.logger.info('Calculate results')
        total_auc_list = list()
        total_acc_list = list()

        for cluster_id in range(self.k):
            pred_list = total_pred_list[cluster_id]

            auc_list = list()
            acc_list = list()

            for idx in range(pred_list.shape[0]):
                target_seq = target_list[idx]
                pred_seq = pred_list[idx]

                right_index = (target_seq != -1.)

                print(target_seq[~right_index])
                print('ROC')
                print(pred_seq[right_index])
                print(target_seq[right_index])
                print(roc_auc_score(target_seq[right_index], pred_seq[right_index]))

                try:
                    right_target = target_seq[right_index]
                    right_pred = pred_seq[right_index]

                    right_pred[right_pred > 0.5] = 1.0
                    right_pred[right_pred <= 0.5] = 0.0
                    acc_list.append(accuracy_score(right_target, right_pred))
                except:
                    print('DEAD')
                    pass

            total_acc_list.append(acc_list)
        # TODO using max instead of greater 
        selected_indices = np.greater(total_acc_list[1], total_acc_list[0]).astype(int)


        selected_pred_list = np.zeros_like(target_list)  
        for idx in range(target_list.shape[0]):
            selected_idx = selected_indices[idx]
            selected_pred_list[idx] = total_pred_list[selected_idx][idx]
         
        right_index = (target_list != -1.) 
        target_list = target_list[right_index]
        selected_pred_list = selected_pred_list[right_index]

        print(roc_auc_score(target_list, selected_pred_list))

        selected_pred_list[selected_pred_list > 0.5] = 1.0
        selected_pred_list[selected_pred_list <= 0.5] = 0.0
        print(accuracy_score(target_list, selected_pred_list))
                 
        
        # scenario #1-1 
        # subDKVMN for each sequence

        # scenario #1-2 : it seems not reasonable
        # subDKVMN for each question

        # scenario #1-3
        # baseDKVMN for n-length questions and subDKVMn for n+1 ~ 

        # scenario #2-1 
        # ensemble 

        # scenario #2-2
        # weighted ensemble

    '''
