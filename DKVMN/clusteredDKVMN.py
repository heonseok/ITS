import numpy as np
import os, time
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

#from logger import *
import logging

from model import *

class ClusteredDKVMN():
    def __init__(self, args, sess, k, baseDKVMN, name='ClustredDKVMN'):  
        self.args = args
        self.name = name
        self.sess = sess
        self.k = k
        self.baseDKVMN = baseDKVMN

        self.logger = logging.getLogger('cDKVMN')
        self.logger.setLevel(eval('logging.{}'.format(self.args.logging_level)))

        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info('Initializing Clustered DKVMN')


    def train(self):
        self.get_mastery_level()

        self.clustering_students()

        for idx in range(self.k):
            self.train_subDKVMN(idx)


    def test(self):
        for idx in range(self.k):
           self.test_subDKVMN(idx)

        self.calculate_result()
    

    def get_mastery_level(self):
        self.logger.info('Get mastery level')
        # load base DKVMN
        # save mastery_level as npy format
        
        
    def clustering_students(self):
        self.logger.info('Clustering students')
        # load npy file
        # k-means clustering 
        # save each cluster as npy file


    def train_subDKVMN(self, idx):
        self.logger.info('Train {}-th sub DKVMN'.format(idx))
        # load npy file
        # train dkvmn 
        # save sub dkvmn as original/k_clusteredDKVMN/subDKVMN_i


    def test_subDKVMN(self, idx):
        self.logger.info('Test {}-th sub DKVMN'.format(idx))
        # init total_test_result_list
        # load i-th subDKVMN model 
        # save result of i-th subDKVMN in total_test_result_list



    def calculate_result(self):
        self.logger.info('Calculate results')
        
        # get total_test_result_list

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

