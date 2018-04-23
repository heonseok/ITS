import numpy as np
import os, time, sys
import tensorflow as tf
sys.path.append('DKVMN')
sys.path.append('DQN')


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

#from utils import *

from model import *
from dkvmn_analysis import *
from setup import * 

target_model_list = [ 
                      # k_growth   summary    cLossW  cGraph
                      #[ 'origin' , 'sigmoid', 0.0   , False ],
                      [ 'summary', 'sigmoid', 0.0002, True  ],
                      [ 'summary', 'sigmoid', 0.0   , False ],
                    ]

args, run_config = setup()

args.dkvmn_train = False
args.dkvmn_test  = False

print(args)


print(target_model_list)
default_batch_size = args.batch_size
default_seq_len = args.seq_len


for model_idx, target_spec in enumerate(target_model_list):
    print(target_spec)
    tf.reset_default_graph()
    # TODO : importing to DKVMN model as reset_argument?

    args.knowledge_growth = target_spec[0]
    args.summary_activation = target_spec[1]
    args.counter_loss_weight = target_spec[2]
    args.using_counter_graph = target_spec[3]

    if args.repeat_idx == 0:
        args.batch_size = default_batch_size 
        args.seq_len = default_seq_len 
        print(args.batch_size)
        print(args.seq_len)
        dkvmn = DKVMNModel(args, name='DKVMN')
        graph = dkvmn.build_step_dkvmn_graph()

        with tf.Session(config = run_config, graph = graph) as sess:
            tf.global_variables_initializer().run()
            dkvmn.set_session(sess)
            dkvmn.load()

            aDKVMN = DKVMNAnalyzer(args, dkvmn)

            aDKVMN.test()
