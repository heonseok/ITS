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

from utils import *

logger = set_logger('mDKVMN', 'merging_dkvmn_analysis.log', 'INFO')
logger.debug('Merging DKVMN analyzing results')

target_model_list = [
                      # k_growth, activation, counter, niLoss, cLoss
                      ['summary', 'sigmoid', False, 0.0, 0.0],
                      ['summary', 'sigmoid', True, 0.0, 0.0002]
                    ]

args, run_config = setup()

args.dkvmn_train = False
args.dkvmn_test  = False

default_batch_size = args.batch_size
default_seq_len = args.seq_len

answer_type = 1
update_value_matrix_flag = True


# load test data
data_directory = os.path.join(args.data_dir, args.dataset)
test_data_path = os.path.join(data_directory, args.data_name + '_test.npz')

test_data = np.load(test_data_path)
test_q_data = test_data['q']
test_qa_data = test_data['qa']

for model_idx, target_spec in enumerate(target_model_list):
    logger.info(target_spec)

    # RUC : right update count
    ruc_list = []
    auc_list = []
    acc_list = []

    tf.reset_default_graph()
    # TODO : importing to DKVMN model as reset_argument?

    args.knowledge_growth = target_spec[0]
    args.summary_activation = target_spec[1]
    args.using_counter_graph = target_spec[2]
    args.negative_influence_loss_weight = target_spec[3]
    args.convergence_loss_weight = target_spec[4]

    for repeat_idx in range(10):
        args.repeat_idx = repeat_idx

        if args.repeat_idx == 0:
            args.batch_size = default_batch_size
            args.seq_len = default_seq_len
            # print(args.batch_size)
            # print(args.seq_len)
            dkvmn = DKVMNModel(args, name='DKVMN')
            # graph = dkvmn.build_step_dkvmn_graph()
            graph = dkvmn.build_dkvmn_graph()

        with tf.Session(config = run_config, graph = graph) as sess:
            tf.global_variables_initializer().run()
            dkvmn.set_session(sess)
            dkvmn.load()

            _, _, auc, acc = dkvmn.test(test_q_data, test_qa_data)
            auc_list.append(auc)
            acc_list.append(acc)

            '''
            aDKVMN = DKVMNAnalyzer(args, dkvmn)

            # aDKVMN.test()
            ruc = aDKVMN.test1_base(answer_type, update_value_matrix_flag)
            ruc_list.append(ruc)
            '''

    logger.info('AUC')
    logger.info(auc_list)

    logger.info('ACC')
    logger.info(acc_list)

    # logger.info('RUC')
    # logger.info(ruc_list)

