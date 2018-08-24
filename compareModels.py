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
                      # k_growth, activation, counter, concept_counter, niLoss
                        # ['origin', 'tanh', False, False, 0.0 ],
                        # ['origin', 'sigmoid', False, False, 0.0 ],
                        # ['summary', 'sigmoid', False, False, 0.0],
                        ['origin', 'sigmoid', False, True, 0.1]
                        # ['origin', 'sigmoid', True, False, 0.0]
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

    converge_speed_seq_list = []
    converge_idx_list = []

    latent_seq_list = []
    latent_idx_list = []

    tf.reset_default_graph()
    # TODO : importing to DKVMN model as reset_argument?

    args.knowledge_growth = target_spec[0]
    args.summary_activation = target_spec[1]
    args.using_counter_graph = target_spec[2]
    args.using_concept_counter_graph = target_spec[3]
    args.negative_influence_loss_weight = target_spec[4]
    # args.convergence_loss_weight = target_spec[4]

    for repeat_idx in range(1):
        args.repeat_idx = repeat_idx

        if args.repeat_idx == 0:
            args.batch_size = default_batch_size
            args.seq_len = default_seq_len
            # print(args.batch_size)
            # print(args.seq_len)
            dkvmn = DKVMNModel(args, name='DKVMN')
            # graph = dkvmn.build_step_dkvmn_graph()
            graph = dkvmn.build_step_dkvmn_graph()

        with tf.Session(config = run_config, graph = graph) as sess:
            tf.global_variables_initializer().run()
            dkvmn.set_session(sess)
            dkvmn.load()

            # auc, acc
            '''
            _, _, auc, acc = dkvmn.test(test_q_data, test_qa_data)
            auc_list.append(auc)
            acc_list.append(acc)
            '''

            # ruc
            aDKVMN = DKVMNAnalyzer(args, dkvmn)

            '''
            ruc = aDKVMN.test_negative_influence(update_value_matrix_flag)
            print(ruc)
            ruc_list.append(ruc)
            '''

            # convergence speed
            # aDKVMN = DKVMNAnalyzer(args, dkvmn)
            '''
            converge_speed_seq, converge_idx = aDKVMN.test_converge_speed(answer_type, update_value_matrix_flag)
            converge_speed_seq_list.append(float2str_list(converge_speed_seq))
            converge_idx_list.append(converge_idx)
            '''

            # latent learning
            # '''
            latent_seq = aDKVMN.test_latent_learning(answer_type)
            latent_seq_list.append(latent_seq)
            # latent_idx_list.append(latent_idx)
            # '''

    '''
    logger.info('AUC')
    logger.info(float2str_list(auc_list))

    logger.info('ACC')
    logger.info(float2str_list(acc_list))
    '''

    logger.info('RUC')
    logger.info(ruc_list)

    logger.info('Adapive converge')
    logger.info(float2str_list(converge_idx_list))
    logger.info(converge_speed_seq_list)

    logger.info('Latent learning')
    logger.info(latent_seq_list)
