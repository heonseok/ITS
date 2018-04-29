##### Intelligent Tutoring System #####
import tensorflow as tf

import argparse, os, sys
sys.path.append('DKVMN')
sys.path.append('DQN')

from agent import *
from model import *
from clusteredDKVMN import *
from dkvmn_analysis import *
from data_loader import *

from sklearn.model_selection import train_test_split 
from setup import *

def main():
    myArgs, run_config = setup()

    if myArgs.dkvmn_train or myArgs.dkvmn_test or myArgs.dkvmn_clustering_actions or myArgs.clustered_dkvmn_train or myArgs.clustered_dkvmn_test or myArgs.dkvmn_ideal_test: 
        ### Split data 
        data_directory = os.path.join(myArgs.data_dir, myArgs.dataset)
        data_path = os.path.join(data_directory, myArgs.data_name + '_data.csv')

        train_data_path = os.path.join(data_directory, myArgs.data_name + '_train.npz')
        valid_data_path = os.path.join(data_directory, myArgs.data_name + '_valid.npz')
        test_data_path = os.path.join(data_directory, myArgs.data_name + '_test.npz')

        if myArgs.split_data_flag == True:
            #print('#####SPLIT DATA#####')
            data = DATA_LOADER(myArgs, ',')
            total_q_data, total_qa_data = data.load_data(data_path)

            _train_q, test_q, _train_qa, test_qa = train_test_split(total_q_data, total_qa_data, test_size=0.2)
            train_q, valid_q, train_qa, valid_qa = train_test_split(_train_q, _train_qa, test_size=0.2)
            
            np.savez(train_data_path, q=train_q, qa=train_qa)
            np.savez(valid_data_path, q=valid_q, qa=valid_qa)
            np.savez(test_data_path, q=test_q, qa=test_qa)

        train_data = np.load(train_data_path)
        train_q_data = train_data['q']
        #print(train_q_data)
        train_qa_data = train_data['qa']

        #print('Number of train_q_data: {}'.format(len(train_q_data)))

        valid_data = np.load(valid_data_path)
        valid_q_data = valid_data['q']
        valid_qa_data = valid_data['qa']
        #print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))

        test_data = np.load(test_data_path)
        test_q_data = test_data['q']
        test_qa_data = test_data['qa']

    dkvmn = DKVMNModel(myArgs, name='DKVMN')

    if myArgs.dkvmn_train or myArgs.dkvmn_test:
        graph = dkvmn.build_dkvmn_graph()
        with tf.Session(config = run_config, graph = graph) as sess:
            dkvmn.set_session(sess)

            if myArgs.dkvmn_train:
                dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data, myArgs.early_stop)

            if myArgs.dkvmn_test:
                dkvmn.load()
                dkvmn.test(test_q_data, test_qa_data)

    if myArgs.dkvmn_analysis:
        graph = dkvmn.build_step_dkvmn_graph()
        with tf.Session(config = run_config, graph = graph) as sess:
            sess.run(tf.global_variables_initializer())

            dkvmn.set_session(sess)
            aDKVMN = DKVMNAnalyzer(myArgs, dkvmn)
            dkvmn.load()
            aDKVMN.test()
    '''

    # TODO : remove sess from __init__
    with tf.Session(config=run_config) as sess:
        dkvmn = DKVMNModel(myArgs, sess, name='DKVMN')

        ##### DKVMN #####
        if myArgs.dkvmn_train:
            dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data, myArgs.early_stop)

        if myArgs.dkvmn_test:
            dkvmn.test(test_q_data, test_qa_data)

        if myArgs.dkvmn_clustering_actions:
            dkvmn.clustering_actions()

        if myArgs.clustered_dkvmn_train:
            cDKVMN = ClusteredDKVMN(myArgs, sess, dkvmn)
            cDKVMN.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data, test_q_data, test_qa_data)

        if myArgs.clustered_dkvmn_test:
            cDKVMN = ClusteredDKVMN(myArgs, sess, dkvmn)
            cDKVMN.test(test_q_data, test_qa_data)

        if myArgs.dkvmn_analysis:
            #myArgs.batch_size = 1
            #myArgs.seq_len = 1
            aDKVMN = DKVMNAnalyzer(myArgs, sess, dkvmn)
            aDKVMN.test()

        if myArgs.dkvmn_ideal_test:
            #myArgs.batch_size = 1
            #myArgs.seq_len = 1
            #dkvmn.build_step_graph()
            dkvmn.ideal_test()
        
        ##### DQN #####
        if myArgs.dqn_train or myArgs.dqn_test:
            sess.run(tf.global_variables_initializer()) 
      
            #myArgs.batch_size = 1
            #myArgs.seq_len = 1
            myAgent = DKVMNAgent(myArgs, sess, dkvmn)
            #dkvmn.build_step_graph()

        if myArgs.dqn_train:
            if os.path.exists('./train.csv'):
                os.system("rm train.csv")
            myAgent.train()

        if myArgs.dqn_test:
            myAgent.play()
    '''

#except KeyboardInterrupt:
    #print('Program END')
    #sess.close()

if __name__ == '__main__':
    main()
