##### Intelligent Tutoring System #####
import tensorflow as tf

import argparse, os, sys
sys.path.append('DKVMN')
sys.path.append('DQN')

from agent import *
from model import *
from data_loader import *

def str2bool(s):
    if s.lower() in ('yes', 'y', '1', 'true', 't'):
        return True
    elif s.lower() in ('no', 'n', '0', 'false', 'f'):
        return False

def setHyperParamsForDataset(args):
    if args.dataset == 'assist2009_updated':
        args.batch_size = 32
        args.memory_size = 20
        args.memory_key_state_dim = 50
        args.memory_value_state_dim = 200
        args.final_fc_dim = 50
        args.n_questions = 110
        args.seq_len = 200
        args.data_name = 'assist2009_updated'

    elif args.dataset == 'synthetic':
        args.batch_size = 32
        args.memory_size = 5 
        args.memory_key_state_dim = 10
        args.memory_value_state_dim = 10
        args.final_fc_dim = 50
        args.n_questions = 50
        args.seq_len = 50
        args.data_name = 'naive_c5_q50_s4000_v1'

    elif args.dataset == 'assist2015':
        args.batch_size = 50 
        args.memory_size = 20
        args.memory_key_state_dim = 50
        args.memory_value_state_dim = 100
        args.final_fc_dim = 50
        args.n_questions = 100
        args.seq_len = 200
        args.data_name = 'assist2015'


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--prefix', type=str, default='')

        ########## Control flag ##########
        parser.add_argument('--dkvmn_train', type=str2bool, default='f')
        parser.add_argument('--dkvmn_test', type=str2bool, default='f')
        parser.add_argument('--dqn_train', type=str2bool, default='f')
        parser.add_argument('--dqn_test', type=str2bool, default='f')
        parser.add_argument('--gpu_id', type=str, default='0')

        ########## Ideal test for DKVMN
        parser.add_argument('--dkvmn_ideal_test', type=str2bool, default='f')
        
        ########## DKVMN ##########
        parser.add_argument('--dataset', type=str, choices=['synthetic', 'assist2009_updated','assist2015','STATICS'], default='assist2009_updated')
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--init_from', type=str2bool, default='t')
        parser.add_argument('--show', type=str2bool, default='f')

        parser.add_argument('--anneal_interval', type=int, default=20)
        parser.add_argument('--maxgradnorm', type=float, default=50.0)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--initial_lr', type=float, default=0.05)

        parser.add_argument('--dkvmn_checkpoint_dir', type=str, default='DKVMN/checkpoint')
        parser.add_argument('--dkvmn_log_dir', type=str, default='DKVMN/log')
        parser.add_argument('--data_dir', type=str, default='DKVMN/data')
        parser.add_argument('--data_name', type=str, default='assist2009_updated')

        ########## Modified DKVMN ##########
        parser.add_argument('--knowledge_growth', type=str, choices=['origin', 'value_matrix', 'read_content', 'summary', 'pred_prob'], default='value_matrix')
        parser.add_argument('--add_signal_activation', type=str, choices=['tanh', 'sigmoid', 'relu'], default='sigmoid')
        parser.add_argument('--erase_signal_activation', type=str, choices=['tanh', 'sigmoid', 'relu'], default='sigmoid')
        parser.add_argument('--summary_activation', type=str, choices=['tanh', 'sigmoid', 'relu'], default='sigmoid')
        
        parser.add_argument('--write_type', type=str, choices=['add_off_erase_off', 'add_off_erase_on', 'add_on_erase_off', 'add_on_erase_on'], default='add_on_erase_on')
       
        ##### Default(STATICS) hyperparameter #####
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--memory_size', type=int, default=50)
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=100)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=1223)
        parser.add_argument('--seq_len', type=int, default=200)


        ########## DQN ##########
        parser.add_argument('--batch_size_dqn', type=int, default=32)
        parser.add_argument('--max_step', type=int, default=50000)
        parser.add_argument('--max_exploration_step', type=int, default=50000)

        parser.add_argument('--replay_memory_size', type=int, default=10000)

        parser.add_argument('--discount_factor', type=float, default=0.95)
        parser.add_argument('--eps_init', type=float, default=1.0)
        parser.add_argument('--eps_min', type=float, default=0.1)
        parser.add_argument('--eps_test', type=float, default=-1)

        parser.add_argument('--training_start_step', type=int, default=100)
        parser.add_argument('--train_interval', type=int, default=1)
        parser.add_argument('--copy_interval', type=int, default=500)
        parser.add_argument('--save_interval', type=int, default=5000)
        parser.add_argument('--show_interval', type=int, default=5000)
        parser.add_argument('--episode_maxstep', type=int, default=500)
        parser.add_argument('--learning_rate', type=float, default=0.01)

        parser.add_argument('--dqn_checkpoint_dir', type=str, default='DQN/checkpoint')
        parser.add_argument('--dqn_tb_log_dir', type=str, default='DQN/tb_log')

        parser.add_argument('--state_type', type=str, choices=['value', 'mastery'], default='value')
        parser.add_argument('--reward_type', type=str, choices=['value', 'read', 'summary', 'prob', 'mastery'], default='value')
        parser.add_argument('--test_policy_type', type=str, choices=['random', 'dqn', 'prob_max', 'prob_min'], default='dqn')

        parser.add_argument('--num_test_episode', type=int, default=100)

        parser.add_argument('--logging_level', type=str, choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'], default='INFO')

        parser.add_argument('--terminal_condition', type=str, choices=['prob, mastery'], default='prob')
        parser.add_argument('--terminal_threshold', type=float, default=0.9)

        myArgs = parser.parse_args()
        setHyperParamsForDataset(myArgs)

        if myArgs.test_policy_type != 'dqn':
            myArgs.dqn_train = False 

        ### check dkvmn dir ###
        myArgs.dkvmn_checkpoint_dir = os.path.join(myArgs.dkvmn_checkpoint_dir, myArgs.dataset)
        if not os.path.exists(myArgs.dkvmn_checkpoint_dir): 
            os.makedirs(myArgs.dkvmn_checkpoint_dir)

        if not os.path.exists(myArgs.dkvmn_log_dir):
            os.makedirs(myArgs.dkvmn_log_dir)

        data = DATA_LOADER(myArgs.n_questions, myArgs.seq_len, ',')
        data_directory = os.path.join(myArgs.data_dir, myArgs.dataset)

        ### check dqn dir ###
        if not os.path.exists(myArgs.dqn_checkpoint_dir):
            os.makedirs(myArgs.dqn_checkpoint_dir)
        if not os.path.exists(myArgs.dqn_tb_log_dir):
            os.makedirs(myArgs.dqn_tb_log_dir)

        os.environ["CUDA_VISIBLE_DEVICES"] = myArgs.gpu_id 
        run_config = tf.ConfigProto()
        #run_config.log_device_placement = True
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=run_config) as sess:
            
            dkvmn = DKVMNModel(myArgs, sess, name='DKVMN')
            ##### DKVMN #####
            if myArgs.dkvmn_train:
                train_data_path = os.path.join(data_directory, myArgs.data_name + '_train1')
                valid_data_path = os.path.join(data_directory, myArgs.data_name + '_valid1')

                train_q_data, train_qa_data = data.load_data(train_data_path)
                print('Train data loaded')
                valid_q_data, valid_qa_data = data.load_data(valid_data_path)
                print('Valid data loaded')
                print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))
                print('Start training')
                print(myArgs.seq_len)
                dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data)

            if myArgs.dkvmn_test:
                test_data_path = os.path.join(data_directory, myArgs.data_name + '_test')
                test_q_data, test_qa_data = data.load_data(test_data_path)
                print('Test data loaded')
                dkvmn.test(test_q_data, test_qa_data)
    
            if myArgs.dkvmn_ideal_test:
                myArgs.batch_size = 1
                myArgs.seq_len = 1
                dkvmn.init_step()
                dkvmn.ideal_test()
            
            ##### DQN #####
            if myArgs.dqn_train or myArgs.dqn_test:
                sess.run(tf.global_variables_initializer()) 
          
                myArgs.batch_size = 1
                myArgs.seq_len = 1
                myAgent = DKVMNAgent(myArgs, sess, dkvmn)
                dkvmn.init_step()

            if myArgs.dqn_train:
                if os.path.exists('./train.csv'):
                    os.system("rm train.csv")
                myAgent.train()

            if myArgs.dqn_test:
                myAgent.play()
    
    except KeyboardInterrupt:
        myArgs.train = False
        myAgent.play(3, False)
        myAgent.save()
        sess.close()

if __name__ == '__main__':
    main()
