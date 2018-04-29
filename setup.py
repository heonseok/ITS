import argparse, os, sys
import tensorflow as tf

def str2bool(s):
    if s.lower() in ('yes', 'y', '1', 'true', 't'):
        return True
    elif s.lower() in ('no', 'n', '0', 'false', 'f'):
        return False

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--logging_level', type=str, choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'], default='INFO')

    ########## Control flag ##########
    parser.add_argument('--dkvmn_train', type=str2bool, default='f')
    parser.add_argument('--dkvmn_test', type=str2bool, default='f')
    parser.add_argument('--dkvmn_ideal_test', type=str2bool, default='f')

    parser.add_argument('--dkvmn_clustering_actions', type=str2bool, default='f')
    parser.add_argument('--dkvmn_analysis', type=str2bool, default='f')

    parser.add_argument('--dqn_train', type=str2bool, default='f')
    parser.add_argument('--dqn_test', type=str2bool, default='f')

    parser.add_argument('--using_cpu', type=str2bool, default='f')
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--repeat_idx', type=int, default=0)

    ########## Data preprocessing #########
    parser.add_argument('--remove_infrequent_skill', type=str2bool, default='f')
    parser.add_argument('--frequency_th', type=int, default=50)

    parser.add_argument('--remove_short_seq', type=str2bool, default='f')
    parser.add_argument('--short_seq_len_th', type=int, default=20)

    parser.add_argument('--split_data_flag', type=str2bool, default='f')

    ########## DKVMN ##########
    parser.add_argument('--dataset', type=str, choices=['synthetic', 'assist2009_updated','assist2015','STATICS'], default='assist2009_updated')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--init_from', type=str2bool, default='f')
    parser.add_argument('--show', type=str2bool, default='f')
    parser.add_argument('--early_stop', type=str2bool, default='f')
    parser.add_argument('--early_stop_th', type=int, default=20)

    parser.add_argument('--anneal_interval', type=int, default=20)
    parser.add_argument('--maxgradnorm', type=float, default=50.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--initial_lr', type=float, default=0.6)

    parser.add_argument('--dkvmn_checkpoint_dir', type=str, default='DKVMN/checkpoint')
    parser.add_argument('--dkvmn_log_dir', type=str, default='DKVMN/log')
    parser.add_argument('--data_dir', type=str, default='DKVMN/data')
    parser.add_argument('--data_name', type=str, default='assist2009_updated')

    '''
    parser.add_argument('--train_postfix', type=str, default='train1')
    parser.add_argument('--valid_postfix', type=str, default='valid1')
    parser.add_argument('--test_postfix', type=str, default='test')
    '''


    ########## Modified DKVMN ##########
    parser.add_argument('--knowledge_growth', type=str, choices=['origin', 'value_matrix', 'read_content', 'summary', 'pred_prob', 'mastery'], default='origin')
    parser.add_argument('--add_activation', type=str, choices=['tanh', 'sigmoid', 'relu'], default='tanh')
    parser.add_argument('--erase_activation', type=str, choices=['tanh', 'sigmoid', 'relu'], default='sigmoid')
    parser.add_argument('--summary_activation', type=str, choices=['tanh', 'sigmoid', 'relu'], default='tanh')

    parser.add_argument('--write_type', type=str, choices=['add_off_erase_off', 'add_off_erase_on', 'add_on_erase_off', 'add_on_erase_on'], default='add_on_erase_on')

    parser.add_argument('--using_counter_graph', type=str2bool, default='t')
    parser.add_argument('--counter_embedding_dim', type=int, default=64)
    #parser.add_argument('--using_counter_loss', type=str2bool, default='f')
    parser.add_argument('--counter_loss_weight', type=float, default=0.0)
    parser.add_argument('--decrease_loss_weight', type=float, default=0.0)

    #parser.add_argument('--using_weighted_update', type=str2bool, default='f')
    #parser.add_argument('--weighted_update_type', type=str, choices=['prob_diff, softmax'], default='prob_diff')


    ########## Clustered DKVMN ##########
    parser.add_argument('--clustered_dkvmn_train', type=str2bool, default='f')
    parser.add_argument('--clustered_dkvmn_test', type=str2bool, default='f')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--target_mastery_index', type=int, default=5)

    ##### Default(STATICS) hyperparameter #####
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--key_memory_dim', type=int, default=50)
    parser.add_argument('--value_memory_dim', type=int, default=100)
    parser.add_argument('--summary_dim', type=int, default=50)
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

    parser.add_argument('--state_type', type=str, choices=['value', 'mastery'], default='mastery')
    parser.add_argument('--reward_type', type=str, choices=['value', 'read', 'summary', 'prob', 'mastery'], default='mastery')
    parser.add_argument('--test_policy_type', type=str, choices=['random', 'dqn', 'prob_max', 'prob_min'], default='dqn')

    parser.add_argument('--terminal_condition', type=str, choices=['prob', 'mastery'], default='prob')
    parser.add_argument('--terminal_condition_type', type=str, choices=['pos', 'posneg', 'when_to_stop'], default='posneg')
    parser.add_argument('--terminal_threshold', type=float, default=0.9)

    parser.add_argument('--num_test_episode', type=int, default=100)

    parser.add_argument('--sampling_action_type', type=str, choices=['uniform', 'clipping'], default='uniform')

    myArgs = parser.parse_args()
    #setHyperParamsForDataset(myArgs)

    if myArgs.dataset == 'assist2009_updated':
        myArgs.batch_size = 32 #512 
        myArgs.memory_size = 20
        myArgs.memory_key_state_dim = 50
        myArgs.memory_value_state_dim = 200
        myArgs.final_fc_dim = 50
        myArgs.n_questions = 110
        myArgs.seq_len = 200
        myArgs.data_name = 'assist2009_updated'

    elif myArgs.dataset == 'synthetic':
        myArgs.batch_size = 32
        myArgs.memory_size = 5 
        myArgs.memory_key_state_dim = 10
        myArgs.memory_value_state_dim = 10
        myArgs.final_fc_dim = 50
        myArgs.n_questions = 50
        myArgs.seq_len = 50
        myArgs.data_name = 'naive_c5_q50_s4000_v1'

    elif myArgs.dataset == 'assist2015':
        myArgs.batch_size = 50 
        myArgs.memory_size = 20
        myArgs.memory_key_state_dim = 50
        myArgs.memory_value_state_dim = 100
        myArgs.final_fc_dim = 50
        myArgs.n_questions = 100
        myArgs.seq_len = 200
        myArgs.data_name = 'assist2015'
        myArgs.episode_maxstep = myArgs.n_questions * 10

    if myArgs.test_policy_type != 'dqn':
        myArgs.dqn_train = False 

    ### check dkvmn dir ###
    myArgs.dkvmn_checkpoint_dir = os.path.join(myArgs.dkvmn_checkpoint_dir, myArgs.dataset)
    if not os.path.exists(myArgs.dkvmn_checkpoint_dir): 
        os.makedirs(myArgs.dkvmn_checkpoint_dir)
    if not os.path.exists(myArgs.dkvmn_log_dir):
        os.makedirs(myArgs.dkvmn_log_dir)

    ### check dqn dir ###
    if not os.path.exists(myArgs.dqn_checkpoint_dir):
        os.makedirs(myArgs.dqn_checkpoint_dir)
    if not os.path.exists(myArgs.dqn_tb_log_dir):
        os.makedirs(myArgs.dqn_tb_log_dir)

    run_config = tf.ConfigProto()

    if myArgs.using_cpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1' 
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = myArgs.gpu_id 
        #run_config.log_device_placement = True
        run_config.gpu_options.allow_growth = True

    return myArgs, run_config
