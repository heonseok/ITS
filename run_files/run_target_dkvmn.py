import os 

#################### DKVMN ####################
# 'origin', 'value_matrix', 'read_content', 'summary', 'pred_prob', 'mastery'
knowledge_growth_list = ['summary']

# 'sigmoid', 'tanh', 'relu'
summary_activation_list = ['sigmoid']

# 'sigmoid', 'tanh', 'relu'
add_activation_list = ['tanh']

# 'sigmoid', 'tanh', 'relu'
erase_activation_list = ['sigmoid']

# 'add_off_erase_off', 'add_off_erase_on', 'add_on_erase_off', 'add_on_erase_on'
write_type_list = ['add_on_erase_on']

learning_rate_list = [0.6]
# 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

counter_loss_weight_list = [0.0001]
#0.0, 0.001, 0.01, 0.1, 1.0

counter_embedding_dim_list = [64]

#################### DQN ####################
state_type_list = ['mastery']
#'value', 'mastery'

reward_type_list = ['mastery'] 
#'value', 'read', 'summary', 'prob', 'mastery'

policy_type_list = ['dqn']
#'dqn', 'random', 'prob_max'


repeat_num = 1 


for repeat_idx in range(repeat_num): 

    ##### DKVMN #####
    for knowledge_growth in knowledge_growth_list:
        for add_activation in add_activation_list:
            for erase_activation in erase_activation_list:
                for write_type in write_type_list:
                    for summary_activation in summary_activation_list:
                        for learning_rate in learning_rate_list:
                            for counter_loss_weight in counter_loss_weight_list:
                                for counter_embedding_dim in counter_embedding_dim_list:

                                    ##### DQN #####
                                    for state_type in state_type_list:
                                        for reward_type in reward_type_list:
                                            for policy_type in policy_type_list:

                                                args_list = []
                                                args_list.append('python main.py')

                                                args_list.append('--prefix squre_')

                                                args_list.append('--repeat_idx')
                                                args_list.append(str(repeat_idx))

                                                args_list.append('--dkvmn_train f --dkvmn_test f --dkvmn_ideal_test f')
                                                args_list.append('--dqn_train f --dqn_test f')
                                                args_list.append('--terminal_condition_type posneg_mastery') 

                                                args_list.append('--using_counter_graph t')
                                                args_list.append('--counter_embedding_dim')                    
                                                args_list.append(str(counter_embedding_dim))
                     
                                                args_list.append('--dkvmn_analysis t')
                                                args_list.append('--dkvmn_clustering_actions f')

                                                args_list.append('--counter_loss_weight')
                                                args_list.append(str(counter_loss_weight))

                                                args_list.append('--gpu_id 0')
                                                args_list.append('--logging_level DEBUG')

                                                args_list.append('--test_policy_type')
                                                args_list.append(policy_type)

                                                args_list.append('--dataset assist2009_updated')
                                                #args_list.append('--dataset synthetic')

                                                args_list.append('--knowledge_growth')
                                                args_list.append(knowledge_growth)

                                                args_list.append('--summary_activation')
                                                args_list.append(summary_activation)

                                                args_list.append('--add_activation')
                                                args_list.append(add_activation)

                                                args_list.append('--erase_activation')
                                                args_list.append(erase_activation)

                                                args_list.append('--write_type')
                                                args_list.append(write_type)

                                                args_list.append('--initial_lr')
                                                args_list.append(str(learning_rate))

                                                args_list.append('--state_type')
                                                args_list.append(state_type)

                                                args_list.append('--reward_type')
                                                args_list.append(reward_type)

                                                model = ' '.join(args_list)
                                                print(model)
                                                os.system(model)
