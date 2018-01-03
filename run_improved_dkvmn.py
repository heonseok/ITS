import os 

# 'origin', 'value_matrix', 'read_content', 'summary', 'pred_prob'
knowledge_growth_list = ['read_content']

# 'sigmoid', 'tanh', 'relu'
add_signal_activation_list = ['sigmoid']

# 'sigmoid', 'tanh', 'relu'
erase_signal_activation_list = ['sigmoid']

# 'sigmoid', 'tanh', 'relu'
summary_activation_list = ['sigmoid']

# 'add_off_erase_off', 'add_off_erase_on', 'add_on_erase_off', 'add_on_erase_on'
write_type_list = ['add_off_erase_on']

learning_rate_list = [0.6]
#learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

for knowledge_growth in knowledge_growth_list:
    for add_signal_activation in add_signal_activation_list:
        for erase_signal_activation in erase_signal_activation_list:
            for write_type in write_type_list:
                for summary_activation in summary_activation_list:
                    for learning_rate in learning_rate_list:

                        args_list = []
                        args_list.append('python main.py --dkvmn_train f --dkvmn_test f --dkvmn_ideal_test f --dqn_train t --dqn_test t --gpu_id 0 --reward_type prob --dkvmn_checkpoint_dir DKVMN/100epoch_checkpoint')

                        args_list.append('--dataset assist2009_updated')

                        args_list.append('--knowledge_growth')
                        args_list.append(knowledge_growth)

                        args_list.append('--summary_activation')
                        args_list.append(summary_activation)

                        args_list.append('--add_signal_activation')
                        args_list.append(add_signal_activation)

                        args_list.append('--erase_signal_activation')
                        args_list.append(erase_signal_activation)

                        args_list.append('--write_type')
                        args_list.append(write_type)

                        args_list.append('--initial_lr')
                        args_list.append(str(learning_rate))

                        model = ' '.join(args_list)
                        print(model)
                        os.system(model)
