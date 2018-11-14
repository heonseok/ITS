import os 

# 'origin', 'value_matrix', 'read_content', 'summary', 'pred_prob', 'mastery'
knowledge_growth_list = ['origin']

# 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
learning_rate_list = [0.6]

# 0.0001, 0.001, 0.01, 0.1
niLoss_weight_list = [0.0]

repeat_start = 0 
repeat_end = 10 

dataset_list = [
    'synthetic',
    # 'assist2009_updated',
    # 'assist2015',
    # 'STATICS',
]

policy_list = [
    'random', 
    #'dqn', 
    'prob_max', 
    #'prob_min'   
]

dkvmn_train_flag = 'f' 
dkvmn_test_flag = 'f'
analysis_flag = 't'

policy_train_flag = 'f'
policy_test_flag = 't'

for dataset in dataset_list:
    for learning_rate in learning_rate_list:
        for knowledge_growth in knowledge_growth_list:
            for niLoss_weight in niLoss_weight_list:
                for repeat_idx in range(repeat_start, repeat_end):
                    for policy_type in policy_list:

                        args_list = []
                        args_list.append('python main.py')

                        args_list.append('--prefix hwang_')

                        args_list.append('--dkvmn_train ' + dkvmn_train_flag)
                        args_list.append('--dkvmn_test ' + dkvmn_test_flag)
                        args_list.append('--dkvmn_analysis ' + analysis_flag)

                        args_list.append('--policy_test_flag ' + policy_test_flag)
                        args_list.append('--test_policy_type' + policy_type)

                        args_list.append('--dataset ' + dataset)
                        args_list.append('--knowledge_growth ' + knowledge_growth)
                        args_list.append('--negative_influence_loss_weight ' + str(niLoss_weight))
                        args_list.append('--repeat_idx ' + str(repeat_idx))

                        args_list.append('--initial_lr ' + str(learning_rate)) 

                        args_list.append('--gpu_id 0')
                        args_list.append('--logging_level DEBUG')

                        model = ' '.join(args_list)
                        print(model)
                        os.system(model)
