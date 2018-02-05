import numpy as np
import os 

import matplotlib.pyplot as plt

import dkvmn_utils

class DATA_Analyzer():
    def __init__(self, path, n_questions, seperate_char, logger):
        self.seperate_char = seperate_char
        self.n_questions = n_questions
        self.path = path+'.csv'

        self.logger = logger

    def analysis_dataset(self):
        f_data = open(self.path, 'r')

        total_seq_num = 0
        total_access_arr = np.zeros(self.n_questions, dtype=np.int)
        sequence_len_list = list()

        net_correct_rate_list = list()
        correct_rate_list = list()

        answer_neg2pos_list = list()
        answer_pos2neg_list = list()

        for lineid, line in enumerate(f_data):
            # strip
            line = line.strip()
            if line[-1] == self.seperate_char:
                line = line[:-1] 

            # Problem 
            if lineid % 3 == 1:
                total_seq_num += 1
                q_tag_list = line.split(self.seperate_char)
                sequence_len_list.append(len(q_tag_list))

                for q in q_tag_list:
                    q = eval(q)
                    total_access_arr[q-1] += 1
            
            # Answer
            elif lineid % 3 == 2:
                answer_list = line.split(self.seperate_char)
                net_correct_arr = np.zeros(self.n_questions)
                correct_counter = 0
                answer_neg2pos = np.zeros(self.n_questions, dtype=np.int)
                answer_pos2neg = np.zeros(self.n_questions, dtype=np.int)
                
                for idx, answer in enumerate(answer_list):
                    answer = eval(answer)
                    q_idx = eval(q_tag_list[idx])

                    if answer == 1:
                        net_correct_arr[q_idx-1] = 1 
                        correct_counter += 1

                        answer_pos2neg[q_idx-1] = -1 

                        if answer_neg2pos[q_idx-1] == -1:
                            answer_neg2pos[q_idx-1] = 1 

                    elif answer == 0:
                        answer_neg2pos[q_idx-1] = -1

                        if answer_pos2neg[q_idx-1] == -1:
                            answer_pos2neg[q_idx-1] = 1
                
                correct_rate = correct_counter/len(q_tag_list)
                correct_rate_list.append(correct_rate)
                
                net_correct_rate = np.sum(net_correct_arr)/len(set(q_tag_list))
                net_correct_rate_list.append(net_correct_rate)
                
                answer_neg2pos_filtered = answer_neg2pos[answer_neg2pos == 1]
                answer_neg2pos_list.append(np.sum(answer_neg2pos_filtered))
                #answer_neg2pos_list.append(np.sum(answer_neg2pos))

                answer_pos2neg_filtered = answer_pos2neg[answer_pos2neg == 1]
                answer_pos2neg_list.append(np.sum(answer_pos2neg_filtered))
                #answer_pos2neg_list.append(np.sum(answer_pos2neg))

        #plt.hist(total_access_arr)
        #plt.show()
        #fig = plt.gcf()
        
        #print('Total access array')
        histogram = ''
        for idx in range(self.n_questions):
            histogram +=  '{} : {}\n'.format(idx+1, total_access_arr[idx])
            self.logger.debug('{} : {}'.format(idx+1, total_access_arr[idx]))

        total_access_count = str(np.sum(total_access_arr))
        #print('Total number of access : {}'.format(np.sum(total_access_arr)))


        avg_sequence_len = '{:.4f}'.format(np.average(sequence_len_list))
        #avg_sequence_len = np.average(sequence_len_list)
        #print('Average sequence len : {:.4f}'.format(np.average(sequence_len_list)))

        avg_correct_rate = '{:.4f}'.format(np.average(correct_rate_list))
        #avg_correct_rate = np.average(correct_rate_list)
        #print('Average correct rate : {:.4f}'.format(np.average(correct_rate_list)))
        #print(correct_rate_list)

        avg_net_correct_rate = '{:.4f}'.format(np.average(net_correct_rate_list))
        #avg_net_correct_rate = np.average(net_correct_rate_list)
        #print('Average net correct rate : {:.4f}'.format(np.average(net_correct_rate_list)))
        #print(net_correct_rate_list)

        avg_neg2pos_count = '{:.4f}'.format(np.average(answer_neg2pos_list))
        #avg_neg2pos_count = np.average(answer_neg2pos_list)
        #print('Average neg2pos count : {:.4f}'.format(np.average(answer_neg2pos_list)))
        #print(answer_neg2pos_list)

        avg_pos2neg_count = '{:.4f}'.format(np.average(answer_pos2neg_list))
        #avg_pos2neg_count = np.average(answer_pos2neg_list)
        #print('Average pos2neg count : {:.4f}'.format(np.average(answer_pos2neg_list)))
        #print(answer_pos2neg_list)
            
        return histogram, [str(total_seq_num), total_access_count, avg_sequence_len, avg_correct_rate, avg_net_correct_rate, avg_neg2pos_count, avg_pos2neg_count]

if __name__ == "__main__":
    seperate_char = ','

    logger = dkvmn_utils.set_logger('dataInfo', 'data_info.log', 'INFO')

    dir_name = 'data_information'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    path_prefix_list = ['data/assist2009_updated/assist2009_updated_']
    file_name_list = ['assist2009']
    n_questions_list = [110]

    #path_prefix_list = ['data/synthetic/naive_c5_q50_s4000_v1_']
    #file_name_list = ['synthetic']
    #n_questions_list = [50]

    target_file_list = ['data'] 
    #target_file_list = ['train1', 'train', 'train_total', 'test'] 
    #path = 'data/assist2009_updated/assist2009_updated_train_total'
    #path = 'data/assist2009_updated/assist2009_updated_train_toy'
    #path = 'data/assist2009_updated/assist2009_updated_test'

    #n_questions = 50 
    #path = 'data/synthetic/naive_c5_q50_s4000_v1_train1'

    header = 'target_file, total_seq_num, total_access_count, avg_sequence_len, avg_correct_rate, avg_net_correct_rate, avg_neg2pos_count, avg_pos2neg_count'

    for idx, path_prefix in enumerate(path_prefix_list):
        n_questions = n_questions_list[idx] 
        #file_name = file_name_list[idx]
        #file_path = os.path.join(dir_name, file_name)
        #log_file = open(file_path, 'w')

        #log_file.write(header+'\n')

        for target_file in target_file_list:
            path = path_prefix + target_file 

            da = DATA_Analyzer(path, n_questions, seperate_char, logger)
            histogram, info_list = da.analysis_dataset()
            info = target_file + ',' +  ','.join(info_list)
            logger.info(header)
            logger.info(info)
            logger.info(histogram)
            #log_file.write(info+'\n') 
            #log_file.write(histogram)
            #log_file.flush()
