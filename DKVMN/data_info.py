import numpy as np
import os 
import tensorflow as tf


class DATA_Analyzer():
    def __init__(self, path, n_questions, seperate_char):
        self.seperate_char = seperate_char
        self.n_questions = n_questions
        self.path = path+'.csv'

    def analysis_dataset(self):
        f_data = open(self.path, 'r')

        total_access_arr = np.zeros(self.n_questions, dtype=np.int)
        sequence_len_list = list()

        net_correct_rate_list = list()
        correct_rate_list = list()

        for lineid, line in enumerate(f_data):
            # strip
            line = line.strip()
            if line[-1] == self.seperate_char:
                line = line[:-1] 

            # Problem 
            if lineid % 3 == 1:
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
                
                for idx, answer in enumerate(answer_list):
                    answer = eval(answer)
                    if answer == 1:
                        q_idx = eval(q_tag_list[idx])
                        net_correct_arr[q_idx-1] = 1 
                        correct_counter += 1

                    elif answer == 0:
                        answer
                
                correct_rate = correct_counter/len(q_tag_list)
                correct_rate_list.append(correct_rate)
                
                net_correct_rate = np.sum(net_correct_arr)/len(set(q_tag_list))
                net_correct_rate_list.append(net_correct_rate)
                    
        
        print('Average sequence len')
        print(np.average(sequence_len_list))

        print('Average correct rate')
        print(np.average(correct_rate_list))
        print(correct_rate_list)

        print('Average net correct rate')
        print(np.average(net_correct_rate_list))
        print(net_correct_rate_list)

        print('Total access array')
        for idx in range(self.n_questions):
            print('{} : {}'.format(idx+1, total_access_arr[idx]))

        print(np.sum(total_access_arr))
            

if __name__ == "__main__":
    seperate_char = ','
    n_questions = 110
    #path = 'data/assist2009_updated/assist2009_updated_train_total'
    path = 'data/assist2009_updated/assist2009_updated_train_toy'
    #path = 'data/assist2009_updated/assist2009_updated_test'
    da = DATA_Analyzer(path, n_questions, seperate_char)
    da.analysis_dataset()


