import numpy as np

from utils import *
from model import *

class DKVMNAnalyzer():
    def __init__(self, args, dkvmn):

        self.args = args
        self.dkvmn = dkvmn
        self.num_actions = self.args.n_questions

        self.logger = set_logger('aDKVMN', self.args.prefix + 'dkvmn_analysis.log', self.args.logging_level)
        self.logger.debug('Initializing DKVMN Analyzer')

    def calc_influence(self, action_idx, answer_type, _value_matrix, _counter, _concept_counter, update_value_matrix_flag=True):

        value_matrix = np.copy(_value_matrix)
        counter = np.copy(_counter)
        concept_counter = np.copy(_concept_counter)
 
        answer = self.dkvmn.expand_dims(answer_type)

        prev_probs = self.dkvmn.get_prediction_probability(value_matrix, counter, concept_counter)
        prev_mastery_level = self.dkvmn.get_mastery_level(value_matrix, counter, concept_counter)
        action = self.dkvmn.expand_dims(action_idx+1)

        if update_value_matrix_flag == True:
            value_matrix = self.dkvmn.update_value_matrix(value_matrix, action, answer, counter, concept_counter)

        # update counters
        counter[0,action_idx + 1] += 1
        concept_counter = self.dkvmn.increase_concept_counter(concept_counter, action)

        probs = self.dkvmn.get_prediction_probability(value_matrix, counter, concept_counter)
        mastery_level = self.dkvmn.get_mastery_level(value_matrix, counter, concept_counter)

        probs_diff = probs - prev_probs
        mastery_diff = mastery_level - prev_mastery_level
    
        if answer_type == 1:
            wrong_response_count_prob = np.sum(probs_diff < 0)
            wrong_response_count_mastery = np.sum(mastery_diff < 0)
        elif answer_type == 0:
            wrong_response_count_prob = np.sum(probs_diff > 0)
            wrong_response_count_mastery = np.sum(mastery_diff > 0)

        return value_matrix, counter, concept_counter, probs, mastery_level, wrong_response_count_prob, wrong_response_count_mastery

    def test(self):
        self.logger.info('#'*120)
        self.logger.info(self.dkvmn.model_dir)

        self.test_negative_influence(True)

        self.test_converge_bound(True)

        self.test_converge_speed(1, True)

        self.test_latent_learning(1)

        self.test_latent_learning2(1)
        
    def test_negative_influence(self, update_value_matrix_flag):
        self.logger.info('Negative Influence Test')
        return self.test_negative_influence_base(1, update_value_matrix_flag)
        # self.test_negative_influence(0, update_value_matrix_flag)

    def test_negative_influence_base(self, answer_type, update_value_matrix_flag):
        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_counter = self.dkvmn.get_init_counter()
        init_concept_counter = self.dkvmn.get_init_concept_counter()

        right_updated_skill_counter = 0
        wrong_updated_skill_list = []

        right_updated_mastery_counter = 0
        wrong_updated_mastery_list = []

        for action_idx in range(self.num_actions):
            _, _, _, _, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, init_value_matrix, init_counter, init_concept_counter, update_value_matrix_flag)

            if wrong_response_count_prob == 0:
                right_updated_skill_counter += 1
            elif wrong_response_count_prob > 0:
                wrong_updated_skill_list.append(action_idx+1)

            if wrong_response_count_mastery == 0:
                right_updated_mastery_counter += 1
            elif wrong_response_count_mastery > 0:
                wrong_updated_mastery_list.append(action_idx+1)

        self.logger.info('Probs   {}, {}'.format(answer_type, right_updated_skill_counter))
        self.logger.info('{}'.format(int2str_list(wrong_updated_skill_list)))

        # self.logger.info('Mastery {}, {}'.format(answer_type, right_updated_mastery_counter))
        # self.logger.info('{}'.format(int2str_list(wrong_updated_mastery_list)))

        return right_updated_skill_counter

    def test_converge_bound(self, update_value_matrix_flag):
        self.logger.info('Convergence Bound Test')
        self.test_converge_bound_base(1, update_value_matrix_flag)
        # self.test_convergence_bound_base(0, update_value_matrix_flag)

    def test_converge_bound_base(self, answer_type, update_value_matrix_flag):
        repeat_num = 10

        init_counter = self.dkvmn.get_init_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()

        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter)
   
        prob_mat = np.zeros((self.num_actions, repeat_num+1))

        for action_idx in range(self.num_actions):
            value_matrix = np.copy(init_value_matrix)
            counter = np.copy(init_counter)

            prob_mat[action_idx][0] = init_probs[action_idx]
            for repeat_idx in range(repeat_num):

                value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, update_value_matrix_flag)
                prob_mat[action_idx][repeat_idx+1] = probs[action_idx]

        increase_count, decrease_count = calc_seq_trend(prob_mat)
        rmse = calc_rmse_with_ones(prob_mat)
        diff = calc_diff_with_ones(prob_mat)
        self.logger.info('{}, {:>3}, {:>3}'.format(answer_type, increase_count, decrease_count))
        self.logger.info(float2str_list(diff))
        self.logger.info(float2str_list(rmse))

    def test_converge_speed(self, answer_type, update_value_matrix_flag):
        # Adaptive knowledge growth test

        self.logger.info('Converge Speed Test')
        target_q = 0

        init_counter = self.dkvmn.get_init_counter()
        init_concept_counter = self.dkvmn.get_init_concept_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()

        prob_diff_th = 0.001
        repeat_lim = 100

        repeat_count = np.zeros(self.num_actions)
        target_q_prob_list = []

        for action_idx in range(self.num_actions):
            value_matrix = np.copy(init_value_matrix)
            counter = np.copy(init_counter)
            concept_counter = np.copy(init_concept_counter)

            converge_count = 0
            repeat_idx = 0

            prev_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)
            if action_idx == 0:
                target_q_prob_list.append(prev_probs[target_q])
            while True:
                repeat_idx += 1
                value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, concept_counter, update_value_matrix_flag)
                if action_idx == target_q:
                    target_q_prob_list.append(probs[action_idx])
                prob_diff = probs[action_idx] - prev_probs[action_idx]
                prev_probs = probs

                if np.abs(prob_diff) < prob_diff_th:
                    converge_count += 1
                else:
                    converge_count = 0

                if converge_count == 5 or repeat_idx > repeat_lim:
                    repeat_count[action_idx] = repeat_idx
                    # self.logger.info('{}, {}'.format(action_idx+1, repeat_idx))
                    break

        self.logger.info('Average : {}'.format(np.average(repeat_count)))
        return target_q_prob_list, np.average(repeat_count)

    def test_latent_learning(self, answer_type):
        # test for latent learning by repeating the same exercises
        self.logger.info('Latent Information Test')

        list1 = self.test_latent_learning_base(answer_type, 0)
        list2 = self.test_latent_learning_base(answer_type, 10)
        list3 = self.test_latent_learning_base(answer_type, 20)

        return [list1, list2, list3]

    def test_latent_learning_base(self, answer_type, no_response_period):
        repeat_num = 30
        init_counter = self.dkvmn.get_init_counter()
        init_concept_counter = self.dkvmn.get_init_concept_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)

        prob_mat = np.zeros((self.num_actions, repeat_num+1))
        for action_idx in range(self.num_actions):
            value_matrix = np.copy(init_value_matrix)
            counter = np.copy(init_counter)
            concept_counter = np.copy(init_concept_counter)

            prob_mat[action_idx][0] = init_probs[action_idx]
            for repeat_idx in range(repeat_num):
                if repeat_idx < no_response_period:
                    update_value_matrix_flag = False
                else:
                    update_value_matrix_flag = True

                value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, concept_counter, update_value_matrix_flag)
                prob_mat[action_idx][repeat_idx+1] = probs[action_idx]

            avg = np.average(prob_mat, axis=0)
        self.logger.info(float2str_list(avg))

        return float2str_list(avg)

    def test_latent_learning2(self, answer_type):
        # test for latent learning by working on different exercises

        self.logger.info('Latent Learning Test2')
        rand_action_list = np.random.randint(self.num_actions, size=10)

        self.test_latent_learning2_base(answer_type, rand_action_list, 0)
        self.test_latent_learning2_base(answer_type, rand_action_list, 100)
        self.test_latent_learning2_base(answer_type, rand_action_list, 200)

    def test_latent_learning2_base(self, answer_type, rand_action_list, no_response_period):
        repeat_num = 30
        init_counter = self.dkvmn.get_init_counter()
        init_concept_counter = self.dkvmn.get_init_concept_counter()
        init_value_matrix = self.dkvmn.get_init_value_matrix()
        init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)

        value_matrix = np.copy(init_value_matrix)
        counter = np.copy(init_counter)
        concept_counter = np.copy(init_concept_counter)

        prob_list = list()
        ex_prob_list = list()
        prob_list.append(np.average(init_probs))

        for non_response_idx in range(no_response_period):
            rand_idx = np.squeeze(np.random.randint(self.num_actions, size=1))
            counter[0][rand_idx] += 1
            rand_action = self.dkvmn.expand_dims(rand_idx+1)
            concept_counter = self.dkvmn.increase_concept_counter(concept_counter, rand_action)
        # print(concept_counter)

        for action_idx in rand_action_list:
            value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, concept_counter, True)
            prob_list.append(np.average(probs))
            ex_prob_list.append(probs[action_idx])

        self.logger.info('avg')
        self.logger.info(float2str_list(prob_list))
        self.logger.info('ex')
        self.logger.info(float2str_list(ex_prob_list))
