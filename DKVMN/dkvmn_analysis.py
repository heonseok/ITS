import numpy as np
import os 

from utils import *
from model import *
from scenario import *

class DKVMNAnalyzer():
    def __init__(self, args, dkvmn):

        self.args = args
        self.num_actions = self.args.n_questions

        self.dkvmn = dkvmn

        self.init_value_matrix = self.dkvmn.get_init_value_matrix()
        self.init_counter = self.dkvmn.get_init_counter()
        self.init_concept_counter = self.dkvmn.get_init_concept_counter()
        self.init_probs = self.dkvmn.get_prediction_probability(self.init_value_matrix, self.init_counter, self.init_concept_counter)
        self.init_probs_avg = np.average(self.init_probs)

        self.init_state()

        ## hyper parameters 
        self.max_cycle_num = 1

        ## Set loggers
        self.path = os.path.join(self.args.prefix, self.dkvmn.model_dir)



    def init_state(self):
        self.value_matrix = self.init_value_matrix


    def update_state(self, action_idx, answer_type):
        return self.calc_influence(action_idx, answer_type, self.value_matrix, self.init_counter, self.init_concept_counter, update_value_matrix_flag=True)


    def get_probability(self, value_matrix):
        return self.dkvmn.get_prediction_probability(value_matrix, self.init_counter, self.init_concept_counter)

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

        ########### calculate probability decrease statistic
        prob_pos_update = probs_diff[probs_diff >= 0]
        prob_neg_update = probs_diff[probs_diff <0] 

        target_prev_prob = prev_probs[action_idx] 
        target_prob_diff = probs_diff[action_idx]
        target_prob      = probs[action_idx]

        stat_summary_total = self.get_stats(probs_diff)
        stat_summary_pos = self.get_stats(prob_pos_update) 
        stat_summary_neg = self.get_stats(prob_neg_update)

        # print(",".join([str(action_idx), str(target_prev_prob), str(target_prob), str(target_prob_diff), stat_summary_total, stat_summary_pos, stat_summary_neg]))

        return value_matrix, counter, concept_counter, probs, mastery_level, wrong_response_count_prob, wrong_response_count_mastery

    def get_stats(self, arr):
        if len(arr) == 0: 
            return "0,-1,-1,-1,-1"

        arr_stat = list()
        arr_stat.append(len(arr))
        arr_stat.append(np.min(arr))
        arr_stat.append(np.max(arr))
        arr_stat.append(np.average(arr))
        arr_stat.append(np.std(arr))  

        '''
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_avg = np.average(arr)
        arr_std = np.std(arr)  
        '''
        stat_summary = ['{:.4f}'.format(x) for x in arr_stat]
        return ",".join(stat_summary)
         
    def analysis(self, test_q, test_qa, scenario='best', ordering='permutation'):
        # self.test_response()

        # self.testset_analysis(test_q, test_qa)

        print('analysis')
        # if scenario == 'best':
        # self.scenario = BestScenario(ordering, self.num_actions)
        # self.scenario_analysis()

        # elif scenario == 'random':
        # self.scenario = RandomScenario(self.num_actions, self.dkvmn)

        self.scenario = MaxProbScenario(self.num_actions, self.dkvmn)
        self.scenario_analysis()


    def scenario_analysis(self):

        print('Scenario Analysis')
        self.init_state()
        log_path = os.path.join(self.path, self.scenario.get_name())

        self.logger = set_logger('aDKVMN', log_path, 'summary.log', 'INFO')

        self.logger_avg_prob = set_logger('aDKVMN_avg_prob', log_path, 'avg_prob.log', self.args.logging_level, display=False)
        self.logger_avg_prob.info('#'*120)
        f_avg_prob = open(os.path.join('log', log_path, 'avg_prob.result'), 'w')

        self.logger_dist = set_logger('aDKVMN_dist', log_path, 'dist.log', self.args.logging_level, display=False)
        self.logger_dist.info('#'*120)

        self.logger_path = set_logger('aDKVMN_path', log_path, 'path.log', self.args.logging_level, display=False)
        self.logger_path.info('#'*120)

        # probs_avg_list = list()
        # probs_avg_list.append(self.init_probs_avg)


        self.logger_dist.info(float2str_list(self.init_probs))
        for cycle_idx in range(self.max_cycle_num):

            for step_idx in range(self.num_actions):
                action_idx, answer = self.scenario.get_action(self.value_matrix)

                prev_prob = self.get_probability(self.value_matrix)[action_idx]

                self.value_matrix, _, _, probs, _, wrong_response_count_prob, wrong_response_count_mastery  = self.update_state(action_idx, answer)
                # probs_avg_list.append(np.average(probs))
                self.logger_avg_prob.info('{:.4f}'.format(np.average(probs)))
                f_avg_prob.write('{:.4f}\n'.format(np.average(probs)))

                self.logger_path.info('{:3d}, {: .4f}, {: .4f}'.format(action_idx+1, prev_prob, probs[action_idx]))

                if action_idx == self.num_actions-1:
                    self.logger_dist.info(float2str_list(probs))

        # self.logger_avg_prob.info(float2str_list(probs_avg_list))


    def testset_analysis_with_run(self, test_q, test_qa, checkpoint_dir='', selected_mastery_index=-1):
        print('Testset Analysis')

        log_path = self.path
        self.logger = set_logger('aDKVMN', log_path, 'test.log', 'INFO')
        self.logger.info('#'*120)
        self.logger.info(self.dkvmn.model_dir)
        self.logger.info('Test')


        batch_size = 32
        steps = test_q.shape[0] // batch_size
        # self.sess.run(tf.global_variables_initializer())

        pred_list = list()
        target_list = list()
        q_list = list()

        self.logger.info('Number of steps : {}'.format(steps))
        for s in range(steps):

            test_q_batch = test_q[s*batch_size:(s+1)*batch_size, :]
            test_qa_batch = test_qa[s*batch_size:(s+1)*batch_size, :]
            target = test_qa_batch[:,:]
            target = target.astype(np.int)
            target_batch = (target - 1) // self.args.n_questions

            target_batch = target_batch.astype(np.float)

            # meta graph version
            # feed_dict = {"q_data_seq:0":test_q_batch, "qa_data:0":test_qa_batch, "target:0":target_batch, "selected_mastery_index:0":selected_mastery_index, self.using_counter_graph:self.args.using_counter_graph}

            feed_dict = {self.dkvmn.q_data_seq: test_q_batch, self.dkvmn.qa_data_seq: test_qa_batch, self.dkvmn.target_seq: target_batch,
                         self.dkvmn.selected_mastery_index: selected_mastery_index,
                         self.dkvmn.using_counter_graph: self.args.using_counter_graph,
                         self.dkvmn.using_concept_counter_graph: self.args.using_concept_counter_graph
                         }
            loss_, pred_ = self.dkvmn.sess.run([self.dkvmn.loss, self.dkvmn.pred], feed_dict=feed_dict)
            # Get right answer index
            # Make [batch size * seq_len, 1]

            pred_ = pred_[:,selected_mastery_index+1:]
            target_batch = target_batch[:,selected_mastery_index+1:]
            q_ = test_q_batch[:,selected_mastery_index+1:]

            if s == 0:
                pred_list_2d = pred_
                target_list_2d = target_batch
            else:
                pred_list_2d = np.concatenate((pred_list_2d, pred_), axis=0)
                target_list_2d = np.concatenate((target_list_2d, target_batch), axis=0)

            target_reshaped = np.asarray(target_batch).reshape(-1,1)
            pred_reshaped = np.asarray(pred_).reshape(-1,1)
            q_reshaped = np.asarray(q_).reshape(-1,1)
            # np.flatnonzero returns indices which is nonzero, convert it list
            right_index = np.flatnonzero(target_reshaped != -1.).tolist()
            # Number of 'training_step' elements list with [batch size * seq_len, ]

            right_pred = pred_reshaped[right_index]
            right_target = target_reshaped[right_index]
            right_q = q_reshaped[right_index]

            pred_list.append(right_pred)
            target_list.append(right_target)
            q_list.append(right_q)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        all_q = np.concatenate(q_list, axis=0)


        self.logger.info('<Before filtering> Number of all target: {}, pred: {}, q: {}'.format(len(all_target), len(all_pred), len(all_q)))
        test_auc, test_accuracy = calculate_auc_acc(all_target, all_pred)
        self.logger.info('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))
        count, metric_for_each_q = calculate_metric_for_each_q(all_target, all_pred, all_q, self.args.n_questions)
        for (idx, metric) in enumerate(metric_for_each_q):
            self.logger.info('{:<3d}: {:>7d}, {: .4f}, {: .4f}'.format(idx+1, count[idx], metric[0], metric[1]))

        th = 10
        all_target, all_pred, all_q = self.filter_result_with_threshold(all_target, all_pred, all_q, th)
        self.logger.info('<After filtering with th {}> Number of all target: {}, pred: {}, q: {}'.format(th, len(all_target), len(all_pred), len(all_q)))
        test_auc, test_accuracy = calculate_auc_acc(all_target, all_pred)
        self.logger.info('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))
        count, metric_for_each_q = calculate_metric_for_each_q(all_target, all_pred, all_q, self.args.n_questions)
        for (idx, metric) in enumerate(metric_for_each_q):
            self.logger.info('{:<3d}: {:>7d}, {: .4f}, {: .4f}'.format(idx+1, count[idx], metric[0], metric[1]))


        # count, metric_for_each_q = dkvmn_utils.calculate_auc_acc_for_each_q(all_target, all_pred, all_q, self.args.n_questions)
        # for (idx, metric) in enumerate(metric_for_each_q):
        #     self.logger.info('{:<3d}: {:>7d}, {: .4f}, {: .4f}'.format(idx+1, count[idx], metric[0], metric[1]))
        #
        # self.logger.info('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))

        return pred_list_2d, target_list_2d, test_auc, test_accuracy


    def filter_result_with_threshold(self, target, pred, q, th=0):
        unique_item = np.unique(q)

        for item in unique_item:
            if np.sum(q == item) < th:
                del_idx = np.where(q == item)
                target = np.delete(target, del_idx)
                pred = np.delete(pred, del_idx)
                q = np.delete(q, del_idx)

        return target, pred, q

    # def test(self):
    #     # self.logger.info('#'*120)
    #     # self.logger.info(self.dkvmn.model_dir)
    #
    #     # self.logger.info('Prbs Avg Test')
    #     self.test_probs_avg_base(1,True)
    #
    #     # self.test_negative_influence(True)
    #
    #     '''
    #     self.test_converge_bound(True)
    #
    #     self.test_converge_speed(1, True)
    #
    #     self.test_latent_learning(1)
    #
    #     self.test_latent_learning2(1)
    #     '''
    #
    # def test_negative_influence(self, update_value_matrix_flag):
    #     self.logger.info('Negative Influence Test')
    #     return self.test_negative_influence_base(1, update_value_matrix_flag)
    #     # self.test_negative_influence(0, update_value_matrix_flag)
    #
    # def test_probs_avg_base(self, answer_type, update_value_matrix_flag):
    #     init_value_matrix = self.dkvmn.get_init_value_matrix()
    #     init_counter = self.dkvmn.get_init_counter()
    #     init_concept_counter = self.dkvmn.get_init_concept_counter()
    #     init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)
    #     init_probs_avg = np.average(init_probs)
    #
    #
    #     value_matrix = init_value_matrix
    #     probs_avg_list = list()
    #     probs_avg_list.append(init_probs_avg)
    #     for action_idx in range(self.num_actions):
    #         value_matrix, _, _, probs, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, value_matrix, init_counter, init_concept_counter, update_value_matrix_flag)
    #
    #         probs_avg_list.append(np.average(probs))
    #
    #     probs_avg_list = ['{:.4f}'.format(x) for x in probs_avg_list]
    #     self.logger.info(self.dkvmn.model_dir+",ascending," + ",".join(probs_avg_list))
    #
    #     value_matrix = init_value_matrix
    #     probs_avg_list = list()
    #     probs_avg_list.append(init_probs_avg)
    #     for action_idx in range(self.num_actions-1,-1,-1):
    #         value_matrix, _, _, probs, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, value_matrix, init_counter, init_concept_counter, update_value_matrix_flag)
    #
    #         probs_avg_list.append(np.average(probs))
    #
    #     probs_avg_list = ['{:.4f}'.format(x) for x in probs_avg_list]
    #     self.logger.info(self.dkvmn.model_dir+",Descending," + ",".join(probs_avg_list))
    #
    #     # rand_idx = [15, 24, 34, 26, 44, 37, 38, 21, 13, 31, 48, 28, 20, 29, 32, 4, 14, 42, 27, 47, 11, 9, 49, 16, 23, 6, 8, 17, 0, 18, 43, 19, 25, 12, 39, 41, 40, 7, 3, 45, 33, 10, 36, 46, 2, 35, 5, 1, 22, 30]
    #     rand_idx = np.random.permutation(self.num_actions)
    #     value_matrix = init_value_matrix
    #     probs_avg_list = list()
    #     probs_avg_list.append(init_probs_avg)
    #     for action_idx in rand_idx:
    #         value_matrix, _, _, probs, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, value_matrix, init_counter, init_concept_counter, update_value_matrix_flag)
    #
    #         probs_avg_list.append(np.average(probs))
    #
    #     probs_avg_list = ['{:.4f}'.format(x) for x in probs_avg_list]
    #     self.logger.info(self.dkvmn.model_dir+",Permu," + ",".join(probs_avg_list))
    #
    #
    # def test_negative_influence_base(self, answer_type, update_value_matrix_flag):
    #     init_value_matrix = self.dkvmn.get_init_value_matrix()
    #     init_counter = self.dkvmn.get_init_counter()
    #     init_concept_counter = self.dkvmn.get_init_concept_counter()
    #
    #     right_updated_skill_counter = 0
    #     wrong_updated_skill_list = []
    #
    #     right_updated_mastery_counter = 0
    #     wrong_updated_mastery_list = []
    #
    #     # NEW metric
    #     right_updated_skill_count_list = []
    #
    #
    #     '''
    #     # Scenario : answer all problem correctly
    #     value_matrix = init_value_matrix
    #     counter = init_counter
    #     concept_counter = init_concept_counter
    #     for idx in range(self.num_actions):
    #         value_matrix, counter, concept_counter, _, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(idx, answer_type, value_matrix, counter, concept_counter, 1)
    #     '''
    #
    #     value_matrix = init_value_matrix
    #     probs_avg_list = list()
    #     for action_idx in range(self.num_actions):
    #         value_matrix, _, _, probs, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, value_matrix, init_counter, init_concept_counter, update_value_matrix_flag)
    #         # _, _, _, _, _, wrong_response_count_prob, wrong_response_count_mastery  = self.calc_influence(action_idx, answer_type, init_value_matrix, init_counter, init_concept_counter, update_value_matrix_flag)
    #
    #         right_updated_skill_count_list.append(self.num_actions - wrong_response_count_prob)
    #
    #         probs_avg_list.append(np.average(probs))
    #
    #         if wrong_response_count_prob == 0:
    #             right_updated_skill_counter += 1
    #         elif wrong_response_count_prob > 0:
    #             wrong_updated_skill_list.append(action_idx+1)
    #
    #         if wrong_response_count_mastery == 0:
    #             right_updated_mastery_counter += 1
    #         elif wrong_response_count_mastery > 0:
    #             wrong_updated_mastery_list.append(action_idx+1)
    #
    #     probs_avg_list = ['{:.4f}'.format(x) for x in probs_avg_list]
    #     self.logger.info(",".join(probs_avg_list))
    #
    #     pi = np.average(right_updated_skill_count_list)/self.num_actions
    #     self.logger.info('Answer type: {}, CUC: {}, PI: {}'.format(answer_type, right_updated_skill_counter, pi))
    #     # self.logger.info('{}'.format(int2str_list(wrong_updated_skill_list)))
    #
    #     # self.logger.info('Mastery {}, {}'.format(answer_type, right_updated_mastery_counter))
    #     # self.logger.info('{}'.format(int2str_list(wrong_updated_mastery_list)))
    #
    #     return right_updated_skill_counter
    #
    # def test_converge_bound(self, update_value_matrix_flag):
    #     self.logger.info('Convergence Bound Test')
    #     self.test_converge_bound_base(1, update_value_matrix_flag)
    #     # self.test_convergence_bound_base(0, update_value_matrix_flag)
    #
    # def test_converge_bound_base(self, answer_type, update_value_matrix_flag):
    #     repeat_num = 10
    #
    #     init_counter = self.dkvmn.get_init_counter()
    #     init_value_matrix = self.dkvmn.get_init_value_matrix()
    #
    #     init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter)
    #
    #     prob_mat = np.zeros((self.num_actions, repeat_num+1))
    #
    #     for action_idx in range(self.num_actions):
    #         value_matrix = np.copy(init_value_matrix)
    #         counter = np.copy(init_counter)
    #
    #         prob_mat[action_idx][0] = init_probs[action_idx]
    #         for repeat_idx in range(repeat_num):
    #
    #             value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, update_value_matrix_flag)
    #             prob_mat[action_idx][repeat_idx+1] = probs[action_idx]
    #
    #     increase_count, decrease_count = calc_seq_trend(prob_mat)
    #     rmse = calc_rmse_with_ones(prob_mat)
    #     diff = calc_diff_with_ones(prob_mat)
    #     self.logger.info('{}, {:>3}, {:>3}'.format(answer_type, increase_count, decrease_count))
    #     self.logger.info(float2str_list(diff))
    #     self.logger.info(float2str_list(rmse))
    #
    # def test_converge_speed(self, answer_type, update_value_matrix_flag):
    #     # Adaptive knowledge growth test
    #
    #     self.logger.info('Converge Speed Test')
    #     target_q = 25
    #
    #     init_counter = self.dkvmn.get_init_counter()
    #     init_concept_counter = self.dkvmn.get_init_concept_counter()
    #     init_value_matrix = self.dkvmn.get_init_value_matrix()
    #
    #     prob_diff_th = 0.01
    #     repeat_lim = 100
    #
    #     repeat_count = np.zeros(self.num_actions)
    #     target_q_prob_list = []
    #
    #     for action_idx in range(self.num_actions):
    #         value_matrix = np.copy(init_value_matrix)
    #         counter = np.copy(init_counter)
    #         concept_counter = np.copy(init_concept_counter)
    #
    #         converge_count = 0
    #         repeat_idx = 0
    #
    #         prev_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)
    #         if action_idx == 0:
    #             target_q_prob_list.append(prev_probs[target_q])
    #         while True:
    #             repeat_idx += 1
    #             value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, concept_counter, update_value_matrix_flag)
    #             if action_idx == target_q:
    #                 target_q_prob_list.append(probs[action_idx])
    #             prob_diff = probs[action_idx] - prev_probs[action_idx]
    #             prev_probs = probs
    #
    #             if np.abs(prob_diff) < prob_diff_th:
    #                 converge_count += 1
    #             else:
    #                 converge_count = 0
    #
    #             if converge_count == 5 or repeat_idx > repeat_lim:
    #                 repeat_count[action_idx] = repeat_idx
    #                 # self.logger.info('{}, {}'.format(action_idx+1, repeat_idx))
    #                 break
    #
    #     self.logger.info('Average : {}'.format(np.average(repeat_count)))
    #     return target_q_prob_list, np.average(repeat_count)
    #
    # def test_latent_learning(self, answer_type):
    #     # test for latent learning by repeating the same exercises
    #     self.logger.info('Latent Information Test')
    #
    #     list0 = self.test_latent_learning_base(answer_type, 0)
    #     list1 = self.test_latent_learning_base(answer_type, 10)
    #     list2 = self.test_latent_learning_base(answer_type, 20)
    #     list3 = self.test_latent_learning_base(answer_type, 30)
    #     list4 = self.test_latent_learning_base(answer_type, 40)
    #     list5 = self.test_latent_learning_base(answer_type, 50)
    #
    #     return [list0, list1, list2, list3, list4, list5]
    #
    # def test_latent_learning_base(self, answer_type, no_response_period):
    #     repeat_num = 100
    #     init_counter = self.dkvmn.get_init_counter()
    #     init_concept_counter = self.dkvmn.get_init_concept_counter()
    #     init_value_matrix = self.dkvmn.get_init_value_matrix()
    #     init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)
    #
    #     prob_mat = np.zeros((self.num_actions, repeat_num+1))
    #
    #     for action_idx in range(self.num_actions):
    #         value_matrix = np.copy(init_value_matrix)
    #         counter = np.copy(init_counter)
    #         concept_counter = np.copy(init_concept_counter)
    #
    #         prob_mat[action_idx][0] = init_probs[action_idx]
    #         for non_response_idx in range(no_response_period):
    #             counter[0][action_idx] += 1
    #             action = self.dkvmn.expand_dims(action_idx+1)
    #             concept_counter = self.dkvmn.increase_concept_counter(concept_counter, action)
    #
    #         for repeat_idx in range(repeat_num):
    #             update_value_matrix_flag = True
    #
    #             value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, concept_counter, update_value_matrix_flag)
    #             prob_mat[action_idx][repeat_idx+1] = probs[action_idx]
    #
    #         avg = np.average(prob_mat, axis=0)
    #     self.logger.info(float2str_list(avg))
    #
    #     return float2str_list(avg)
    #
    # def test_latent_learning2(self, answer_type):
    #     # test for latent learning by working on different exercises
    #
    #     self.logger.info('Latent Learning Test2')
    #     rand_action_list = np.random.randint(self.num_actions, size=10)
    #
    #     self.test_latent_learning2_base(answer_type, rand_action_list, 0)
    #     self.test_latent_learning2_base(answer_type, rand_action_list, 100)
    #     self.test_latent_learning2_base(answer_type, rand_action_list, 200)
    #
    # def test_latent_learning2_base(self, answer_type, rand_action_list, no_response_period):
    #     repeat_num = 30
    #     init_counter = self.dkvmn.get_init_counter()
    #     init_concept_counter = self.dkvmn.get_init_concept_counter()
    #     init_value_matrix = self.dkvmn.get_init_value_matrix()
    #     init_probs = self.dkvmn.get_prediction_probability(init_value_matrix, init_counter, init_concept_counter)
    #
    #     value_matrix = np.copy(init_value_matrix)
    #     counter = np.copy(init_counter)
    #     concept_counter = np.copy(init_concept_counter)
    #
    #     prob_list = list()
    #     ex_prob_list = list()
    #     prob_list.append(np.average(init_probs))
    #
    #     for non_response_idx in range(no_response_period):
    #         rand_idx = np.squeeze(np.random.randint(self.num_actions, size=1))
    #         counter[0][rand_idx] += 1
    #         rand_action = self.dkvmn.expand_dims(rand_idx+1)
    #         concept_counter = self.dkvmn.increase_concept_counter(concept_counter, rand_action)
    #     # print(concept_counter)
    #
    #     for action_idx in rand_action_list:
    #         value_matrix, counter, concept_counter, probs, mastery, _, _ = self.calc_influence(action_idx, answer_type, value_matrix, counter, concept_counter, True)
    #         prob_list.append(np.average(probs))
    #         ex_prob_list.append(probs[action_idx])
    #
    #     self.logger.info('avg')
    #     self.logger.info(float2str_list(prob_list))
    #     self.logger.info('ex')
    #     self.logger.info(float2str_list(ex_prob_list))
