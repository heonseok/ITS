import gym
import tensorflow as tf
import numpy as np
import random
import copy

class DKVMNEnvironment():
    def __init__(self, args, sess, dkvmn, logger):
        self.args = args

        self.logger = logger 
        self.sess = sess
        self.logger.info('Initializing ENVIRONMENT')
        self.env = dkvmn 

        self.num_actions = self.args.n_questions
        self.initial_ckpt = np.copy(self.env.memory.memory_value)
        self.episode_step = 0

        self.value_matrix = self.get_init_value_matrix()  
        mastery_level = self.get_mastery_level()
        
        if self.args.state_type == 'value':
            self.state_shape = list(self.value_matrix.shape)
            self.state = self.value_matrix
        elif self.args.state_type == 'mastery':
            self.state_shape = [self.args.memory_size] 
            self.state = mastery_level 

        self.answer_checker = np.zeros(self.num_actions)
        self.action_count = 0

    def get_init_value_matrix(self):
        return self.sess.run(self.env.init_memory_value)

    def get_prediction_probability(self):
        return np.squeeze(self.sess.run(self.env.total_pred_probs, feed_dict={self.env.total_value_matrix: self.value_matrix}))

    def get_mastery_level(self):
        return np.squeeze(self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix}))

    def env_status(self):
        probs = self.get_prediction_probability()
        prev_mastery_level = self.get_mastery_level() 

        answer = np.expand_dims(np.expand_dims(1, axis=0), axis=0)

        self.logger.debug('Prob valdiff readdiff sumdiff probdiff masterydiff')
        for action_index in range(self.num_actions):
            action = np.asarray(action_index+1, dtype=np.int32)
            action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

            ops = [self.env.stepped_value_matrix, self.env.value_matrix_difference, self.env.read_content_difference, self.env.summary_difference, self.env.pred_prob_difference]
            val_matrix, val_diff, read_diff, summary_diff, prob_diff = self.sess.run(ops, feed_dict={self.env.q: action, self.env.a: answer, self.env.value_matrix: self.value_matrix})
            mastery_level = self.get_mastery_level()
            mastery_diff = np.sum(mastery_level[0]-prev_mastery_level[0])

            self.logger.debug('{: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}'.format(probs[action_index], val_diff, read_diff, summary_diff, prob_diff, mastery_diff)) 
            

    def new_episode(self):
        correct = np.sum(self.answer_checker)
        final_mastery_level = self.get_mastery_level()

        final_values_probs = self.get_prediction_probability()
        final_value_matrix = self.value_matrix

        self.logger.debug('Final env status')
        self.env_status()

        ##### init variables #####
        self.value_matrix = self.get_init_value_matrix()
        self.answer_checker = np.zeros(self.num_actions)

        starting_values_probs = self.get_prediction_probability()
        starting_mastery_level = self.get_mastery_level()
 
        self.logger.debug('Starting env status')
        self.env_status()

        ###### count pos/neg #####
        mastery_level_diff = final_mastery_level - starting_mastery_level
        mastery_pos = np.sum(mastery_level_diff >= 0.)
        mastery_neg = np.sum(mastery_level_diff < 0.)

        prob_diff = final_values_probs - starting_values_probs
        prob_pos = np.sum(prob_diff >= 0.)
        prob_neg = np.sum(prob_diff < 0.)

        mastery_count_log = 'p: %d, n: %d' % (mastery_pos, mastery_neg)
        prob_count_log = 'p: %d, n: %d' % (prob_pos, prob_neg)

        ###### caculate average #####
        starting_mastery_level_avg = np.average(starting_mastery_level)
        final_mastery_level_avg = np.average(final_mastery_level)
        mastery_log = 'Mastery f: %.4f, d: %.4f' % (final_mastery_level_avg, final_mastery_level_avg - starting_mastery_level_avg)

        starting_prob_avg = np.average(starting_values_probs)
        final_prob_avg = np.average(final_values_probs)
        prob_log = 'Prob f: %.4f, d: %.4f' % (final_prob_avg, final_prob_avg - starting_prob_avg)

        ##### logging #####
        self.logger.info('NEW EPISODE Corret: %d %s %s %s %s Action count: %d' % (correct, mastery_log, mastery_count_log, prob_log, prob_count_log, self.action_count))

        self.action_count = 0
        
    def check_terminal(self, total_pred_probs):
        total_preds = total_pred_probs >= self.args.terminal_threshold
        mask = np.squeeze(total_preds) * self.answer_checker

        if np.prod(mask) == 1:
            return True
        else: 
            return False

    def baseline_action(self):
        total_preds = self.get_prediction_probability()
        total_preds = self.mask_actions(total_preds)

        if self.args.test_policy_type == 'prob_max':
            action = np.argmax(total_preds)
        elif self.args.test_policy_type == 'prob_min':
            action = np.argmin(total_preds)
        return action 


    def mask_actions(self, values):
        total_preds = self.get_prediction_probability()
        total_preds = total_preds >= self.args.terminal_threshold
        mask = np.squeeze(total_preds) * self.answer_checker

        return (1-mask) * values


    def act(self, action):

        self.action_count += 1

        action = np.asarray(action+1, dtype=np.int32)
        action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

        # -1 for sampling 
        # 0, 1 for input given
        # 0 : worst, 1 : best 
        answer = np.asarray(-1, dtype=np.int32)
        answer = np.expand_dims(np.expand_dims(answer, axis=0), axis=0)

        prev_mastery_level = self.get_mastery_level()

        ## update value matrix 
        ops = [self.env.stepped_value_matrix, self.env.value_matrix_difference, self.env.read_content_difference, self.env.summary_difference, self.env.qa, self.env.stepped_pred_prob, self.env.pred_prob_difference]
        self.value_matrix, val_diff, read_diff, summary_diff, qa, stepped_prob, prob_diff = self.sess.run(ops, feed_dict={self.env.q: action, self.env.a: answer, self.env.value_matrix: self.value_matrix})

        mastery_level = self.get_mastery_level()
 
        if qa > self.num_actions:
            a = qa - self.num_actions
            self.answer_checker[a-1] = 1 
        else:
            a = qa 
            self.answer_checker[a-1] = 0 


        if self.args.reward_type == 'value':
            self.reward = np.sum(val_diff) 
        elif self.args.reward_type == 'read':
            self.reward = np.sum(read_diff)
        elif self.args.reward_type == 'summary':
            self.reward = np.sum(summary_diff)
        elif self.args.reward_type == 'prob':
            self.reward = np.sum(prob_diff)
        elif self.args.reward_type == 'mastery':
            self.reward = np.sum(mastery_level[0]-prev_mastery_level[0])


        if self.args.state_type == 'value':
            self.state = self.value_matrix
        elif self.args.state_type == 'mastery':
            self.state = mastery_level


        self.episode_step += 1

        total_pred_probs = self.get_prediction_probability()
        self.logger.debug('QA : %3d, Reward : %+5.4f, Prob : %1.4f, ProbDiff : %+1.4f' % (qa, self.reward, stepped_prob, prob_diff))

        if self.episode_step == self.args.episode_maxstep:
            terminal = True
        elif self.check_terminal(total_pred_probs) == True:
            terminal = True
        else:
            terminal = False

        return np.squeeze(self.state), self.reward, terminal, mastery_level

    def random_action(self):
        return random.randrange(0, self.num_actions)
