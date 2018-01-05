import gym
import tensorflow as tf
import numpy as np
import random
import copy

class Environment(object):
    def __init__(self, args):
        self.args = args

        
class SimpleEnvironment(Environment):
    def __init__(self, args):
        super(SimpleEnvironment, self).__init__(args)
        self.env = gym.make(self.args.env_name)
        self.num_actions = self.env.action_space.n
        self.state_shape = list(self.env.observation_space.shape)

    def new_episode(self):
        return self.env.reset()

    def act(self, action):
        self.state, self.reward, self.terminal, _ = self.env.step(action)

        return self.state, self.reward, self.terminal

    def random_action(self):
        return self.env.action_space.sample()

class DKVMNEnvironment(Environment):
    def __init__(self, args, sess, dkvmn):
        super(DKVMNEnvironment, self).__init__(args)

        self.sess = sess
        print('Initializing ENVIRONMENT')
        self.env = dkvmn 

        self.num_actions = self.args.n_questions
        self.initial_ckpt = np.copy(self.env.memory.memory_value)
        self.episode_step = 0

        self.value_matrix = self.sess.run(self.env.init_memory_value)  
        mastery_level = self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix})[0]
        

        #print(list(self.value_matrix.shape))
        if self.args.state_type == 'value':
            self.state_shape = list(self.value_matrix.shape)
            self.state = self.value_matrix
        elif self.args.state_type == 'mastery':
            self.state_shape = [self.args.memory_size] 
            self.state = mastery_level 

        print('state shape')
        print(self.state_shape)

    def new_episode(self):
        print('NEW EPISODE')
        
        final_mastery_level = self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix})[0]

        final_values_probs = self.sess.run(self.env.total_pred_probs, feed_dict={self.env.total_value_matrix: self.value_matrix})
        final_value_matrix = self.value_matrix

        self.value_matrix = self.sess.run(self.env.init_memory_value)
        starting_values_probs = self.sess.run(self.env.total_pred_probs, feed_dict={self.env.total_value_matrix: self.value_matrix})
        starting_mastery_level = self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix})[0]
 
        print('final_mastery_level')
        for i, (s,f) in enumerate(zip(starting_mastery_level, final_mastery_level)):
            print(i, s, f, f-s)

        val_matrix_diff = np.sum(final_value_matrix - self.value_matrix) 
        prob_diff = np.sum(final_values_probs - starting_values_probs)
        pos_count = 0
        neg_count = 0
        print('final_values_probs')
        for i, (s,f) in enumerate(zip(starting_values_probs, final_values_probs)):
            print(i, s, f, f-s)
            if f>=s:
                pos_count += 1
            else:
                neg_count +=1


        log_file_name = 'episode_log.txt'
        log_file = open(log_file_name, 'a')
        log = 'Pos %d, neg %d, prob_diff %f, val_diff %f \n' % (pos_count, neg_count, prob_diff, val_matrix_diff)  
        log_file.write(log)
        log_file.flush()

    def check_terminal(self, total_pred_probs):
        return False

    def baseline_act(self):
        total_preds = self.sess.run(self.env.total_pred_probs, feed_dict={self.env.total_value_matrix: self.value_matrix})
        best_prob_index = np.argmax(total_preds)
        print('Action:%d, prob:%3.4f' % (best_prob_index+1, total_preds[best_prob_index]))
        return best_prob_index+1, total_preds[best_prob_index]

    def act(self, action):
        action = np.asarray(action+1, dtype=np.int32)
        action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

        # -1 for sampling 
        # 0, 1 for input given
        # 0 : worst, 1 : best 
        answer = np.asarray(-1, dtype=np.int32)
        answer = np.expand_dims(np.expand_dims(answer, axis=0), axis=0)

        prev_mastery_level = self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix})

        ops = [self.env.stepped_value_matrix, self.env.value_matrix_difference, self.env.read_content_difference, self.env.summary_difference, self.env.qa, self.env.stepped_pred_prob, self.env.pred_prob_difference]
        self.value_matrix, val_diff, read_diff, summary_diff, qa, stepped_prob, prob_diff = self.sess.run(ops, feed_dict={self.env.q: action, self.env.a: answer, self.env.value_matrix: self.value_matrix})
        mastery_level = self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix})
        #print(mastery_level)
        #print(mastery_level[0])
        #print(mastery_level[0][0])

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

        ######## calculate probabilty for total problems
        #total_preds = self.sess.run(self.env.total_pred_probs, feed_dict={self.env.total_value_matrix: self.value_matrix})
        #print(total_preds)

        self.episode_step += 1

        # For scaling
        #self.reward = self.reward/100
        
        ########## Concept mastery level ##############
        #print(mastery_level)
        '''
        for i in range(0,20):
            mastery_level = self.sess.run([self.env.concept_mastery_level], feed_dict={self.env.mastery_value_matrix: self.value_matrix, self.env.target_concept_index: i})
            print(i)
            print(mastery_level[0])
            print('Concept %d mastery level: %d' %(i,mastery_level[0]))
        '''


        total_pred_probs = self.sess.run(self.env.total_pred_probs, feed_dict={self.env.total_value_matrix: self.value_matrix})
        print('QA : %3d, Reward : %+5.4f, Prob : %1.4f, ProbDiff : %+1.4f' % (qa, self.reward, stepped_prob, prob_diff))
        if self.episode_step == self.args.episode_maxstep:
            terminal = True
        elif self.check_terminal(total_pred_probs) == True:
            terminal = True
        else:
            terminal = False

        return np.squeeze(self.state), self.reward, terminal

    def random_action(self):
        return random.randrange(0, self.num_actions)
