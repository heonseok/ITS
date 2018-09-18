import numpy as np
import math
import sys
import os, time
import tensorflow as tf
import operations
import shutil
from memory import DKVMN

import dkvmn_utils

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#import _model
#import _model_merged
#import _model_merged_multi
import _model_refactored

class DKVMNModel(_model_refactored.Mixin):
    def __init__(self, args, name='KT'):

        self.args = args
        self.name = name

        # tf.set_random_seed(224)
        self.logger = dkvmn_utils.set_logger('DKVMN', self.args.prefix + 'dkvmn.log', self.args.logging_level)
        self.model_name = 'DKVMN'

        # tf.reset_default_graph()

        '''
        self.build_model()

        if self.args.dkvmn_train != True:
            self.load()
        #if self.args.dkvmn_test == True:
            #self.load()

        '''
        '''
        elif self.args.dkvmn_train == False and self.args.dkvmn_test == False:
            #self.logger.debug('Non training & testing')
            self.args.batch_size = 1
            self.args.seq_len = 1
            #self.build_step_graph()
            self.load()
        '''

        '''
        graph = self.build_dkvmn_graph()
        with tf.Session(config = run_confing, graph=graph) as sess:
            self.set_session(sess)
        '''
             
    def set_session(self, session):
        self.sess = session

    def build_dkvmn_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            self.using_counter_graph = tf.placeholder(tf.bool)
            self.using_concept_counter_graph = tf.placeholder(tf.bool)
            self.build_model()
            # if self.args.dkvmn_train != True:
                # self.build_step_graph()

        return graph

    def build_step_dkvmn_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            self.using_counter_graph = tf.placeholder(tf.bool)
            self.using_concept_counter_graph = tf.placeholder(tf.bool)
            self.build_model()
            self.args.batch_size = 1
            self.args.batch_size = 1
            self.build_step_graph()

        return graph

    def get_init_counter(self):
        return np.zeros([1, self.args.n_questions+1])

    def get_init_concept_counter(self):
        return np.zeros([1, self.args.memory_size])

    def get_init_value_matrix(self):
        return self.sess.run(self.init_memory_value)

    def get_prediction_probability(self, value_matrix, counter, concept_counter):
        feed_dict = {self.value_matrix: value_matrix, self.counter: counter, self.concept_counter: concept_counter,
                     self.using_counter_graph: self.args.using_counter_graph,
                     self.using_concept_counter_graph: self.args.using_concept_counter_graph }
        return np.squeeze(self.sess.run(self.total_pred_probs, feed_dict=feed_dict))

    def get_mastery_level(self, value_matrix, counter, concept_counter):
        feed_dict = {self.value_matrix: value_matrix, self.counter: counter, self.concept_counter: concept_counter,
                     self.using_counter_graph: self.args.using_counter_graph,
                     self.using_concept_counter_graph: self.args.using_concept_counter_graph }
        return np.squeeze(self.sess.run(self.concept_mastery, feed_dict=feed_dict))

    def update_value_matrix(self, value_matrix, action, answer, counter, concept_counter):
        ops = [self.stepped_value_matrix]
        feed_dict = {self.q: action, self.a: answer, self.value_matrix: value_matrix,
                     self.counter: counter, self.concept_counter: concept_counter,
                     self.using_counter_graph: self.args.using_counter_graph,
                     self.using_concept_counter_graph: self.args.using_concept_counter_graph }
        value_matrix = self.sess.run(ops, feed_dict=feed_dict)
        return np.squeeze(value_matrix)

    def expand_dims(self, val):
        return np.expand_dims(np.expand_dims(val, axis=0), axis=0)

    def increase_concept_counter(self, concept_counter, q):
        # q = self.expand_dims(q)
        q_embed, cor_weight = self.sess.run([self.q_embed, self.cor_weight], feed_dict={self.q: q})

        return concept_counter + cor_weight

    def train(self, train_q_data, train_qa_data, valid_q_data, valid_qa_data, early_stop=False,
              checkpoint_dir='', selected_mastery_index=-1):

        # np.random.seed(224)
        # q_data, qa_data : [samples, seq_len]
        self.logger.info('#'*120)
        self.logger.info(self.model_dir)
        self.logger.info('Train')

        training_step = train_q_data.shape[0] // self.args.batch_size
        self.sess.run(tf.global_variables_initializer())
        '''
        value_mem = self.init_memory_value.eval()
        print(np.sum(value_mem))

        for i in self.tr_vrbs:
            print(i.name)
            print(i.shape)
            print(np.sum(i.eval()))
        '''
        
        if self.args.show:
            from dkvmn_utils import ProgressBar
            bar = ProgressBar(label, max=training_step)

        self.train_count = 0
        # if self.args.init_from:
        '''
        if self.load():
            self.logger.debug('Checkpoint exists')
            #print('Checkpoint exists and skip training')
            #return 
        else:
            self.logger.debug('No checkpoint')
        '''
        '''
        else:
            if os.path.exists(os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)):
                try:
                    shutil.rmtree(os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir))
                    shutil.rmtree(os.path.join(self.args.dkvmn_log_dir, self.model_dir+'.csv'))
                except(FileNotFoundError, IOError) as e:
                    print('[Delete Error] %s - %s' % (e.filename, e.strerror))
        '''
        
        best_valid_auc = 0
        early_stop_counter = 0

        # Training
        for epoch in range(0, self.args.num_epochs):
            shuffle_index = np.random.permutation(train_q_data.shape[0])
            q_data_shuffled = train_q_data[shuffle_index, :]
            qa_data_shuffled = train_qa_data[shuffle_index, :]

            if self.args.show:
                bar.next()

            pred_list = list()
            target_list = list()        
            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_ni_loss = 0
            epoch_co_loss = 0
            learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, decay_steps=self.args.anneal_interval*training_step, decay_rate=0.667, staircase=True)
            lr = learning_rate.eval()
            for steps in range(training_step):
                # [batch size, seq_len]
                q_batch_seq = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                qa_batch_seq = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
    
                # qa : exercise index + answer(0 or 1)*exercies_number
                # right : 1, wrong : 0, padding : -1
                target = qa_batch_seq[:,:]
                # Make integer type to calculate target
                target = target.astype(np.int)
                target_batch = (target - 1) // self.args.n_questions  
                target_batch = target_batch.astype(np.float)

                op = [self.loss, self.pred, self.train_op,
                      self.cross_entropy_loss, self.negative_influence_loss, self.convergence_loss,
                      # self.prob_debug, self.prb_diff_debug, self.negative_index_debug, self.prob_diff_negative_debug, self.sq_debug, self.mean_debug, self.cul_mean_debug
                      ]

                feed_dict = {self.q_data_seq:q_batch_seq, self.qa_data_seq:qa_batch_seq, self.target_seq:target_batch,
                             self.lr:lr, self.selected_mastery_index:selected_mastery_index,
                             self.using_counter_graph:self.args.using_counter_graph,
                             self.using_concept_counter_graph:self.args.using_concept_counter_graph}

                # loss_, pred_, _, ce_loss, ni_loss, co_loss, \ prob_debug, prob_diff_debug, negative_index_debug, prob_diff_negative_debug, sq_debug, mean_debug, cul_mean_debug = self.sess.run(op, feed_dict=feed_dict)
                loss_, pred_, _, ce_loss, ni_loss, co_loss = self.sess.run(op, feed_dict=feed_dict)

                # print(prob_list.shape)
                '''
                print(prob_debug)
                print(prob_diff_debug)
                print(negative_index_debug)
                print(prob_diff_negative_debug)
                print(sq_debug)
                print(mean_debug)
                print('NI LOSS')
                print(cul_mean_debug)
                if math.isnan(cul_mean_debug):
                   sys.exit()
                '''


                pred_ = pred_[:,selected_mastery_index+1:] 
                target_batch = target_batch[:,selected_mastery_index+1:]

                # Get right answer index
                # Make [batch size * seq_len, 1]
                right_target = np.asarray(target_batch).reshape(-1,1)
                right_pred = np.asarray(pred_).reshape(-1,1)
                # np.flatnonzero returns indices which is nonzero, convert it list 
                right_index = np.flatnonzero(right_target != -1.).tolist()
                # print(len(right_index)/self.args.batch_size)
                # Number of 'training_step' elements list with [batch size * seq_len, ]
                pred_list.append(right_pred[right_index])
                target_list.append(right_target[right_index])

                epoch_loss += loss_
                epoch_ce_loss += ce_loss
                epoch_ni_loss += ni_loss
                epoch_co_loss += co_loss
                # print('Epoch %d/%d, steps %d/%d, loss : %3.5f' % (epoch+1, self.args.num_epochs, steps+1, training_step, loss_))
                

            if self.args.show:
                bar.finish()        
            
            all_pred = np.concatenate(pred_list, axis=0)
            all_target = np.concatenate(target_list, axis=0)


            train_auc, train_accuracy = dkvmn_utils.calculate_metric(all_target, all_pred)
            epoch_loss = epoch_loss / training_step
            epoch_ce_loss = epoch_ce_loss / training_step
            epoch_ni_loss = epoch_ni_loss / training_step
            epoch_co_loss = epoch_co_loss / training_step
            self.logger.debug('Epoch %d/%d, loss : %3.5f, auc : %3.5f, acc : %3.5f ce: %3.5f, ni: %3.5f, co: %3.5f'
                              % (epoch+1, self.args.num_epochs, epoch_loss, train_auc, train_accuracy, epoch_ce_loss, epoch_ni_loss, epoch_co_loss))
            # self.logger.debug('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (epoch+1, self.args.num_epochs, epoch_loss, train_auc, train_accuracy))
            # self.write_log(epoch=epoch+1, auc=train_auc, accuracy=train_accuracy, loss=epoch_loss, name='training_')

            valid_steps = valid_q_data.shape[0] // self.args.batch_size
            valid_pred_list = list()
            valid_target_list = list()
            for s in range(valid_steps):
                # Validation
                valid_q = valid_q_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                valid_qa = valid_qa_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                # right : 1, wrong : 0, padding : -1
                valid_target = (valid_qa - 1) // self.args.n_questions
                valid_target = valid_target.astype(np.float)
                valid_feed_dict = {self.q_data_seq : valid_q, self.qa_data_seq : valid_qa, self.target_seq : valid_target,
                                   self.selected_mastery_index: selected_mastery_index,
                                   self.using_counter_graph: self.args.using_counter_graph,
                                   self.using_concept_counter_graph: self.args.using_concept_counter_graph }
                valid_loss, valid_pred = self.sess.run([self.loss, self.pred], feed_dict=valid_feed_dict)
                # Same with training set

                valid_pred = valid_pred[:,selected_mastery_index+1:] 
                valid_target = valid_target[:,selected_mastery_index+1:]

                valid_right_target = np.asarray(valid_target).reshape(-1,)
                valid_right_pred = np.asarray(valid_pred).reshape(-1,)
                valid_right_index = np.flatnonzero(valid_right_target != -1).tolist()    
                valid_target_list.append(valid_right_target[valid_right_index])
                valid_pred_list.append(valid_right_pred[valid_right_index])
            
            all_valid_pred = np.concatenate(valid_pred_list, axis=0)
            all_valid_target = np.concatenate(valid_target_list, axis=0)

            valid_auc, valid_accuracy = dkvmn_utils.calculate_metric(all_valid_target, all_valid_pred)
            self.logger.debug('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' %(epoch+1, self.args.num_epochs, valid_auc, valid_accuracy))
            # Valid log
            # self.write_log(epoch=epoch+1, auc=valid_auc, accuracy=valid_accuracy, loss=valid_loss, name='valid_')
            if valid_auc > best_valid_auc:
                self.logger.debug('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
                best_valid_auc = valid_auc
                best_epoch = epoch + 1
                self.save(best_epoch, checkpoint_dir)
                early_stop_counter = 0
            else:
                early_stop_counter += 1 

            if early_stop and early_stop_counter > self.args.early_stop_th:
                self.logger.debug('Eearly stop')
                return best_epoch

        self.logger.info('Number of non-update epochs: {}'.format(early_stop_counter))
        return best_epoch    
    
    def test(self, test_q, test_qa, checkpoint_dir='', selected_mastery_index=-1):
        self.logger.info('#'*120)
        self.logger.info(self.model_dir)
        self.logger.info('Test')

        steps = test_q.shape[0] // self.args.batch_size
        # self.sess.run(tf.global_variables_initializer())

        '''
        if self.load(checkpoint_dir):
            self.logger.debug('CKPT Loaded')
        else:
            self.logger.debug('CKPT need')
            raise Exception()
        '''


        pred_list = list()
        target_list = list()
        q_list = list()

        for s in range(steps):
            # self.writer.add_summary(s)

            test_q_batch = test_q[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            test_qa_batch = test_qa[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            target = test_qa_batch[:,:]
            target = target.astype(np.int)
            target_batch = (target - 1) // self.args.n_questions  
                            
            target_batch = target_batch.astype(np.float)

            # meta graph version
            # feed_dict = {"q_data_seq:0":test_q_batch, "qa_data:0":test_qa_batch, "target:0":target_batch, "selected_mastery_index:0":selected_mastery_index, self.using_counter_graph:self.args.using_counter_graph}

            feed_dict = {self.q_data_seq: test_q_batch, self.qa_data_seq: test_qa_batch, self.target_seq: target_batch,
                         self.selected_mastery_index: selected_mastery_index,
                         self.using_counter_graph: self.args.using_counter_graph,
                         self.using_concept_counter_graph: self.args.using_concept_counter_graph
                         }
            loss_, pred_ = self.sess.run([self.loss, self.pred], feed_dict=feed_dict)
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
        # all_q = np.concatenate(q_list, axis=0)

        test_auc, test_accuracy = dkvmn_utils.calculate_metric(all_target, all_pred)
        # count, metric_for_each_q = dkvmn_dkvmn_utils.calculate_metric_for_each_q(all_target, all_pred, all_q, self.args.n_questions)
        # print(metric_for_each_q)
        '''
        for (idx, metric) in enumerate(metric_for_each_q):
            self.logger.info('{:<3d}: {:>7d}, {: .4f}, {: .4f}'.format(idx+1, count[idx], metric[0], metric[1]))
        '''

        self.logger.info('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))
        # self.write_log(epoch=1, auc=test_auc, accuracy=test_accuracy, loss=0, name='test_')

        return pred_list_2d, target_list_2d, test_auc, test_accuracy
        
    def clustering_actions(self):
        '''
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')
        '''

        value_matrix = self.get_init_value_matrix()
        # value_matrix = self.sess.run(self.init_memory_value)

        total_cor_weight = self.sess.run(self.total_correlation_weight)
    
        kmeans = KMeans(n_clusters=self.args.memory_size).fit(total_cor_weight)
        # print('kmeans centers')
        # print(kmeans.cluster_centers_)
        cluster_labels = kmeans.predict(total_cor_weight)

        vis_weight = TSNE().fit_transform(total_cor_weight)
        vis_x = vis_weight[:,0]
        vis_y = vis_weight[:,1]

        plt.scatter(vis_x, vis_y, c=cluster_labels)
        plt.colorbar(ticks=range(self.args.n_questions))

        ids = ['{}'.format(i) for i in range(self.args.n_questions)] 

        #'''
        for _id, label, x, y in zip(ids, cluster_labels, vis_x, vis_y):
            plt.annotate('{}'.format(_id), xy = (x,y))
            # plt.annotate('{},{}'.format(label, _id), xy = (x,y))
        #'''

        plt.show()

    def ideal_test(self):
        type_list = [-1]
        for t in type_list: 
            self.ideal_test_given_type(t) 
    
    def ideal_test_given_type(self, input_type):
        
        '''
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')
        '''

        log_file_name = 'logs/'+self.model_dir
        if input_type == 0:
            log_file_name = log_file_name + '_neg.csv'
        elif input_type == 1:
            log_file_name = log_file_name + '_pos.csv' 
        elif input_type == -1:
            log_file_name = log_file_name + '_rand.csv' 

        log_file = open(log_file_name, 'w')
        value_matrix = self.sess.run(self.init_memory_value)
        for i in range(20):

            for q_idx in range(1, self.args.n_questions+1):
                q = np.expand_dims(np.expand_dims(q_idx, axis=0), axis=0) 
                a = np.expand_dims(np.expand_dims(input_type, axis=0), axis=0) 
        
                ops = [self.stepped_value_matrix, self.stepped_pred_prob, self.value_matrix_difference, self.read_content_difference, self.summary_difference, self.pred_logit_difference, self.pred_prob_difference, self.qa]
                feed_dict = { self.q : q, self.a : a, self.value_matrix: value_matrix }

                value_matrix, pred_prob, value_matrix_diff, read_content_diff, summary_diff, pred_logit_diff, pred_prob_diff, qa = np.squeeze(self.sess.run(ops, feed_dict=feed_dict))
                a = (qa-1) // self.args.n_questions
                pred_prob = np.squeeze(np.squeeze(pred_prob))

                log = str(i)+','+ str(q_idx) +','+str(a)+','+str(np.sum(value_matrix))+','+str(pred_prob) + ','
                log = log + str(value_matrix_diff) + ','  + str(read_content_diff) + ',' + str(summary_diff) + ',' + str(pred_logit_diff) + ',' + str(pred_prob_diff) + '\n'  
                log_file.write(log) 

        log_file.flush()    

    @property
    def model_dir(self):
        '''
        network_spec = 'Knowledge_{}_Summary_{}_Add_{}_Erase_{}_WriteType_{}_Counter_{}_coLoss_{}' \
            .format(self.args.knowledge_growth, self.args.summary_activation, self.args.add_activation, self.args.erase_activation, self.args.write_type, self.args.convergence_loss_weight, self.args.using_counter_graph)
        '''

        # hyper_parameters = '_lr{}_{}epochs'.format(self.args.initial_lr, self.args.num_epochs)
        # data_postfix = '_{}_{}_{}'.format(self.args.train_postfix, self.args.valid_postfix, self.args.test_postfix)
        # remove_short = '_RemoveShort_{}_th_{}'.format(self.args.remove_short_seq, self.args.short_seq_len_th)

        '''
        if self.args.dataset == 'assist2009_updated':
            dataset_spec = ''
        else:
            dataset_spec = '_{}'.format(self.args.dataset)
        '''
        '''
        if self.args.negative_influence_loss_weight == 0.0:
            ni_weight_spec = ''
        else:
            ni_weight_spec = '_niLoss_{}'.format(self.args.negative_influence_loss_weight)
        '''

        network_spec = 'Knowledge_{}'.format(self.args.knowledge_growth)

        # counter_spec = '_Counter_{}_cDim_{}'.format(self.args.using_counter_graph, self.args.counter_embedding_dim)
        # if self.args.using_concept_counter_graph == False:
        #     concept_counter_spec = ''
        # else:
        #     concept_counter_spec = '_ConCounter_{}'.format(self.args.using_concept_counter_graph)

        ni_weight_spec = '_niLoss_{}'.format(self.args.negative_influence_loss_weight)
        # co_weight_spec = '_coLoss_{}'.format(self.args.convergence_loss_weight)

        repeat_spec = '_rIdx_{}'.format(self.args.repeat_idx)

        model_spec = 'knowledge_{}_niLoss_{}_rIdx_{}'.format(self.args.knowledge_growth, self.args.negative_influence_loss_weight, self.args.repeat_idx)
        return model_spec
        # return network_spec + counter_spec + concept_counter_spec + ni_weight_spec + co_weight_spec + repeat_spec
        # return network_spec + network_detail + counter_detail + ni_weight_detail + dataset_detail + repeat_detail

    def load(self, checkpoint_dir=''):
        self.logger.debug('Loading ckpt')
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        self.logger.debug(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.logger.debug(ckpt_name)
            self.train_count = int(ckpt_name.split('-')[-1])

            '''
            ## import meta graph codes
            self.new_saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, ckpt_name+'.meta'))
            self.logger.debug('DKVMN graph loaded')
            self.new_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            '''

            # '''
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # '''

            self.logger.debug('DKVMN ckpt loaded')
            return True
        else:
            self.logger.debug('DKVMN cktp not loaded')
            return False

    '''
    @property
    def ckpt_dir(self, checkpoint_dir=''):
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        return checkpoint_dir
    '''

    def save(self, global_step, checkpoint_dir=''):
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=global_step)
        self.logger.debug('Save checkpoint at %d' % (global_step+1))

    # Log file
    '''
    def write_log(self, auc, accuracy, loss, epoch, name='training_'):
        log_path = os.path.join(self.args.dkvmn_log_dir, name+self.model_dir+'.csv')
        if not os.path.exists(log_path):
            self.log_file = open(log_path, 'w')
            self.log_file.write('Epoch\tAuc\tAccuracy\tloss\n')
        else:
            self.log_file = open(log_path, 'a')    
        
        self.log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(loss) + '\n')
        self.log_file.flush()    
    '''
