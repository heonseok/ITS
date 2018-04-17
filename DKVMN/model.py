import numpy as np
import os, time
import tensorflow as tf
import operations
import shutil
from memory import DKVMN
#rom sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score

import dkvmn_utils

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import _model

class DKVMNModel(_model.Mixin):
    def __init__(self, args, sess, name='KT'):

        self.args = args
        self.name = name
        self.sess = sess

        #tf.set_random_seed(224)
        self.logger = dkvmn_utils.set_logger('DKVMN', self.args.prefix + 'dkvmn.log', self.args.logging_level)


        #self.condition = tf.placeholder(tf.int32, [self.args.n_questions], name='condition') 
        
        self.build_model()
        self.build_total_prob_graph()
        self.build_mastery_graph()

    def get_init_counter(self):
        return np.zeros([1,self.args.n_questions+1])  

    def get_init_value_matrix(self):
        return self.sess.run(self.init_memory_value)

    def get_prediction_probability(self, value_matrix, counter):
        return np.squeeze(self.sess.run(self.total_pred_probs, feed_dict={self.total_value_matrix: value_matrix, self.total_counter: counter, self.total_using_counter_graph: self.args.using_counter_graph}))

    def get_mastery_level(self, value_matrix, counter):
        return np.squeeze(self.sess.run(self.concept_mastery_level, feed_dict={self.mastery_value_matrix: value_matrix, self.mastery_counter: counter, self.mastery_using_counter_graph: self.args.using_counter_graph}))
    
    def update_value_matrix(self, value_matrix, action, answer, counter):
       ops = [self.stepped_value_matrix]
       value_matrix = self.sess.run(ops, feed_dict={self.q: action, self.a: answer, self.value_matrix: value_matrix, self.step_counter: counter, self.step_using_counter_graph: self.args.using_counter_graph})
       return np.squeeze(value_matrix)

    def expand_dims(self, val):
        return np.expand_dims(np.expand_dims(val, axis=0), axis=0)


    def train(self, train_q_data, train_qa_data, valid_q_data, valid_qa_data, early_stop=False, checkpoint_dir='', selected_mastery_index=-1):
        #np.random.seed(224)
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
            from utils import ProgressBar
            bar = ProgressBar(label, max=training_step)

        self.train_count = 0
        #if self.args.init_from:
        if self.load():
            self.logger.debug('Checkpoint exists')
            #print('Checkpoint exists and skip training')
            #return 
        else:
            self.logger.debug('No checkpoint')
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

                feed_dict = {self.q_data_seq:q_batch_seq, self.qa_data_seq:qa_batch_seq, self.target_seq:target_batch, self.lr:lr, self.selected_mastery_index:selected_mastery_index, self.using_counter_graph:self.args.using_counter_graph}
                loss_, pred_, _, = self.sess.run([self.loss, self.pred, self.train_op], feed_dict=feed_dict)

                pred_ = pred_[:,selected_mastery_index+1:] 
                target_batch = target_batch[:,selected_mastery_index+1:]

                # Get right answer index
                # Make [batch size * seq_len, 1]
                right_target = np.asarray(target_batch).reshape(-1,1)
                right_pred = np.asarray(pred_).reshape(-1,1)
                # np.flatnonzero returns indices which is nonzero, convert it list 
                right_index = np.flatnonzero(right_target != -1.).tolist()
                #print(len(right_index)/self.args.batch_size)
                # Number of 'training_step' elements list with [batch size * seq_len, ]
                pred_list.append(right_pred[right_index])
                target_list.append(right_target[right_index])

                epoch_loss += loss_
                #print('Epoch %d/%d, steps %d/%d, loss : %3.5f' % (epoch+1, self.args.num_epochs, steps+1, training_step, loss_))
                

            if self.args.show:
                bar.finish()        
            
            all_pred = np.concatenate(pred_list, axis=0)
            all_target = np.concatenate(target_list, axis=0)


            train_auc, train_accuracy = dkvmn_utils.calculate_metric(all_target, all_pred)
            epoch_loss = epoch_loss / training_step    
            self.logger.debug('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (epoch+1, self.args.num_epochs, epoch_loss, train_auc, train_accuracy))
            #self.write_log(epoch=epoch+1, auc=train_auc, accuracy=train_accuracy, loss=epoch_loss, name='training_')

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
                valid_feed_dict = {self.q_data_seq : valid_q, self.qa_data_seq : valid_qa, self.target_seq : valid_target, self.selected_mastery_index:selected_mastery_index, self.using_counter_graph:self.args.using_counter_graph}
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
            #self.write_log(epoch=epoch+1, auc=valid_auc, accuracy=valid_accuracy, loss=valid_loss, name='valid_')
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
        self.sess.run(tf.global_variables_initializer())
        if self.load(checkpoint_dir):
            self.logger.debug('CKPT Loaded')
        else:
            self.logger.debug('CKPT need')
            raise Exception()


        pred_list = list()
        target_list = list()
        q_list = list()

        for s in range(steps):
            test_q_batch = test_q[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            test_qa_batch = test_qa[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            target = test_qa_batch[:,:]
            target = target.astype(np.int)
            target_batch = (target - 1) // self.args.n_questions  
            target_batch = target_batch.astype(np.float)
            feed_dict = {self.q_data_seq:test_q_batch, self.qa_data_seq:test_qa_batch, self.target_seq:target_batch, self.selected_mastery_index:selected_mastery_index, self.using_counter_graph:self.args.using_counter_graph}
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
        all_q = np.concatenate(q_list, axis=0)

        test_auc, test_accuracy = dkvmn_utils.calculate_metric(all_target, all_pred)
        #count, metric_for_each_q = dkvmn_utils.calculate_metric_for_each_q(all_target, all_pred, all_q, self.args.n_questions)
        #print(metric_for_each_q)
        '''
        for (idx, metric) in enumerate(metric_for_each_q):
            self.logger.info('{:<3d}: {:>7d}, {: .4f}, {: .4f}'.format(idx+1, count[idx], metric[0], metric[1]))
        '''

        self.logger.info('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))
        #self.write_log(epoch=1, auc=test_auc, accuracy=test_accuracy, loss=0, name='test_')

        return pred_list_2d, target_list_2d, test_auc, test_accuracy
        


    def clustering_actions(self):
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        value_matrix = self.get_init_value_matrix()
        #value_matrix = self.sess.run(self.init_memory_value)

        total_cor_weight = self.sess.run(self.total_correlation_weight, feed_dict={self.total_value_matrix : value_matrix})
    
        kmeans = KMeans(n_clusters=self.args.memory_size).fit(total_cor_weight)
        #print('kmeans centers')
        #print(kmeans.cluster_centers_)
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
            #plt.annotate('{},{}'.format(label, _id), xy = (x,y))
        #'''

        plt.show()
        


    def ideal_test(self):
        type_list = [-1]
        for t in type_list: 
            self.ideal_test_given_type(t) 
    

    def ideal_test_given_type(self, input_type): 
        
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

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
        network_spec = 'Knowledge_{}_Summary_{}_Add_{}_Erase_{}_WriteType_{}_cLossWeight_{}_Counter_{}'.format(self.args.knowledge_growth, self.args.summary_activation, self.args.add_activation, self.args.erase_activation, self.args.write_type, self.args.counter_loss_weight, self.args.using_counter_graph)
 
        if network_spec == 'Knowledge_origin_Summary_tanh_Add_tanh_Erase_sigmoid_WriteType_add_on_erase_on_cLossWeight_0.0_Counter_False':
            network_spec = 'Original'

        hyper_parameters = '_lr{}_{}epochs'.format(self.args.initial_lr, self.args.num_epochs)
        network_detail = '_MemSize{}'.format(self.args.memory_size)
        #hyper_parameters = '_lr{}_{}epochs_{}batch'.format(self.args.initial_lr, self.args.num_epochs, self.args.batch_size)
        #data_postfix = '_{}_{}_{}'.format(self.args.train_postfix, self.args.valid_postfix, self.args.test_postfix)
        #remove_short = '_RemoveShort_{}_th_{}'.format(self.args.remove_short_seq, self.args.short_seq_len_th)
        
        # TODO refactoring if/else statement

        if self.args.counter_embedding_dim == 20:
            counter_dtail = ''
        else:
            counter_detail = 'cDim_{}'.format(self.args.counter_embedding_dim)

        batch = '_batch_{}'.format(self.args.batch_size) 

        repeat_detail = '_rIdx_{}'.format(self.args.repeat_idx)
    
        if self.args.dataset == 'assist2009_updated':
            dataset_detail = ''
        else:
            dataset_detail = '_{}'.format(self.args.dataset)
        
        return network_spec + network_detail + counter_detail + dataset_detail + repeat_detail
        #return self.args.prefix + network_spec + network_detail + counter_detail + repeat_detail
        #return self.args.prefix + network_spec + network_detail + remove_short


    def load(self, checkpoint_dir=''):
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.logger.debug('DKVMN ckpt loaded')
            return True
        else:
            self.logger.debug('DKVMN cktp not loaded')
            return False

    def save(self, global_step, checkpoint_dir=''):
        model_name = 'DKVMN'
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
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
        
