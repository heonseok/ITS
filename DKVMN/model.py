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


class DKVMNModel():
    def __init__(self, args, sess, name='KT'):

        self.args = args
        self.name = name
        self.sess = sess

        #tf.set_random_seed(224)
        #self.logger = self.set_logger()
        self.logger = dkvmn_utils.set_logger('DKVMN', 'dkvmn.log')
        self.logger.setLevel(eval('logging.{}'.format(self.args.logging_level)))

        self.logger.info('#'*120)
        self.logger.info(self.model_dir)

        self.condition = tf.placeholder(tf.int32, [self.args.n_questions], name='condition') 
        
        self.init_model()
        self.init_total_prediction_probability()
        self.init_mastery_level()


    def inference_with_counter(self, q_embed, correlation_weight, value_matrix, reuse_flag, counter):
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        ##### ADD new FC layer for q_embedding. There is an layer in MXnet implementation
        q_embed_content_logit = operations.linear(q_embed, 50, name='input_embed_content', reuse=reuse_flag)
        q_embed_content = tf.tanh(q_embed_content_logit)

        counter_content_logit = operations.linear(counter, 20, name='counter_content', reuse=reuse_flag)
        counter_content = tf.sigmoid(counter_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, q_embed_content, counter_content], 1)
        #mastery_level_prior_difficulty = tf.concat([read_content, q_embed_content], 1)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)

        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob

    def inference(self, q_embed, correlation_weight, value_matrix, reuse_flag):
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        ##### ADD new FC layer for q_embedding. There is an layer in MXnet implementation
        #q_embed_content_logit = operations.linear(q_embed, 50, name='input_embed_content', reuse=reuse_flag)
        #q_embed_content = tf.tanh(q_embed_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, q_embed], 1)
        #mastery_level_prior_difficulty = tf.concat([read_content, q_embed_content], 1)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)

        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob

    def calculate_mastery_level(self, value_matrix, reuse_flag):
        #self.mastery_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='mastery_value_matrix')
        #self.target_concept_index = tf.placeholder(tf.int32, name='target_concept_index')

        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)
        stacked_one_hot_correlation_weight = tf.tile(tf.expand_dims(one_hot_correlation_weight, 0), tf.stack([self.args.batch_size, 1, 1]))
        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(value_matrix, 1), tf.stack([1, self.args.memory_size, 1, 1]))

        # read_content : batch_size memory_size memory_state_dim 
        read_content = self.memory.value.read_for_mastery(stacked_mastery_value_matrix, stacked_one_hot_correlation_weight)
        #read_content = self.memory.value.read(stacked_mastery_value_matrix, one_hot_correlation_weight)
        #print('READ content shape')
        #print(read_content.shape)

        zero_q_embed = tf.zeros(shape=[self.args.batch_size, self.args.memory_size, self.args.memory_key_state_dim]) 
        #zero_q_embed = tf.zeros(shape=[self.args.memory_size,self.args.n_questions]) 

        #zero_q_embed_content_logit = operations.linear(zero_q_embed, 50, name='input_embed_content', reuse=True)
        #zero_q_embed_content = tf.tanh(zero_q_embed_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, zero_q_embed], 2)
        mastery_level_prior_difficulty_reshaped = tf.reshape(mastery_level_prior_difficulty, shape=[self.args.batch_size*self.args.memory_size, -1])
        #print('Mastery level prior difficulty')
        #print(mastery_level_prior_difficulty.shape)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty_reshaped, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)

        pred_logits_reshaped = tf.reshape(pred_logits, shape=[self.args.batch_size, -1])
        #print('HEllow owrld')
        #print(tf.shape(pred_logits_reshaped))

        return tf.sigmoid(pred_logits_reshaped)
        #self.concept_mastery_level = tf.sigmoid(pred_logits)

    def init_memory(self):
        with tf.variable_scope('Memory'):
            init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.memory_key_state_dim], \
                initializer=tf.random_normal_initializer(stddev=0.1))
            self.init_memory_value = tf.get_variable('value', [self.args.memory_size,self.args.memory_value_state_dim], \
                initializer=tf.random_normal_initializer(stddev=0.1))
                
        # Broadcast memory value tensor to match [batch size, memory size, memory state dim]
        # First expand dim at axis 0 so that makes 'batch size' axis and tile it along 'batch size' axis
        # tf.tile(inputs, multiples) : multiples length must be thes saame as the number of dimensions in input
        # tf.stack takes a list and convert each element to a tensor
        self.stacked_init_memory_value = tf.tile(tf.expand_dims(self.init_memory_value, 0), tf.stack([self.args.batch_size, 1, 1]))
                
        return DKVMN(self.args.memory_size, self.args.memory_key_state_dim, \
                self.args.memory_value_state_dim, init_memory_key=init_memory_key, init_memory_value=self.stacked_init_memory_value, args=self.args, name='DKVMN')

    def init_embedding_mtx(self):
        # Embedding to [batch size, seq_len, memory_state_dim(d_k or d_v)]
        with tf.variable_scope('Embedding'):
            # A
            self.q_embed_mtx = tf.get_variable('q_embed', [self.args.n_questions+1, self.args.memory_key_state_dim],\
                initializer=tf.random_normal_initializer(stddev=0.1))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))
            # B
            self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.random_normal_initializer(stddev=0.1))        
            #self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))        
        

    def embedding_q(self, q):
        return tf.nn.embedding_lookup(self.q_embed_mtx, q)

    def embedding_qa(self, qa):
        return tf.nn.embedding_lookup(self.qa_embed_mtx, qa)
        

    def calculate_knowledge_growth(self, value_matrix, correlation_weight, qa_embed, read_content, summary, pred_prob, mastery_level):
        if self.args.knowledge_growth == 'origin': 
            return qa_embed
        
        elif self.args.knowledge_growth == 'value_matrix':
            value_matrix_reshaped = tf.reshape(value_matrix, [self.args.batch_size, -1])
            return tf.concat([value_matrix_reshaped, qa_embed], 1)

        elif self.args.knowledge_growth == 'read_content':
            read_content_reshaped = tf.reshape(read_content, [self.args.batch_size, -1])
            return tf.concat([read_content_reshaped, qa_embed], 1)

        elif self.args.knowledge_growth == 'summary':
            summary_reshaped = tf.reshape(summary, [self.args.batch_size, -1])
            return tf.concat([summary_reshaped, qa_embed], 1)
 
        elif self.args.knowledge_growth == 'pred_prob':
            pred_prob_reshaped = tf.reshape(pred_prob, [self.args.batch_size, -1])
            return tf.concat([pred_prob_reshaped, qa_embed], 1)

        elif self.args.knowledge_growth == 'mastery':
            mastery_reshaped = tf.reshape(mastery_level, [self.args.batch_size, -1])
            return tf.concat([mastery_reshaped, qa_embed], 1)
            
            

    def init_model(self):
        # 'seq_len' means question sequences
        self.q_data_seq = tf.placeholder(tf.int32, [None, self.args.seq_len], name='q_data_seq') 
        self.qa_data_seq = tf.placeholder(tf.int32, [None, self.args.seq_len], name='qa_data')
        self.target_seq = tf.placeholder(tf.float32, [None, self.args.seq_len], name='target')

        self.selected_mastery_index = tf.placeholder(tf.int32, name='selected_mastery_index')

        '''
        self.q_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data_seq') 
        self.qa_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
        self.target_seq = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='target')
        '''

        self.memory = self.init_memory()
        self.init_embedding_mtx()
            
        slice_q_data = tf.split(self.q_data_seq, self.args.seq_len, 1) 
        slice_qa_data = tf.split(self.qa_data_seq, self.args.seq_len, 1) 

        
        prediction = list()
        mastery_level_list = list()
        reuse_flag = False

        counter = tf.zeros([self.args.batch_size, self.args.memory_key_state_dim])
        #counter = tf.zeros([self.args.batch_size, self.args.n_questions])

        mastery_level = self.calculate_mastery_level(self.stacked_init_memory_value, False)
        mastery_level_list.append(mastery_level)

        # Logics
        for i in range(self.args.seq_len):
            # To reuse linear vectors
            if i != 0:
                reuse_flag = True

            q = tf.squeeze(slice_q_data[i], 1)
            qa = tf.squeeze(slice_qa_data[i], 1)
            a = tf.cast(tf.greater(qa, tf.constant(self.args.n_questions)), tf.float32)

            '''
            one_hot_q = tf.one_hot(q, self.args.n_questions)
            counter = counter + one_hot_q
            '''

            q_embed = self.embedding_q(q)
            counter = counter + q_embed
            qa_embed = self.embedding_qa(qa)

            correlation_weight = self.memory.attention(q_embed)

            #prev_mastery_level = self.calculate_mastery_level(self.stacked_init_memory_value, True)
                
            #prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = self.inference_with_counter(q_embed, correlation_weight, self.memory.memory_value, reuse_flag, counter)
            prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = self.inference(q_embed, correlation_weight, self.memory.memory_value, True)
            prediction.append(prev_pred_logit)

            knowledge_growth = self.calculate_knowledge_growth(self.memory.memory_value, correlation_weight, qa_embed, prev_read_content, prev_summary, prev_pred_prob, mastery_level)
            self.memory.memory_value = self.memory.value.write_given_a(self.memory.memory_value, correlation_weight, knowledge_growth, a, reuse_flag)
            mastery_level = self.calculate_mastery_level(self.memory.memory_value, True)
            mastery_level_list.append(mastery_level)
            
        self.mastery_level_seq = mastery_level_list
        #self.prediction_seq = tf.sigmoid(prediction) 
        
        # 'prediction' : seq_len length list of [batch size ,1], make it [batch size, seq_len] tensor
        # tf.stack convert to [batch size, seq_len, 1]
        pred_logits = tf.reshape(tf.stack(prediction, axis=1), [self.args.batch_size, self.args.seq_len]) 
        self.pred = tf.sigmoid(pred_logits)

        # filtered by selected_mastery_index
        #self.target_seq = self.target_seq[:, self.selected_mastery_index+1:]
        #pred_logits = pred_logits[:, self.selected_mastery_index+1:]

        # Define loss : standard cross entropy loss, need to ignore '-1' label example
        # Make target/label 1-d array
        target_1d = tf.reshape(self.target_seq, [-1])
        pred_logits_1d = tf.reshape(pred_logits, [-1])
        index = tf.where(tf.not_equal(target_1d, tf.constant(-1., dtype=tf.float32)))
        # tf.gather(params, indices) : Gather slices from params according to indices
        filtered_target = tf.gather(target_1d, index)
        filtered_logits = tf.gather(pred_logits_1d, index)


        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))

        # Optimizer : SGD + MOMENTUM with learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        #self.learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, decay_steps=self.args.anneal_interval*(tf.shape(self.q_data_seq)[0] // self.args.batch_size), decay_rate=0.667, staircase=True)
        optimizer = tf.train.MomentumOptimizer(self.lr, self.args.momentum)
        grads, vrbs = zip(*optimizer.compute_gradients(self.loss))
        self.grads = grads
        grad, self.global_norm = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
        
        self.train_op = optimizer.apply_gradients(list(zip(grad, vrbs)), global_step=self.global_step)
        self.tr_vrbs = tf.trainable_variables()
        for i in self.tr_vrbs:
            print(i.name)
            print(i.shape)

        self.saver = tf.train.Saver(max_to_keep=1000)
        print('Finish init_model')


    def train(self, train_q_data, train_qa_data, valid_q_data, valid_qa_data, early_stop=False, checkpoint_dir='', selected_mastery_index=-1):
        #np.random.seed(224)
        # q_data, qa_data : [samples, seq_len]

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
            print('Checkpoint exists')
            #print('Checkpoint exists and skip training')
            #return 
        else:
            print('No checkpoint')
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

                feed_dict = {self.q_data_seq:q_batch_seq, self.qa_data_seq:qa_batch_seq, self.target_seq:target_batch, self.lr:lr, self.selected_mastery_index:selected_mastery_index}
                #loss_, pred_, _, = self.sess.run([self.loss, self.pred, self.train_op], feed_dict=feed_dict)
                #loss_, pred_, _, global_norm, grads, _lr = self.sess.run([self.loss, self.pred, self.train_op, self.global_norm, self.grads, self.learning_rate], feed_dict=feed_dict)
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
            print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (epoch+1, self.args.num_epochs, epoch_loss, train_auc, train_accuracy))
            self.write_log(epoch=epoch+1, auc=train_auc, accuracy=train_accuracy, loss=epoch_loss, name='training_')

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
                valid_feed_dict = {self.q_data_seq : valid_q, self.qa_data_seq : valid_qa, self.target_seq : valid_target, self.selected_mastery_index:selected_mastery_index}
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

            vliad_auc, valid_accuracy = dkvmn_utils.calculate_metric(all_valid_target, all_valid_pred)
            print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' %(epoch+1, self.args.num_epochs, valid_auc, valid_accuracy))
            # Valid log
            self.write_log(epoch=epoch+1, auc=valid_auc, accuracy=valid_accuracy, loss=valid_loss, name='valid_')
            if valid_auc > best_valid_auc:
                print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
                best_valid_auc = valid_auc
                best_epoch = epoch + 1
                self.save(best_epoch, checkpoint_dir)
                early_stop_counter = 0
            else:
                early_stop_counter += 1 

            if early_stop and early_stop_counter > self.args.early_stop_th:
                print('Eearly stop')
                return best_epoch

        return best_epoch    
    
    def test(self, test_q, test_qa, checkpoint_dir='', selected_mastery_index=-1):
        steps = test_q.shape[0] // self.args.batch_size
        self.sess.run(tf.global_variables_initializer())
        if self.load(checkpoint_dir):
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        #print('Initial value of probability')
        #print(init_probability)

        pred_list = list()
        #pred_list_2d = list()
        target_list = list()
        #target_list_2d = list()
        q_list = list()

        for s in range(steps):
            test_q_batch = test_q[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            test_qa_batch = test_qa[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            target = test_qa_batch[:,:]
            target = target.astype(np.int)
            target_batch = (target - 1) // self.args.n_questions  
            target_batch = target_batch.astype(np.float)
            feed_dict = {self.q_data_seq:test_q_batch, self.qa_data_seq:test_qa_batch, self.target_seq:target_batch, self.selected_mastery_index:selected_mastery_index }
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
        count, metric_for_each_q = dkvmn_utils.calculate_metric_for_each_q(all_target, all_pred, all_q, self.args.n_questions)
        #print(metric_for_each_q)
        for (idx, metric) in enumerate(metric_for_each_q):
            self.logger.info('{:<3d}: {:>7d}, {: .4f}, {: .4f}'.format(idx+1, count[idx], metric[0], metric[1]))
            

        self.logger.info('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))
        self.write_log(epoch=1, auc=test_auc, accuracy=test_accuracy, loss=0, name='test_')

        log_file_name = '{}_{}_test_result.txt'.format(self.args.prefix, self.args.dataset)
        log_file_path = os.path.join(self.args.dkvmn_test_result_dir, log_file_name)
        log_file = open(log_file_path, 'a')
        log = 'Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy)

        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)

        log_file.write(checkpoint_dir + '\n')
        #log_file.write(self.model_dir + '\n')
        log_file.write(log + '\n') 
        log_file.flush()    

        return pred_list_2d, target_list_2d, test_auc, test_accuracy
        


    def clustering_actions(self):
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        value_matrix = self.sess.run(self.init_memory_value)
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
        #fig = plt.figure()
        #fig.savefig('actions.png')
        
         

########################################################## FOR Reinforcement Learning ##############################################################

    def sampling_a_given_q(self, q, value_matrix):
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        _, _, _, pred_prob = self.inference(q_embed, correlation_weight, value_matrix, reuse_flag = True)

        # TODO : arguemnt check for various algorithms
        pred_prob = tf.clip_by_value(pred_prob, 0.3, 1.0)

        threshold = tf.random_uniform(pred_prob.shape)

        a = tf.cast(tf.less(threshold, pred_prob), tf.int32)
        qa = q + tf.multiply(a, self.args.n_questions)[0]

        return qa 


    def init_total_prediction_probability(self):
        #self.total_q_data = tf.placeholder(tf.int32, [self.args.n_questions], name='total_q_data') 
        self.total_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='total_value_matrix')

        total_q_data = tf.constant(np.arange(1,self.args.n_questions+1))
        q_embeds = self.embedding_q(total_q_data)
        self.total_correlation_weight = self.memory.attention(q_embeds)
       
        stacked_total_value_matrix = tf.tile(tf.expand_dims(self.total_value_matrix, 0), tf.stack([self.args.n_questions, 1, 1]))
        _, _, _, self.total_pred_probs = self.inference(q_embeds, self.total_correlation_weight, stacked_total_value_matrix, True)

    def init_step(self):
        # q : action for RL
        # value_matrix : state for RL
        self.q = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='step_q') 
        self.a = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='step_a') 
        self.value_matrix = tf.placeholder(tf.float32, [self.args.memory_size, self.args.memory_value_state_dim], name='step_value_matrix')

        slice_a = tf.split(self.a, self.args.seq_len, 1) 
        a = tf.squeeze(slice_a[0], 1)
 
        slice_q = tf.split(self.q, self.args.seq_len, 1) 
        q = tf.squeeze(slice_q[0], 1)
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        stacked_value_matrix = tf.tile(tf.expand_dims(self.value_matrix, 0), tf.stack([self.args.batch_size, 1, 1]))
         
        # -1 for sampling
        # 0, 1 for given answer
        self.qa = tf.cond(tf.squeeze(a) < 0, lambda: self.sampling_a_given_q(q, stacked_value_matrix), lambda: q + tf.multiply(a, self.args.n_questions))
        a = (self.qa-1) // self.args.n_questions
        qa_embed = self.embedding_qa(self.qa) 

        ######### Before Step ##########
        prev_read_content, prev_summary, prev_pred_logits, prev_pred_prob = self.inference(q_embed, correlation_weight, stacked_value_matrix, reuse_flag = True)
        prev_mastery_level = self.calculate_mastery_level(stacked_value_matrix, reuse_flag = True)

        ######### STEP #####################
        knowledge_growth = self.calculate_knowledge_growth(stacked_value_matrix, correlation_weight, qa_embed, prev_read_content, prev_summary, prev_pred_prob, prev_mastery_level)
        # TODO : refactor sampling_a_given_q to return a only for below function call
        self.stepped_value_matrix = tf.squeeze(self.memory.value.write_given_a(stacked_value_matrix, correlation_weight, knowledge_growth, a, True), axis=0)
        self.stepped_read_content, self.stepped_summary, self.stepped_pred_logits, self.stepped_pred_prob = self.inference(q_embed, correlation_weight, self.stepped_value_matrix, reuse_flag = True)

        ######### After Step #########
        self.value_matrix_difference = tf.squeeze(tf.reduce_sum(self.stepped_value_matrix - stacked_value_matrix))
        self.read_content_difference = tf.squeeze(tf.reduce_sum(self.stepped_read_content - prev_read_content))
        self.summary_difference = tf.squeeze(tf.reduce_sum(self.stepped_summary - prev_summary))
        self.pred_logit_difference = tf.squeeze(tf.reduce_sum(self.stepped_pred_logits - prev_pred_logits))
        self.pred_prob_difference = tf.squeeze(tf.reduce_sum(self.stepped_pred_prob - prev_pred_prob))

    def init_mastery_level(self):
        self.mastery_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='mastery_value_matrix')
        #self.target_concept_index = tf.placeholder(tf.int32, name='target_concept_index')

        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)
        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(self.mastery_value_matrix, 0), tf.stack([self.args.memory_size, 1, 1]))

        read_content = self.memory.value.read(stacked_mastery_value_matrix, one_hot_correlation_weight)
        print('READ content shape')
        print(read_content.shape)
        zero_q_embed = tf.zeros(shape=[self.args.memory_size, self.args.memory_key_state_dim]) 
        #zero_q_embed = tf.zeros(shape=[self.args.memory_size,self.args.n_questions]) 

        #zero_q_embed_content_logit = operations.linear(zero_q_embed, 50, name='input_embed_content', reuse=True)
        #zero_q_embed_content = tf.tanh(zero_q_embed_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, zero_q_embed], 1)
        print('Mastery level prior difficulty')
        print(mastery_level_prior_difficulty.shape)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=True)
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=True)

        self.concept_mastery_level = tf.sigmoid(pred_logits)

    '''
    def init_mastery_level(self):
        self.mastery_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='mastery_value_matrix')
        self.target_concept_index = tf.placeholder(tf.int32, name='target_concept_index')

        one_hot_correlation_weight = tf.one_hot(self.target_concept_index, self.args.memory_size)
        read_content = self.memory.value.read(self.mastery_value_matrix, one_hot_correlation_weight)
        print('READ content shape')
        print(read_content.shape)
        zero_q_embed = tf.zeros(shape=[1,50]) 
        mastery_level_prior_difficulty = tf.concat([read_content, zero_q_embed], 1)
        print('Mastery level prior difficulty')
        print(mastery_level_prior_difficulty.shape)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=True)
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=True)

        self.concept_mastery_level = tf.sigmoid(pred_logits)
    '''

        

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
        network_spec = 'Knowledge_{}_Summary_{}_Add_{}_Erase_{}_WriteType_{}'.format(self.args.knowledge_growth, self.args.summary_activation, self.args.add_signal_activation, self.args.erase_signal_activation, self.args.write_type)
 
        if network_spec == 'Knowledge_origin_Summary_tanh_Add_tanh_Erase_sigmoid_WriteType_add_on_erase_on':
            network_spec = 'Original'

        hyper_parameters = '_lr{}_{}epochs'.format(self.args.initial_lr, self.args.num_epochs)
        #hyper_parameters = '_lr{}_{}epochs_{}batch'.format(self.args.initial_lr, self.args.num_epochs, self.args.batch_size)
        #data_postfix = '_{}_{}_{}'.format(self.args.train_postfix, self.args.valid_postfix, self.args.test_postfix)

        remove_short = '_RemoveShort_{}_th_{}'.format(self.args.remove_short_seq, self.args.short_seq_len_th)
        
        return self.args.prefix + network_spec + hyper_parameters + remove_short
        #return self.args.prefix + network_spec + hyper_parameters + data_postfix

        #return '{}Knowledge_{}_Summary_{}_Add_{}_Erase_{}_WriteType_{}_lr{}_{}epochs'.format(self.args.prefix, self.args.knowledge_growth, self.args.summary_activation, self.args.add_signal_activation, self.args.erase_signal_activation, self.args.write_type, self.args.initial_lr, self.args.num_epochs)
        #return '{}Knowledge_{}_Summary_{}_Add_{}_Erase_{}_WriteType_{}_{}_lr{}_{}epochs'.format(self.args.prefix, self.args.knowledge_growth, self.args.summary_activation, self.args.add_signal_activation, self.args.erase_signal_activation, self.args.write_type, self.args.dataset, self.args.initial_lr, self.args.num_epochs)

    def load(self, checkpoint_dir=''):
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print('DKVMN ckpt loaded')
            return True
        else:
            print('DKVMN cktp not loaded')
            return False

    def save(self, global_step, checkpoint_dir=''):
        model_name = 'DKVMN'
        if checkpoint_dir == '':
            checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Save checkpoint at %d' % (global_step+1))

    # Log file
    def write_log(self, auc, accuracy, loss, epoch, name='training_'):
        log_path = os.path.join(self.args.dkvmn_log_dir, name+self.model_dir+'.csv')
        if not os.path.exists(log_path):
            self.log_file = open(log_path, 'w')
            self.log_file.write('Epoch\tAuc\tAccuracy\tloss\n')
        else:
            self.log_file = open(log_path, 'a')    
        
        self.log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(loss) + '\n')
        self.log_file.flush()    
        
