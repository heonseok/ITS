import tensorflow as tf
from memory import DKVMN
import numpy as np
import operations

class Mixin:

    def build_memory(self):
        with tf.variable_scope('Memory', reuse=tf.AUTO_REUSE):
            init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.memory_key_state_dim], initializer=tf.random_normal_initializer(stddev=0.1))
            self.init_memory_value = tf.get_variable('value', [self.args.memory_size,self.args.memory_value_state_dim], initializer=tf.random_normal_initializer(stddev=0.1))

        # Broadcast memory value tensor to match [batch size, memory size, memory state dim]
        # First expand dim at axis 0 so that makes 'batch size' axis and tile it along 'batch size' axis
        # tf.tile(inputs, multiples) : multiples length must be thes saame as the number of dimensions in input
        # tf.stack takes a list and convert each element to a tensor
        self.stacked_init_memory_value = tf.tile(tf.expand_dims(self.init_memory_value, 0), tf.stack([self.args.batch_size, 1, 1]))

        return DKVMN(self.args.memory_size, self.args.memory_key_state_dim,
                     self.args.memory_value_state_dim, init_memory_key=init_memory_key, init_memory_value=self.stacked_init_memory_value, args=self.args, name='DKVMN')

    def build_embedding_mtx(self):
        # Embedding to [batch size, seq_len, memory_state_dim(d_k or d_v)]
        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            # A
            self.q_embed_mtx = tf.get_variable('q_embed', [self.args.n_questions+1, self.args.memory_key_state_dim], \
                                               initializer=tf.random_normal_initializer(stddev=0.1))
            #initializer=tf.truncated_normal_initializer(stddev=0.1))
            # B
            self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.random_normal_initializer(stddev=0.1))
            #self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))


    def embedding_q(self, q):
        return tf.nn.embedding_lookup(self.q_embed_mtx, q)

    def embedding_qa(self, qa):
        return tf.nn.embedding_lookup(self.qa_embed_mtx, qa)

    def apply_summary_activation(self, summary_logit):
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        return summary_vector

    def inference_with_condition(self, q_embed, correlation_weight, value_matrix, counter):
        ###############################################################################################################
        # Output
        # read_content, summary, pred_logit, pred_prob
        ###############################################################################################################
        return self.inference_with_counter(q_embed, correlation_weight, value_matrix, counter),
        '''
        return tf.cond(self.using_counter_graph,
                       lambda:self.inference_with_counter(q_embed, correlation_weight, value_matrix, counter),
                       lambda:self.inference(q_embed, correlation_weight, value_matrix)
                      )
        '''
    def get_total_q(self):
        pass

    def get_counter_content(selfs):
        pass

    def get_summary(self):
        pass

    def inference_average_with_counter(self, value_matrix, counter):
        ###############################################################################################################
        # Input
        # value_matrix   (3dim) : [ batch_size, memory_size, value_memory_dim ]
        # counter        (2dim) : [ batch_size, n_questions+1 ]

        # Output
        # Probability    (2dim) : [ batch_size*n_questions, 1]

        # Intermediate
        # q_embeds       (2dim) : [ batch_size*n_questions, key_memory_dim    ]
        # read_content   (2dim) : [ batch_size*n_questions, value_memory_dim  ]
        # counter_embeds (2dim) : [ batch_size*n_questions, counter_embed_dim ]
        # summary        (2dim) : [ batch_size*n_questions, summary_dim       ]
        ###############################################################################################################

        # q_embeds : for all questions
        total_q = tf.range(1, self.args.n_questions+1)
        total_q_embeds = self.embedding_q(total_q)
        stacked_total_q_embeds = tf.tile(tf.expand_dims(total_q_embeds, 0), tf.stack([self.args.batch_size, 1,1]))
        stacked_total_q_embeds = tf.reshape(stacked_total_q_embeds, [self.args.batch_size*self.args.n_questions, -1])

        # read content : not using attention mechanism
        uniform_correlation_weight = tf.divide(tf.ones(shape=self.args.batch_size * self.args.memory_size, dtype=tf.float32), self.args.memory_size)
        averaged_read_content = self.memory.value.read(value_matrix, uniform_correlation_weight)

        stacked_averaged_read_content = tf.tile(tf.expand_dims(averaged_read_content,1), tf.stack([1, self.args.n_questions, 1]))
        stacked_averaged_read_content = tf.reshape(stacked_averaged_read_content, [self.args.batch_size*self.args.n_questions, -1])

        # counter
        counter_content_logit = operations.linear(tf.cast(counter, tf.float32), self.args.counter_embedding_dim, name='counter_content')
        counter_content = tf.sigmoid(counter_content_logit)

        stacked_counter_content = tf.tile(tf.expand_dims(counter_content, 1), tf.stack([1, self.args.n_questions, 1]))
        stacked_counter_content  = tf.reshape(stacked_counter_content, [self.args.batch_size*self.args.n_questions, -1])

        # summary
        summary_input = tf.concat([stacked_averaged_read_content, stacked_total_q_embeds, stacked_counter_content], 1)
        summary_logit = operations.linear(summary_input, self.args.final_fc_dim, name='Counter_Summary_Vector')
        summary_vector = self.apply_summary_activation(summary_logit)

        # probability for all questions
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')
        pred_prob = tf.sigmoid(pred_logits)

        return pred_prob

    def calc_summary(self, read_content, q_embed):
        summary_input = tf.concat([read_content, q_embed], 1)
        summary_logit = operations.linear(summary_input, self.args.final_fc_dim, name='Summary_Vector')
        return self.apply_summary_activation(summary_logit)

    def calc_summary_with_counter(self, read_content, q_embed, counter_embed):
        summary_input = tf.concat([read_content, q_embed, counter_embed], 1)
        summary_logit = operations.linear(summary_input, self.args.final_fc_dim, name='Counter_Summary_Vector')
        return self.apply_summary_activation(summary_logit)



    def inference(self, q_embed, correlation_weight, value_matrix, counter):
        ###############################################################################################################
        # Input
        # q_embeds           (2dim) : [ batch_size, key_memory_dim ]
        # correlation_weight (2dim) : [ batch_size, memory_size ]
        # value_matrix       (3dim) : [ batch_size, memory_size, value_memory_dim ]
        # counter            (2dim) : [ batch_size, n_questions+1 ]

        # Output
        # read_content       (2dim) : [ batch_size, value_memory_dim ]
        # summary            (2dim) : [ batch_size, summary_dim ]
        # Probability        (2dim) : [ batch_size, 1 ]

        # Intermediate
        # counter_embeds     (2dim) : [ batch_size, counter_embed_dim ]
        ###############################################################################################################

        # read content
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        # counter
        counter_content_logit = operations.linear(tf.cast(counter, tf.float32), self.args.counter_embedding_dim, name='counter_content')
        counter_content = tf.sigmoid(counter_content_logit)

        # summary
        summary_vector = tf.cond(self.using_counter_graph,
                          lambda:self.calc_summary_with_counter(read_content, q_embed, counter_content),
                          lambda:self.calc_summary(read_content, q_embed)
                         )

        '''
        summary_input = tf.concat([read_content, q_embed, counter_content], 1)
        summary_logit = operations.linear(summary_input, self.args.final_fc_dim, name='Counter_Summary_Vector')
        summary_vector = self.apply_summary_activation(summary_logit)
        '''

        # probabiltiy for input question
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')
        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob

    '''
    def inference(self, q_embed, correlation_weight, value_matrix):
        ###############################################################################################################
        # Input
        # q_embeds           (2dim) : [ batch_size, key_memory_dim ]
        # correlation_weight (2dim) : [ batch_size, memory_size ]
        # value_matrix       (3dim) : [ batch_size, memory_size, value_memory_dim ]

        # Output
        # read_content       (2dim) : [ batch_size, value_memory_dim ]
        # summary            (2dim) : [ batch_size, summary_dim ]
        # Probability        (2dim) : [ batch_size, 1 ]
        ###############################################################################################################

        # read content
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        # summary
        summary_input = tf.concat([read_content, q_embed], 1)
        summary_logit = operations.linear(summary_input, self.args.final_fc_dim, name='Summary_Vector')
        summary_vector = self.apply_summary_activation(summary_logit)

        # probability
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')
        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob
    '''

    # TODO : It should input value matrix/ counter?
    def calculate_mastery_level(self, value_matrix):
        zero_q_embed = tf.zeros(shape=[self.args.batch_size, self.args.memory_size, self.args.memory_key_state_dim])
        #zero_q_embed = tf.zeros(shape=[self.args.memory_size,self.args.n_questions])

        #zero_q_embed_content_logit = operations.linear(zero_q_embed, 50, name='input_embed_content', reuse=True)
        #zero_q_embed_content = tf.tanh(zero_q_embed_content_logit)

        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)
        stacked_one_hot_correlation_weight = tf.tile(tf.expand_dims(one_hot_correlation_weight, 0), tf.stack([self.args.batch_size, 1, 1]))
        #stacked_one_hot_correlation_weight = tf.tile(tf.expand_dims(one_hot_correlation_weight, 0), tf.stack([self.args.batch_size, 1, 1]))
        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(value_matrix, 1), tf.stack([1, self.args.memory_size, 1, 1]))

        # read_content : batch_size memory_size memory_state_dim 
        read_content = self.memory.value.read_for_mastery(stacked_mastery_value_matrix, stacked_one_hot_correlation_weight)
        #read_content = self.memory.value.read(stacked_mastery_value_matrix, one_hot_correlation_weight)
        #print(read_content.shape)


        # summary
        summary_input = tf.concat([read_content, zero_q_embed], 2)
        summary_input_reshaped = tf.reshape(summary_input, shape=[self.args.batch_size*self.args.memory_size, -1])
        summary_logit = operations.linear(summary_input_reshaped, self.args.final_fc_dim, name='Summary_Vector')
        summary_vector = self.apply_summary_activation(summary_logit)

        # probability
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')
        pred_logits_reshaped = tf.reshape(pred_logits, shape=[self.args.batch_size, -1])

        return tf.sigmoid(pred_logits_reshaped)

    '''
    def calculate_pred_probs(self, value_matrix, counter, using_counter_graph):

        ##### ATTENTION
        total_q_data = tf.constant(np.arange(1, self.args.n_questions+1))
        #print(total_q_data.shape)

        stacked_total_q_data = tf.tile(tf.expand_dims(total_q_data, 0), tf.stack([self.args.batch_size, 1]))
        #print(stacked_total_q_data.shape)

        stacked_total_q_data_reshaped = tf.reshape(stacked_total_q_data, [-1])
        #stacked_total_q_data_reshaped = tf.reshape(stacked_total_q_data, [self.args.batch_size * self.args.n_questions, -1])
        #print(stacked_total_q_data_reshaped.shape)

        q_embeds = self.embedding_q(stacked_total_q_data_reshaped)
        #print(q_embeds.shape)

        #print(q_embeds_reshaped.shape)

        total_correlation_weight = self.memory.attention(q_embeds)
        #print(total_correlation_weight.shape)

    
        ##### READ
        #stacked_total_correlation_weight = tf.tile(tf.expand_dims(total_correlation_weight, 0), tf.stack([self.args.batch_size, 1, 1]))
        #print(stacked_total_correlation_weight.shape)
        
        #print(value_matrix.shape)
        stacked_value_matrix = tf.tile(tf.expand_dims(value_matrix, 1), tf.stack([1, self.args.n_questions, 1, 1]))
        #print(stacked_value_matrix.shape)
        stacked_value_matrix_reshaped =  tf.reshape(stacked_value_matrix, [-1, self.args.memory_size, self.args.memory_value_state_dim])
        #print(stacked_value_matrix_reshaped.shape)
        #stacked_total_value_matrix = tf.tile(tf.expand_dims(value_matrix, 0), tf.stack([1, self.args.n_questions, 1, 1]))

        #print(counter.shape)
        stacked_counter = tf.tile(tf.expand_dims(counter, 1), tf.stack([1, self.args.n_questions, 1]))
        #print(stacked_counter.shape)
        stacked_counter_reshaped = tf.reshape(stacked_counter, [self.args.batch_size*self.args.n_questions, -1])
        #print(stacked_counter_reshaped.shape)
        #_, _, _, self.total_pred_probs = self.inference(q_embeds, self.total_correlation_weight, stacked_total_value_matrix)
        _, _, _, total_pred_probs = tf.cond(using_counter_graph, lambda:self.inference_with_counter(q_embeds, total_correlation_weight, stacked_value_matrix_reshaped, stacked_counter_reshaped), lambda:self.inference(q_embeds, total_correlation_weight, stacked_value_matrix_reshaped))

        return total_pred_probs
    '''



    def calculate_knowledge_growth(self, value_matrix, qa_embed, read_content, summary, pred_prob, mastery_level):
        qa_embed = tf.reshape(qa_embed, [self.args.batch_size, -1])
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

    def build_model(self):
        # 'seq_len' means question sequences
        #### TODO : Naming... terrible case. it should be fixed after paper published
        self.q_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data_seq') 
        self.qa_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
        self.target_seq = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='target')

        self.selected_mastery_index = tf.placeholder(tf.int32, name='selected_mastery_index')

        self.memory = self.build_memory()
        self.build_embedding_mtx()
            
        slice_q_data = tf.split(self.q_data_seq, self.args.seq_len, 1) 
        slice_qa_data = tf.split(self.qa_data_seq, self.args.seq_len, 1) 
        slice_target = tf.split(self.target_seq, self.args.seq_len, 1)

        prediction = list()
        mastery_level_list = list()

        #mastery_level = self.calculate_mastery_level(self.stacked_init_memory_value, False)
        #mastery_level_list.append(mastery_level)

        self.q_counter  = tf.get_variable('self.q_counter', [self.args.batch_size, self.args.n_questions+1], dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)
        self.counter_loss = 0   
        self.negative_loss_term = 0

        q_constant = tf.constant(np.arange(self.args.batch_size), dtype=tf.int32)
        # TODO : add tf.cond for non-using counter case
        prev_total_pred_probs = self.inference_average_with_counter(self.memory.memory_value, self.q_counter)
        # Logics
        for i in range(self.args.seq_len):

            q = tf.squeeze(slice_q_data[i], 1)
            q_concat = tf.stack([q_constant, q], axis=1)

            qa = tf.squeeze(slice_qa_data[i], 1)
            a = tf.cast(tf.greater(qa, tf.constant(self.args.n_questions)), tf.float32)

            target = tf.squeeze(slice_target[i], 1)

            q_embed = self.embedding_q(q)
            qa_embed = self.embedding_qa(qa)

            correlation_weight = self.memory.attention(q_embed)

            # TODO : There was a bug 
            # mastery level 
            mastery_level = self.calculate_mastery_level(self.memory.memory_value)
            #mastery_level = self.calculate_mastery_level(self.stacked_init_memory_value)

            #prev_total_pred_probs = self.calculate_pred_probs(self.memory.memory_value, self.q_counter, self.using_counter_graph)
            #print(prev_total_pred_probs.shape)
                
            prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = self.inference(q_embed, correlation_weight, self.memory.memory_value, self.q_counter)

            prediction.append(prev_pred_logit)

            knowledge_growth = self.calculate_knowledge_growth(self.memory.memory_value, qa_embed, prev_read_content, prev_summary, prev_pred_prob, mastery_level)
            self.memory.memory_value = self.memory.value.write_given_a(self.memory.memory_value, correlation_weight, knowledge_growth, a)

            mastery_level = self.calculate_mastery_level(self.memory.memory_value)
            #total_pred_probs = self.calculate_pred_probs(self.memory.memory_value, self.q_counter, self.using_counter_graph)
            total_pred_probs = self.inference_average_with_counter(self.memory.memory_value, self.q_counter)

            mastery_level_list.append(mastery_level)

            total_pred_probs_diff = total_pred_probs - prev_total_pred_probs
            prev_total_pred_probs = total_pred_probs


            ## calculate counter loss
            ## TODO : remove -1 target.
            valid_idx = tf.where(tf.not_equal(target, tf.constant(-1, dtype=tf.float32)))
            #print('valid_idx : ' + str(valid_idx))
            #valid_q = tf.gather(q, valid_idx)
            #valid_q = tf.squeeze(tf.gather(q, valid_idx))
            #print('valid_q : ' + str(valid_q))

            p = tf.gather(tf.sigmoid(prev_pred_logit) , valid_idx)

            probs_diff = tf.gather(total_pred_probs_diff, valid_idx)
            negative_idx = tf.where(tf.less(probs_diff, 0))
            probs_diff_negative = tf.gather_nd(probs_diff, negative_idx)

            self.negative_loss_term += tf.reduce_mean(tf.square(probs_diff_negative))
            '''
            print(p.shape)
            print((-p*tf.log(p)).shape)
            print(tf.gather(self.q_counter,valid_q, axis=0).shape)
            print(tf.gather(self.q_counter,valid_q, axis=1).shape)
            print(tf.gather(self.q_counter,valid_q, axis=2).shape)
            '''

            valid_counter = tf.gather_nd(self.q_counter, q_concat)
            #print('valid_counter : ' + str(valid_counter))
            valid_counter = tf.squeeze(tf.gather(valid_counter, valid_idx))
            #print('valid_counter : ' + str(valid_counter))
            #print('entropy')
            #print( (tf.squeeze(-p*tf.log(p))).shape)
            #loss_term = tf.squeeze((-p * tf.log(p)))
            loss_term = tf.squeeze(tf.square(1-p))
            #print(loss_term.shape)
            #print(tf.squeeze(loss_term).shape)
            #print(tf.squeeze(valid_counter).shape)
            self.counter_loss += tf.reduce_mean(tf.cast(valid_counter, tf.float32) * loss_term)

            #self.counter_loss += tf.reduce_mean(tf.squeeze(tf.cast(tf.gather(self.q_counter,valid_q, axis=1), tf.float32)) * tf.squeeze((-p * tf.log(p))))

            #couter_loss = tf.cast(tf.gather(self.q_counter,valid_q), tf.float32) * (1-p)

            q_one_hot = tf.one_hot(q, self.args.n_questions+1, dtype=tf.int32)
            self.q_counter += q_one_hot 
            
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

 
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target)) +\
                    self.args.counter_loss_weight * self.counter_loss +\
                    self.args.decrease_loss_weight * self.negative_loss_term

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

########################################################## FOR Reinforcement Learning ##############################################################

    def sampling_a_given_q(self, q, value_matrix):
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        _, _, _, pred_prob = self.inference(q_embed, correlation_weight, value_matrix)

        # TODO : arguemnt check for various algorithms
        pred_prob = tf.clip_by_value(pred_prob, 0.3, 1.0)

        threshold = tf.random_uniform(pred_prob.shape)

        a = tf.cast(tf.less(threshold, pred_prob), tf.int32)
        qa = q + tf.multiply(a, self.args.n_questions)[0]

        return qa 

    def build_step_graph(self):
        self.q = tf.placeholder(tf.int32, [1, 1], name='step_q')
        self.a = tf.placeholder(tf.int32, [1, 1], name='step_a') 
        self.value_matrix = tf.placeholder(tf.float32, [self.args.memory_size, self.args.memory_value_state_dim], name='step_value_matrix')

        self.counter = tf.placeholder(tf.int32, [1, self.args.n_questions+1], name='step_counter')

        #slice_a = tf.split(self.a, self.args.seq_len, 1)
        #a = tf.squeeze(slice_a[0], 1)
        a = tf.squeeze(self.a, 1)
        #a = self.a
 
        #slice_q = tf.split(self.q, self.args.seq_len, 1) 
        #q = tf.squeeze(self.q)
        q = tf.squeeze(self.q, 1)
        #q = self.q
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        stacked_value_matrix = tf.tile(tf.expand_dims(self.value_matrix, 0), tf.stack([1, 1, 1]))
        #stacked_value_matrix = tf.tile(tf.expand_dims(self.value_matrix, 0), tf.stack([self.args.batch_size, 1, 1]))
         
        # -1 for sampling
        # 0, 1 for given answer
        self.qa = tf.cond(tf.squeeze(a) < 0,
                          lambda: self.sampling_a_given_q(q, stacked_value_matrix),
                          lambda: q + tf.multiply(a, self.args.n_questions)
                          )

        a = (self.qa-1) // self.args.n_questions
        qa_embed = self.embedding_qa(self.qa) 

        ## Before Step
        prev_read_content, prev_summary, prev_pred_logits, prev_pred_prob = self.inference(q_embed, correlation_weight, stacked_value_matrix, self.counter)
        prev_mastery_level = self.calculate_mastery_level(stacked_value_matrix)

        ## Step
        knowledge_growth = self.calculate_knowledge_growth(stacked_value_matrix, qa_embed, prev_read_content, prev_summary, prev_pred_prob, prev_mastery_level)
        # TODO : refactor sampling_a_given_q to return a only for below function call
        self.stepped_value_matrix = tf.squeeze(self.memory.value.write_given_a(stacked_value_matrix, correlation_weight, knowledge_growth, a), axis=0)

        ##### Building other useful graphs
        # TODO : remove using_counter graph because it is member variable of model
        self.build_total_prob_graph(self.value_matrix, self.counter, self.using_counter_graph)
        self.build_mastery_graph(self.value_matrix, self.counter, self.using_counter_graph)


    def build_total_prob_graph(self, value_matrix, counter, using_counter_graph):

        total_q_data = tf.constant(tf.range(1,self.args.n_questions+1))
        q_embeds = self.embedding_q(total_q_data)

        self.total_correlation_weight = self.memory.attention(q_embeds)
       
        stacked_total_value_matrix = tf.tile(tf.expand_dims(value_matrix, 0), tf.stack([self.args.n_questions, 1, 1]))
        stacked_total_counter = tf.tile(counter, tf.stack([self.args.n_questions, 1]))

        _, _, _, self.total_pred_probs = self.inference(q_embeds, self.total_correleation_wewgith, stacked_total_value_matrix, stacked_total_counter)

    def build_mastery_graph(self, value_matrix, counter, using_counter_graph):

        zero_q_embed = tf.zeros(shape=[self.args.memory_size, self.args.memory_key_state_dim])
        #zero_q_embed = tf.zeros(shape=[self.args.memory_size,self.args.n_questions])

        #zero_q_embed_content_logit = operations.linear(zero_q_embed, 50, name='input_embed_content', reuse=True)
        #zero_q_embed_content = tf.tanh(zero_q_embed_content_logit)

        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)

        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(value_matrix, 0), tf.stack([self.args.memory_size, 1, 1]))
        stacked_mastery_counter = tf.tile(counter, tf.stack([self.args.memory_size, 1]))

        read_content = self.memory.value.read(stacked_mastery_value_matrix, one_hot_correlation_weight)


        # counter
        counter_content_logit = operations.linear(tf.cast(stacked_mastery_counter, tf.float32), self.args.counter_embedding_dim, name='counter_content')
        counter_content = tf.sigmoid(counter_content_logit)

        # summary
        summary_input = tf.cond(using_counter_graph,
                                                 lambda: tf.concat([read_content, zero_q_embed, counter_content], 1),
                                                 lambda: tf.concat([read_content, zero_q_embed], 1)
                                                )
        summary_logit = operations.linear(summary_input, self.args.final_fc_dim, name='Counter_Summary_Vector')
        summary_vector = self.apply_summary_activation(summary_logit)

        # mastery
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')
        self.concept_mastery_level = tf.sigmoid(pred_logits)
