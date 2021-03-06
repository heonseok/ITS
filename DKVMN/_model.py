import tensorflow as tf
from memory import DKVMN
import numpy as np
import operations

class Mixin:

    def inference_with_counter(self, q_embed, correlation_weight, value_matrix, counter):
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        ##### ADD new FC layer for q_embedding. There is an layer in MXnet implementation
        #q_embed_content_logit = operations.linear(q_embed, 50, name='input_embed_content', reuse=reuse_flag)
        #q_embed_content = tf.tanh(q_embed_content_logit)

        counter_content_logit = operations.linear(counter, 20, name='counter_content')
        counter_content = tf.sigmoid(counter_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, q_embed, counter_content], 1)
        #mastery_level_prior_difficulty = tf.concat([read_content, q_embed_content], 1)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Counter_Summary_Vector')
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')

        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob

    def inference(self, q_embed, correlation_weight, value_matrix):
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        ##### ADD new FC layer for q_embedding. There is an layer in MXnet implementation
        #q_embed_content_logit = operations.linear(q_embed, 50, name='input_embed_content', reuse=reuse_flag)
        #q_embed_content = tf.tanh(q_embed_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, q_embed], 1)
        #mastery_level_prior_difficulty = tf.concat([read_content, q_embed_content], 1)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector')
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')

        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob

    def calculate_mastery_level(self, value_matrix):
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
        summary_logit = operations.linear(mastery_level_prior_difficulty_reshaped, self.args.final_fc_dim, name='Summary_Vector')
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')

        pred_logits_reshaped = tf.reshape(pred_logits, shape=[self.args.batch_size, -1])
        #print(tf.shape(pred_logits_reshaped))

        return tf.sigmoid(pred_logits_reshaped)
        #self.concept_mastery_level = tf.sigmoid(pred_logits)

    def build_memory(self):
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

    def build_embedding_mtx(self):
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



    def build_model(self):
        # 'seq_len' means question sequences
        self.q_data_seq = tf.placeholder(tf.int32, [None, self.args.seq_len], name='q_data_seq') 
        self.qa_data_seq = tf.placeholder(tf.int32, [None, self.args.seq_len], name='qa_data')
        self.target_seq = tf.placeholder(tf.float32, [None, self.args.seq_len], name='target')

        self.selected_mastery_index = tf.placeholder(tf.int32, name='selected_mastery_index')

        self.using_counter = tf.placeholder(tf.bool)

        self.memory = self.build_memory()
        self.build_embedding_mtx()
            
        slice_q_data = tf.split(self.q_data_seq, self.args.seq_len, 1) 
        slice_qa_data = tf.split(self.qa_data_seq, self.args.seq_len, 1) 

        
        prediction = list()
        mastery_level_list = list()

        counter = tf.zeros([self.args.batch_size, self.args.memory_key_state_dim])
        #counter = tf.zeros([self.args.batch_size, self.args.n_questions])

        #mastery_level = self.calculate_mastery_level(self.stacked_init_memory_value, False)
        #mastery_level_list.append(mastery_level)

        # Logics
        for i in range(self.args.seq_len):

            q = tf.squeeze(slice_q_data[i], 1)
            qa = tf.squeeze(slice_qa_data[i], 1)
            a = tf.cast(tf.greater(qa, tf.constant(self.args.n_questions)), tf.float32)

            #one_hot_q = tf.one_hot(q, self.args.n_questions)
            #counter = counter + one_hot_q

            q_embed = self.embedding_q(q)
            counter = counter + q_embed
            qa_embed = self.embedding_qa(qa)

            correlation_weight = self.memory.attention(q_embed)

            mastery_level = self.calculate_mastery_level(self.stacked_init_memory_value)
                
            prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = tf.cond(self.using_counter, lambda:self.inference_with_counter(q_embed, correlation_weight, self.memory.memory_value, counter), lambda:self.inference(q_embed, correlation_weight, self.memory.memory_value))
            #prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = self.inference_with_counter(q_embed, correlation_weight, self.memory.memory_value, reuse_flag, counter)
            #prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = self.inference(q_embed, correlation_weight, self.memory.memory_value, True)
            prediction.append(prev_pred_logit)

            knowledge_growth = self.calculate_knowledge_growth(self.memory.memory_value, correlation_weight, qa_embed, prev_read_content, prev_summary, prev_pred_prob, mastery_level)
            self.memory.memory_value = self.memory.value.write_given_a(self.memory.memory_value, correlation_weight, knowledge_growth, a)
            mastery_level = self.calculate_mastery_level(self.memory.memory_value)
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


    def build_total_prob_graph(self):
        #self.total_q_data = tf.placeholder(tf.int32, [self.args.n_questions], name='total_q_data') 
        self.total_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='total_value_matrix')

        total_q_data = tf.constant(np.arange(1,self.args.n_questions+1))
        q_embeds = self.embedding_q(total_q_data)
        self.total_correlation_weight = self.memory.attention(q_embeds)
       
        stacked_total_value_matrix = tf.tile(tf.expand_dims(self.total_value_matrix, 0), tf.stack([self.args.n_questions, 1, 1]))
        _, _, _, self.total_pred_probs = self.inference(q_embeds, self.total_correlation_weight, stacked_total_value_matrix)

    def build_step_graph(self):
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
        prev_read_content, prev_summary, prev_pred_logits, prev_pred_prob = self.inference(q_embed, correlation_weight, stacked_value_matrix)
        prev_mastery_level = self.calculate_mastery_level(stacked_value_matrix)

        ######### STEP #####################
        knowledge_growth = self.calculate_knowledge_growth(stacked_value_matrix, correlation_weight, qa_embed, prev_read_content, prev_summary, prev_pred_prob, prev_mastery_level)
        # TODO : refactor sampling_a_given_q to return a only for below function call
        self.stepped_value_matrix = tf.squeeze(self.memory.value.write_given_a(stacked_value_matrix, correlation_weight, knowledge_growth, a), axis=0)
        self.stepped_read_content, self.stepped_summary, self.stepped_pred_logits, self.stepped_pred_prob = self.inference(q_embed, correlation_weight, self.stepped_value_matrix)

        ######### After Step #########
        self.value_matrix_difference = tf.squeeze(tf.reduce_sum(self.stepped_value_matrix - stacked_value_matrix))
        self.read_content_difference = tf.squeeze(tf.reduce_sum(self.stepped_read_content - prev_read_content))
        self.summary_difference = tf.squeeze(tf.reduce_sum(self.stepped_summary - prev_summary))
        self.pred_logit_difference = tf.squeeze(tf.reduce_sum(self.stepped_pred_logits - prev_pred_logits))
        self.pred_prob_difference = tf.squeeze(tf.reduce_sum(self.stepped_pred_prob - prev_pred_prob))

    def build_mastery_graph(self):
        mastery_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='mastery_value_matrix')
        #self.target_concept_index = tf.placeholder(tf.int32, name='target_concept_index')

        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)
        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(mastery_value_matrix, 0), tf.stack([self.args.memory_size, 1, 1]))

        read_content = self.memory.value.read(stacked_mastery_value_matrix, one_hot_correlation_weight)
        print(read_content.shape)
        zero_q_embed = tf.zeros(shape=[self.args.memory_size, self.args.memory_key_state_dim]) 
        #zero_q_embed = tf.zeros(shape=[self.args.memory_size,self.args.n_questions]) 

        #zero_q_embed_content_logit = operations.linear(zero_q_embed, 50, name='input_embed_content', reuse=True)
        #zero_q_embed_content = tf.tanh(zero_q_embed_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, zero_q_embed], 1)
        print('Mastery level prior difficulty')
        print(mastery_level_prior_difficulty.shape)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector')
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction')

        self.concept_mastery_level = tf.sigmoid(pred_logits)
