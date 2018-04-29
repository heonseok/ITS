import tensorflow as tf
from memory import DKVMN
import numpy as np
import operations

class Mixin:

    def build_memory(self):
        with tf.variable_scope('Memory', reuse=tf.AUTO_REUSE):
            init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.key_memory_dim],
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            self.init_memory_value = tf.get_variable('value', [self.args.memory_size,self.args.value_memory_dim],
                                                     initializer=tf.random_normal_initializer(stddev=0.1))

        # Broadcast memory value tensor to match [batch size, memory size, memory state dim]
        # First expand dim at axis 0 so that makes 'batch size' axis and tile it along 'batch size' axis
        # tf.tile(inputs, multiples) : multiples length must be thes saame as the number of dimensions in input
        # tf.stack takes a list and convert each element to a tensor
        stacked_init_memory_value = tf.tile(tf.expand_dims(self.init_memory_value, 0),
                                                 tf.stack([self.args.batch_size, 1, 1]))

        return DKVMN(self.args.memory_size, self.args.key_memory_dim, self.args.value_memory_dim,
                     init_memory_key=init_memory_key, init_memory_value=stacked_init_memory_value, args=self.args)

    def build_embedding_mtx(self):
        # todo : seq_len dimension exists?
        # Embedding to [batch size, seq_len, memory_state_dim(d_k or d_v)]
        # initializer=tf.truncated_normal_initializer(stddev=0.1))

        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            self.q_embed_mtx = tf.get_variable('q_embed', [self.args.n_questions+1, self.args.key_memory_dim],
                                               initializer=tf.random_normal_initializer(stddev=0.1))
            self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.value_memory_dim],
                                                initializer=tf.random_normal_initializer(stddev=0.1))

    def embedding_q(self, q):
        return tf.nn.embedding_lookup(self.q_embed_mtx, q)

    def embedding_qa(self, qa):
        return tf.nn.embedding_lookup(self.qa_embed_mtx, qa)

    def embedding_counter(self, counter):
        counter_embed_logit = operations.linear(tf.cast(counter, tf.float32),
                                                self.args.counter_embedding_dim, name='counter_content')
        return tf.sigmoid(counter_embed_logit)

    def apply_summary_activation(self, summary_logit):
        if self.args.summary_activation == 'tanh':
            summary = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary = tf.nn.relu(summary_logit)

        return summary

    def calc_summary(self, read, q_embed):
        ################################################################################################################
        # Input
        # read               (2dim) : [ batch_size, value_memory_dim ]
        # q_embeds           (2dim) : [ batch_size, key_memory_dim ]

        # Output
        # summary            (2dim) : [ batch_size, summary_dim ]
        ################################################################################################################

        summary_input = tf.concat([read, q_embed], 1)
        summary_logit = operations.linear(summary_input, self.args.summary_dim, name='Summary_Vector')
        return self.apply_summary_activation(summary_logit)

    def calc_summary_with_counter(self, read, q_embed, counter_embed):
        ################################################################################################################
        # Input
        # read               (2dim) : [ batch_size, value_memory_dim ]
        # q_embeds           (2dim) : [ batch_size, key_memory_dim ]
        # counter_embed      (2dim) : [ batch_size, counter_embed_dim ]

        # Output
        # summary            (2dim) : [ batch_size, summary_dim ]
        ################################################################################################################

        summary_input = tf.concat([read, q_embed, counter_embed], 1)
        summary_logit = operations.linear(summary_input, self.args.summary_dim, name='Counter_Summary_Vector')
        return self.apply_summary_activation(summary_logit)

    def inference(self, q_embed, read, counter_embed):
        ################################################################################################################
        # -1 dim means
        # batch_attention           : batch_size
        # skill_attention           : n_question
        # batch_skill_uniform       : bach_size*n_questions
        # step?                     : 1

        # Input should be reshaped to below form
        # q_embed            (2dim) : [ -1, key_memory_dim ]
        # read               (2dim) : [ -1, value_memory_dim ]
        # counter_embed      (2dim) : [ -1, n_questions+1 ]

        # Output
        # read               (2dim) : [ -1, value_memory_dim ]
        # summary            (2dim) : [ -1, summary_dim ]
        # Probability        (2dim) : [ -1, 1 ]
        ################################################################################################################

        # reshape
        q_embed = tf.reshape(q_embed, [-1, self.args.key_memory_dim])
        read = tf.reshape(read, [-1, self.args.value_memory_dim])
        counter_embed = tf.reshape(counter_embed, [-1, self.args.counter_embedding_dim])

        # summary
        summary = tf.cond(self.using_counter_graph,
                          lambda: self.calc_summary_with_counter(read, q_embed, counter_embed),
                          lambda: self.calc_summary(read, q_embed) )

        # probability for input question
        prob_logit = operations.linear(summary, 1, name='Prediction')
        prob = tf.sigmoid(prob_logit)

        return read, summary, prob_logit, prob

    def batch_attention_inference(self, q, value_matrix, counter):
        ################################################################################################################
        # pre-process data for inference

        # Input
        # q                  (2dim) : [ batch_size, 1 ]
        # value_matrix       (3dim) : [ batch_size, memory_size, value_memory_dim ]
        # counter            (2dim) : [ batch_size, n_question+1 ]

        # Output
        # read               (2dim) : [ batch_size, value_memory_dim ]
        # summary            (2dim) : [ batch_size, summary_dim ]
        # Probability        (2dim) : [ batch_size, 1 ]
        ################################################################################################################

        # q
        q_embed = self.embedding_q(q)

        # read
        correlation_weight = self.memory.attention(q_embed)
        read = self.memory.value.read(value_matrix, correlation_weight)

        # counter_embed
        counter_embed = self.embedding_counter(counter)

        return self.inference(q_embed, read, counter_embed)

    def skill_attention_inference(self, value_matrix, counter):
        ################################################################################################################
        # pre-process data for inference

        # Input
        # value_matrix       (2dim) : [ memory_size, value_memory_dim ]
        # counter            (2dim) : [ 1, n_question+1 ]

        # Intermediate
        # total_q            (2dim) : [ n_question, 1 ]

        # Output
        # read               (2dim) : [ n_question, value_memory_dim ]
        # summary            (2dim) : [ n_question, summary_dim ]
        # Probability        (2dim) : [ n_question, 1 ]
        ################################################################################################################

        # q_embed
        total_q = tf.range(1, self.args.n_questions+1)
        q_embed = self.embedding_q(total_q)

        # read
        self.total_correlation_weight = self.memory.attention(q_embed)
        stacked_value_matrix = tf.tile(tf.expand_dims(value_matrix, 0), tf.stack([self.args.n_questions, 1, 1]))
        stacked_read = self.memory.value.read(stacked_value_matrix, self.total_correlation_weight)

        # counter_embed
        counter_embed = self.embedding_counter(counter)
        stacked_counter_embed = tf.tile(counter_embed, tf.stack([self.args.n_questions, 1]))

        _, _, _, prob = self.inference(q_embed, stacked_read, stacked_counter_embed)
        # _, _, _, self.total_pred_probs = self.inference(q_embed, stacked_read, stacked_counter_embed)

        return prob

    def batch_skill_uniform_inference(self, value_matrix, counter):
        ################################################################################################################
        # pre-process data for inference

        # Input
        # value_matrix          (3dim) : [ batch_size, memory_size, value_memory_dim ]
        # counter               (2dim) : [ batch_size, n_question+1 ]

        # Intermediate
        # total_q               (2dim) : [ n_question, 1 ]
        # stacked_q_embed       (3dim) : [ batch_size, n_question, key_memory_dim ]
        # stacked_read          (3dim) : [ batch_size, n_question, value_memory_dim ]
        # stacked_counter_embed (3dim) : [ batch_size, n_question, counter_embed_dim ]

        # Output
        # read                  (2dim) : [ batch_size*n_question, value_memory_dim ]
        # summary               (2dim) : [ batch_size*n_question, summary_dim ]
        # Probability           (2dim) : [ batch_size*n_question, 1 ]
        ################################################################################################################

        # q_embed
        total_q = tf.range(1, self.args.n_questions+1)
        q_embed = self.embedding_q(total_q)
        stacked_q_embed = tf.tile(tf.expand_dims(q_embed, 0), tf.stack([self.args.batch_size, 1, 1]))

        # read
        uniform_correlation_weight = tf.ones(self.args.batch_size*self.args.memory_size, tf.float32)
        uniform_correlation_weight = tf.divide(uniform_correlation_weight, self.args.memory_size)
        read = self.memory.value.read(value_matrix, uniform_correlation_weight)
        stacked_read = tf.tile(tf.expand_dims(read, 1), tf.stack([1, self.args.n_questions, 1]))

        # counter_embed
        counter_embed = self.embedding_counter(counter)
        stacked_counter_embed = tf.tile(tf.expand_dims(counter_embed, 1), tf.stack([1, self.args.n_questions, 1]))

        _, _, _, prob = self.inference(stacked_q_embed, stacked_read, stacked_counter_embed)

        return prob

    def concept_attention_inference_mastery(self, value_matrix, counter):
    # def build_mastery_graph(self, value_matrix, counter):
        ################################################################################################################
        # pre-process data for inference mastery

        # Input
        # value_matrix          (2dim) : [ memory_size, value_memory_dim ]
        # counter               (2dim) : [ 1, n_question+1 ]

        # Intermediate
        # zero_q_embed          (2dim) : [ memory_size, key_memory_dim ]

        # Output
        # mastery?               (2dim) : [ 1, memory_size ]
        ################################################################################################################

        # q_embed
        zero_q_embed = tf.zeros(shape=[self.args.memory_size, self.args.key_memory_dim])

        # read
        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)
        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(value_matrix, 0), tf.stack([self.args.memory_size, 1, 1]))
        read = self.memory.value.read(stacked_mastery_value_matrix, one_hot_correlation_weight)

        # counter_embed
        counter_embed = self.embedding_counter(counter)
        stacked_counter_embed = tf.tile(counter_embed, tf.stack([self.args.memory_size, 1]))

        _, _, _, prob = self.inference(zero_q_embed, read, stacked_counter_embed)
        return prob

    def batch_concept_attention_inference_mastery(self, value_matrix, counter):
        ################################################################################################################
        # pre-process data for inference mastery

        # Input
        # value_matrix          (3dim) : [ batch_size, memory_size, value_memory_dim ]
        # counter               (2dim) : [ batch_size, n_question+1 ]

        # Intermediate
        # stacked_zero_q_embed  (3dim) : [ batch_size, memory_size, key_memory_dim ]

        # Output
        # mastery               (2dim) : [ batch_size, memory_size ]
        ################################################################################################################

        # q_embed
        stacked_zero_q_embed = tf.zeros(shape=[self.args.batch_size, self.args.memory_size, self.args.key_memory_dim])

        # read
        one_hot_correlation_weight = tf.one_hot(np.arange(self.args.memory_size), self.args.memory_size)
        stacked_one_hot_correlation_weight = tf.tile(tf.expand_dims(one_hot_correlation_weight, 0),
                                                     tf.stack([self.args.batch_size, 1, 1]))
        stacked_mastery_value_matrix = tf.tile(tf.expand_dims(value_matrix, 1),
                                               tf.stack([1, self.args.memory_size, 1, 1]))
        stacked_read = self.memory.value.read(stacked_mastery_value_matrix, stacked_one_hot_correlation_weight)

        # counter_embed
        counter_embed = self.embedding_counter(counter)
        stacked_counter_embed = tf.tile(tf.expand_dims(counter_embed, 1), tf.stack([1, self.args.memory_size, 1]))

        _, _, _, mastery = self.inference(stacked_zero_q_embed, stacked_read, stacked_counter_embed)
        return tf.reshape(mastery, [self.args.batch_size, self.args.memory_size])

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
        stacked_value_matrix_reshaped =  tf.reshape(stacked_value_matrix, [-1, self.args.memory_size, self.args.value_memory_dim])
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

    def extend_knowledge_growth(self, value_matrix, qa_embed, read, summary, prob, mastery):
        qa_embed = tf.reshape(qa_embed, [self.args.batch_size, -1])

        if self.args.knowledge_growth == 'origin':
            return qa_embed

        elif self.args.knowledge_growth == 'value_matrix':
            target = value_matrix
        elif self.args.knowledge_growth == 'read_content':
            target = read
        elif self.args.knowledge_growth == 'summary':
            target = summary
        elif self.args.knowledge_growth == 'pred_prob':
            target = prob
        elif self.args.knowledge_growth == 'mastery':
            target = mastery

        target = tf.reshape(target, [self.args.batch_size, -1])
        return tf.concat([target, qa_embed], 1)

    def build_model(self):
        # TODO : Naming... terrible case. it should be fixed after paper published
        self.q_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data_seq')
        self.qa_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
        self.target_seq = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='target')

        self.selected_mastery_index = tf.placeholder(tf.int32, name='selected_mastery_index')

        self.q_counter = tf.get_variable('self.q_counter',
                                         [self.args.batch_size, self.args.n_questions+1], dtype=tf.int32,
                                         initializer=tf.zeros_initializer, trainable=False)

        # TODO : move to __init__
        self.memory = self.build_memory()
        self.build_embedding_mtx()

        slice_q_data = tf.split(self.q_data_seq, self.args.seq_len, 1)
        slice_qa_data = tf.split(self.qa_data_seq, self.args.seq_len, 1)
        slice_target = tf.split(self.target_seq, self.args.seq_len, 1)

        prediction = list()
        mastery_list = list()

        prev_total_uniform_prob = self.batch_skill_uniform_inference(self.memory.memory_value, self.q_counter)
        prev_mastery = self.batch_concept_attention_inference_mastery(self.memory.memory_value, self.q_counter)

        mastery_list.append(prev_mastery)

        # loss sum init
        convergence_loss_term = 0
        negative_influence_loss_term = 0

        # for counter indexing
        q_constant = tf.constant(np.arange(self.args.batch_size), dtype=tf.int32)

        # logic
        for i in range(self.args.seq_len):

            # pre-process input
            q = tf.squeeze(slice_q_data[i], 1)
            qa = tf.squeeze(slice_qa_data[i], 1)
            a = tf.cast(tf.greater(qa, tf.constant(self.args.n_questions)), tf.float32)
            target = tf.squeeze(slice_target[i], 1)

            # attention
            q_embed = self.embedding_q(q)
            correlation_weight = self.memory.attention(q_embed)

            # read
            read, summary, prob_logit, prob = self.batch_attention_inference(q, self.memory.memory_value, self.q_counter)
            prediction.append(prob_logit)

            # write
            qa_embed = self.embedding_qa(qa)
            knowledge_growth = \
                self.extend_knowledge_growth(self.memory.memory_value, qa_embed, read, summary, prob, prev_mastery)
            self.memory.memory_value = \
                self.memory.value.write_given_a(self.memory.memory_value, correlation_weight, knowledge_growth, a)

            # update prev variables
            mastery = self.batch_concept_attention_inference_mastery(self.memory.memory_value, self.q_counter)
            mastery_list.append(mastery)
            prev_mastery = mastery

            total_uniform_prob = self.batch_skill_uniform_inference(self.memory.memory_value, self.q_counter)
            total_uniform_prob_diff = total_uniform_prob - prev_total_uniform_prob
            prev_total_uniform_prob = total_uniform_prob

            # calc losses
            # TODO : remove -1 target.
            valid_idx = tf.where(tf.not_equal(target, tf.constant(-1, dtype=tf.float32)))
            correct_idx = tf.where(tf.equal(a, tf.constant(1, tf.float32)))

            # calc negative influence loss
            prob_diff = tf.gather(total_uniform_prob_diff, correct_idx)
            negative_idx = tf.where(tf.less(prob_diff, 0))
            prob_diff_negative = tf.gather_nd(prob_diff, negative_idx)
            negative_influence_loss_term += tf.reduce_mean(tf.square(prob_diff_negative))

            # calc convergence loss
            p = tf.gather(tf.sigmoid(prob_logit) , valid_idx)
            q_concat = tf.stack([q_constant, q], axis=1)
            valid_counter = tf.gather_nd(self.q_counter, q_concat)
            valid_counter = tf.squeeze(tf.gather(valid_counter, valid_idx))
            loss_term = tf.squeeze(tf.square(1-p))
            convergence_loss_term += tf.reduce_mean(tf.cast(valid_counter, tf.float32) * loss_term)

            q_one_hot = tf.one_hot(q, self.args.n_questions+1, dtype=tf.int32)
            self.q_counter += q_one_hot

        self.mastery_level_seq = mastery_list
        # self.prediction_seq = tf.sigmoid(prediction)

        # 'prediction' : seq_len length list of [batch size ,1], make it [batch size, seq_len] tensor
        # tf.stack convert to [batch size, seq_len, 1]
        pred_logits = tf.reshape(tf.stack(prediction, axis=1), [self.args.batch_size, self.args.seq_len])
        self.pred = tf.sigmoid(pred_logits)

        # filtered by selected_mastery_index
        # self.target_seq = self.target_seq[:, self.selected_mastery_index+1:]
        # pred_logits = pred_logits[:, self.selected_mastery_index+1:]

        # Define loss : standard cross entropy loss, need to ignore '-1' label example
        # Make target/label 1-d array
        target_1d = tf.reshape(self.target_seq, [-1])
        pred_logits_1d = tf.reshape(pred_logits, [-1])
        index = tf.where(tf.not_equal(target_1d, tf.constant(-1., dtype=tf.float32)))
        # tf.gather(params, indices) : Gather slices from params according to indices
        filtered_target = tf.gather(target_1d, index)
        filtered_logits = tf.gather(pred_logits_1d, index)

        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
        self.negative_influence_loss = self.args.negative_influence_loss_weight * negative_influence_loss_term
        self.convergence_loss = self.args.convergence_loss_weight * convergence_loss_term

        self.loss = self.cross_entropy_loss + self.negative_influence_loss + self.convergence_loss

        # Optimizer : SGD + MOMENTUM with learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        # self.learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, decay_steps=self.args.anneal_interval*(tf.shape(self.q_data_seq)[0] // self.args.batch_size), decay_rate=0.667, staircase=True)
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

    ####################################################################################################################
    # FOR Reinforcement Learning
    ####################################################################################################################

    def sampling_a_given_q(self, q, value_matrix):
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        # TODO : adapt inference to other inference
        _, _, _, prob = self.inference(q_embed, correlation_weight, value_matrix)

        # TODO : arguemnt check for various algorithms
        prob = tf.clip_by_value(prob, 0.3, 1.0)

        threshold = tf.random_uniform(prob.shape)

        a = tf.cast(tf.less(threshold, prob), tf.int32)
        qa = q + tf.multiply(a, self.args.n_questions)[0]

        return qa

    def build_step_graph(self):
        self.q = tf.placeholder(tf.int32, [1, 1], name='step_q')
        self.a = tf.placeholder(tf.int32, [1, 1], name='step_a')
        self.value_matrix = tf.placeholder(tf.float32, [self.args.memory_size, self.args.value_memory_dim],
                                           name='step_value_matrix')

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

        self.qa = q + tf.multiply(a, self.args.n_questions)
        '''
        # TODO : use cond for sampling 
        self.qa = tf.cond(tf.squeeze(a) < 0,
                          lambda: self.sampling_a_given_q(q, stacked_value_matrix),
                          lambda: q + tf.multiply(a, self.args.n_questions) )
        '''

        a = (self.qa-1) // self.args.n_questions
        qa_embed = self.embedding_qa(self.qa)

        # before Step
        # read, summary, prob_logit, prob = self.inference(q_embed, correlation_weight, stacked_value_matrix, self.counter)
        read, summary, prob_logit, prob = self.batch_attention_inference(q, stacked_value_matrix, self.counter)
        # prev_read_content, prev_summary, prev_pred_logits, prev_pred_prob = self.inference(q_embed, correlation_weight, stacked_value_matrix, self.counter)
        mastery = self.batch_concept_attention_inference_mastery(stacked_value_matrix, self.counter)

        # step
        knowledge_growth = self.extend_knowledge_growth(stacked_value_matrix, qa_embed, read, summary, prob, mastery)
        # TODO : refactor sampling_a_given_q to return a only for below function call
        self.stepped_value_matrix = tf.squeeze(self.memory.value.write_given_a(stacked_value_matrix, correlation_weight, knowledge_growth, a), axis=0)

        # build other useful graphs
        # TODO : remove using_counter graph because it is member variable of model
        self.total_pred_probs = self.skill_attention_inference(self.value_matrix, self.counter)
        self.concept_mastery = self.concept_attention_inference_mastery(self.value_matrix, self.counter)

