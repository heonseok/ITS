import numpy as np
import os
import tensorflow as tf
import operations


# This class defines Memory architecture in DKVMN
class DKVMN_Memory():
    def __init__(self, memory_size, memory_state_dim, args, name):
        tf.set_random_seed(224)
        self.name = name
        #print('%s initialized' % self.name)
        # Memory size : N
        self.memory_size = memory_size
        # Memory state dim : D_V or D_K
        self.memory_state_dim = memory_state_dim
       
        self.args = args
        '''
            Key matrix or Value matrix
            Key matrix is used for calculating correlation weight(attention weight)
        '''
            

    def cor_weight(self, embed, key_matrix):
        '''
            embed : [batch size, memory state dim(d_k)]
            Key_matrix : [memory size * memory state dim(d_k)]
            Correlation weight : w(i) = k * Key matrix(i)
            => batch size * memory size
        '''    
           # embedding_result : [batch size, memory size], each row contains each concept correlation weight for 1 question
        embedding_result = tf.matmul(embed, tf.transpose(key_matrix))
        correlation_weight = tf.nn.softmax(embedding_result)
        #print('Correlation weight shape : %s' % (correlation_weight.get_shape()))
        return correlation_weight

    
    def read_for_mastery(self, value_matrix, correlation_weight):
        '''
            Value matrix : [batch size, memory size, memory size, memory state dim]
            Correlation weight : [batch size, memory size, memory size], each element represents each concept embedding for 1 question
        '''
        # Reshaping
        # [batch size * memory size, memory state dim(d_v)]
        vmtx_reshaped = tf.reshape(value_matrix, [-1, self.memory_size, self.memory_state_dim])
        # [batch size * memory size, 1]
        cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])        
        #print('Transformed shape : %s, %s' %(vmtx_reshaped.get_shape(), cw_reshaped.get_shape()))
        # Read content, will be [batch size * memory size, memory state dim] and reshape it to [batch size, memory size, memory state dim]
        rc = tf.multiply(vmtx_reshaped, cw_reshaped)
        read_content = tf.reshape(rc, [-1, self.memory_size, self.memory_size, self.memory_state_dim])
        # Summation through memory size axis, make it [batch size, memory state dim(d_v)]
        #read_content = tf.log(tf.reduce_sum(read_content, axis=1, keep_dims=False))
        read_content = tf.reduce_sum(read_content, axis=2, keep_dims=False)
        #print('Read content shape : %s' % (read_content.get_shape()))
        return read_content

    # Getting read content
    def read(self, value_matrix, correlation_weight):
        '''
            Value matrix : [batch size ,memory size ,memory state dim]
            Correlation weight : [batch size ,memory size], each element represents each concept embedding for 1 question
        '''
        # Reshaping
        # [batch size * memory size, memory state dim(d_v)]
        vmtx_reshaped = tf.reshape(value_matrix, [-1, self.memory_state_dim])
        # [batch size * memory size, 1]
        cw_reshaped = tf.reshape(correlation_weight, [-1,1])        
        #print('Transformed shape : %s, %s' %(vmtx_reshaped.get_shape(), cw_reshaped.get_shape()))
        # Read content, will be [batch size * memory size, memory state dim] and reshape it to [batch size, memory size, memory state dim]
        rc = tf.multiply(vmtx_reshaped, cw_reshaped)
        read_content = tf.reshape(rc, [-1,self.memory_size,self.memory_state_dim])
        # Summation through memory size axis, make it [batch size, memory state dim(d_v)]
        #read_content = tf.log(tf.reduce_sum(read_content, axis=1, keep_dims=False))
        read_content = tf.reduce_sum(read_content, axis=1, keep_dims=False)
        #print('Read content shape : %s' % (read_content.get_shape()))
        return read_content


    def activate_add_signal(self, add_vector):
        if self.args.add_signal_activation == 'tanh':
            return tf.tanh(add_vector)
        elif self.args.add_signal_activation == 'sigmoid':
            return tf.sigmoid(add_vector)
        elif self.args.add_signal_activation == 'relu':
            return tf.nn.relu(add_vector)

    def activate_erase_signal(self, erase_vector):
        if self.args.erase_signal_activation == 'tanh':
            return tf.tanh(erase_vector)
        elif self.args.erase_signal_activation == 'sigmoid':
            return tf.sigmoid(erase_vector)
        elif self.args.erase_signal_activation == 'relu':
            return tf.nn.relu(erase_vector)

    def add(self, value_matrix, correlation_weight, knowledge_growth):
    #def add(self, value_matrix, correlation_weight, knowledge_growth, reuse=False):
        add_vector = operations.linear(knowledge_growth, self.memory_state_dim, name=self.name+'/Add_Vector')
        add_signal = self.activate_add_signal(add_vector)
        cw_reshaped = tf.reshape(correlation_weight, [-1,self.memory_size,1])
        add_reshaped = tf.reshape(add_signal, [-1, 1, self.memory_state_dim])
        add_mul = tf.multiply(add_reshaped, cw_reshaped)
  
        return add_mul

    def erase(self, value_matrix, correlation_weight, knowledge_growth):
        erase_vector = operations.linear(knowledge_growth, self.memory_state_dim, name=self.name+'/Erase_Vector')
        erase_signal = self.activate_erase_signal(erase_vector)
        erase_reshaped = tf.reshape(erase_signal, [-1,1,self.memory_state_dim])
        cw_reshaped = tf.reshape(correlation_weight, [-1,self.memory_size,1])
        erase_mul = tf.multiply(erase_reshaped, cw_reshaped)
        erase = tf.multiply(value_matrix,1 - erase_mul)
      
        return erase

    def write_given_a(self, value_matrix, correlation_weight, knowledge_growth, a):
        '''
            Value matrix : [batch size, memory size, memory state dim(d_k)]
            Correlation weight : [batch size, memory size]
        '''
        add_mul = self.add(value_matrix, correlation_weight, knowledge_growth)
        erase = self.erase(value_matrix, correlation_weight, knowledge_growth)
        
        a_reshaped = tf.reshape(tf.cast(a, tf.float32), [-1, 1, 1])
        ones = tf.ones(tf.shape(a_reshaped))
        
        # TODO : split add and erase to two argument 
        if self.args.write_type == 'add_off_erase_off':
            new_memory = tf.multiply(a_reshaped, add_mul) + tf.multiply(ones-a_reshaped, erase)
        elif self.args.write_type == 'add_on_erase_on':
            new_memory = add_mul + erase
        elif self.args.write_type == 'add_on_erase_off':
            new_memory = add_mul + tf.multiply(ones-a_reshaped, erase)
        elif self.args.write_type == 'add_off_erase_on':
            new_memory = tf.multiply(a_reshaped, add_mul) + erase

        # [batch size, memory size, memory value staet dim]
        return new_memory



# This class construct key matrix and value matrix
class DKVMN():
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key, init_memory_value, args, name='DKVMN'):
        print('Initializing memory..')
        tf.set_random_seed(224)
        self.name = name
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        
        self.key = DKVMN_Memory(self.memory_size, self.memory_key_state_dim, args, name=self.name+'_key_matrix')
        self.value = DKVMN_Memory(self.memory_size, self.memory_value_state_dim, args, name=self.name+'_value_matrix')

        self.memory_key = init_memory_key
        self.memory_value = init_memory_value

    def attention(self, q_embed):
        correlation_weight = self.key.cor_weight(embed=q_embed, key_matrix=self.memory_key)
        return correlation_weight
