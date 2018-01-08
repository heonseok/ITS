import tensorflow as tf
#from StringIO import StringIO
import matplotlib.pyplot as plt
import numpy as np

class Logger(object):
    
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        #counts, bin_edges = np.histogram(values, bins=bins)
        summary = tf.summary.histogram('tag', values)
        self.writer.add_summary(summary, step)


