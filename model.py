# coding: utf8
import sys
import os
import numpy as np
import tensorflow as tf

class BOWEncoder(object):
    def __init__(self, inputs=None, embedding=None, hidden_size=None, name=None):
        vocab_size, embedding_size = embedding.shape
        with tf.variable_scope("embedding_%s" % name, reuse=tf.AUTO_REUSE):
            self.inputs_emb = tf.nn.embedding_lookup(embedding, inputs)

        with tf.variable_scope("pooling_%s" % name, reuse=tf.AUTO_REUSE):
            self.seq_pooling_emb = tf.nn.softsign(tf.reduce_sum(self.inputs_emb, axis=1))

        with tf.variable_scope("fc_%s" % name, reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(name="fc_w", shape=[embedding_size, hidden_size],
                    initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            self.biases = tf.get_variable(name="fc_b", shape=[hidden_size],
                    initializer=tf.constant_initializer(0.001))
            self.fc_out = tf.add(tf.matmul(self.seq_pooling_emb, self.weights), self.biases)


class CNNEncoder(object):
    def __init__(self, inputs=None, embedding=None, embedding_size=128, 
            filter_sizes=[3, 4, 5], num_filters=50, max_seq_len=48):
        self.inputs_emb = tf.nn.embedding_lookup(embedding, inputs)
        self.inputs_emb_expanded = tf.expand_dims(self.inputs_emb, -1)
        self.seq_len = inputs.shape[1]

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.Variable(tf.constant(0.001, shape=[num_filters]))

                conv = tf.nn.conv2d(self.inputs_emb_expanded, 
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(
                        h, 
                        ksize=[1, max_seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID')
                #pooled = tf.reduce_max(
                #        h, 
                #        axis=2, 
                #        keep_dims=True)
                pooled_outputs.append(pooled) 

        self.num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])


class RNNEncoder(object):
    """ rnn lstm
    """
    def __init__(self, inputs=None, embedding=None, 
            hidden_size=None, name=None, num_layers=1, keep_prob=0.5):
        vocab_size, embedding_size = embedding.shape
        self._keop_prob = keep_prob
        self.inputs_emb = tf.nn.embedding_lookup(embedding, inputs)
        with tf.variable_scope("bi-rnn_%s" % name, reuse=tf.AUTO_REUSE):
            def make_cell():
                cell = tf.nn.rnn_cell.GRUCell(hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._keop_prob)
                return cell
            with tf.variable_scope("fw", reuse=tf.AUTO_REUSE):
                fw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(num_layers)])
            with tf.variable_scope("bw", reuse=tf.AUTO_REUSE):
                bw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(num_layers)])
            with tf.variable_scope("fw-bw", reuse=tf.AUTO_REUSE):
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
                        self.inputs_emb, dtype=tf.float32)
            fw_outputs = rnn_outputs[0]
            bw_outputs = rnn_outputs[1]
            merged_output = tf.concat([fw_outputs, bw_outputs], axis=2)
            merged_output = tf.transpose(merged_output, [1, 0, 2])
            self.rnn_out = merged_output[-1]


class TextClassification(object):
    """ text classification
    """
    def __init__(self, vocab_size=None, embedding_size=128,
            encoder_type="BOW", hidden_size=128, num_class=2, max_seq_len=48):
        # inputs
        self.label_in = tf.placeholder(tf.int32, [None, None], name="label")
        self.text_in = tf.placeholder(tf.int32, [None, None], name="text")

        # embedding init
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="emb_mat")

        if encoder_type == "BOW":
            with tf.variable_scope("encoder"):
                self.text_encoder = BOWEncoder(inputs=self.text_in,
                        embedding=self.embedding, hidden_size=hidden_size, name="text")
                self.text_vec = tf.nn.softsign(self.text_encoder.fc_out)
                self.logits = self.fc_layer(self.text_vec, name="fc", shape=[hidden_size, num_class],
                        activation_function=None)
        elif encoder_type == "RNN":
            with tf.variable_scope("encoder"):
                self.text_encoder = RNNEncoder(inputs=self.text_in,
                        embedding=self.embedding, hidden_size=hidden_size, name="text")
                self.text_vec = tf.nn.softsign(self.text_encoder.rnn_out)
                self.logits = self.fc_layer(self.text_vec, name="fc", shape=[2*hidden_size, num_class],
                        activation_function=None)
        elif encoder_type == "CNN":
            with tf.variable_scope("encoder"):
                self.text_encoder = CNNEncoder(inputs=self.text_in,
                        embedding=self.embedding, embedding_size=embedding_size, max_seq_len=max_seq_len)
                self.text_vec = tf.nn.softsign(self.text_encoder.h_pool_flat)
                self.logits = self.fc_layer(self.text_vec, name="fc", 
                        shape=[self.text_encoder.num_filters_total, num_class], activation_function=None)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                        labels=self.label_in))

    def fc_layer(self, inputs, shape, name, activation_function=None):
        """ fc layer
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(name="%s_w" % name, shape=shape, 
                    initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            biases = tf.get_variable(name="%s_b" % name, shape=[shape[1]], 
                    initializer=tf.constant_initializer(0.001))
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            return outputs

