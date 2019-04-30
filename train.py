# coding: utf8
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
from model import TextClassification
from reader import DataReader

def train(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    tf.reset_default_graph()
    model = TextClassification(vocab_size=args.vocab_size, 
            encoder_type=args.encoder_type, max_seq_len=args.max_seq_len)
    # optimizer
    train_step = tf.contrib.opt.LazyAdamOptimizer(learning_rate=args.learning_rate).minimize(model.loss)
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar("train_loss", model.loss)
    init = tf.group(tf.global_variables_initializer(), 
            tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        # feeding embedding
        _writer = tf.summary.FileWriter(args.logdir, sess.graph)

        # summary
        summary_op = tf.summary.merge([loss_summary])
        step = 0
        for epoch in range(args.epochs):
            train_reader = DataReader(args.vocab_path, args.train_data_path, 
                    args.vocab_size, args.batch_size, args.max_seq_len)
            for train_batch in train_reader.batch_generator():
                text, label = train_batch
                _, _loss, _summary, _logits = sess.run([train_step, model.loss, summary_op, model.logits],
                        feed_dict={model.label_in: label, model.text_in: text})
                _writer.add_summary(_summary, step)
                step += 1


                # test
                summary = tf.Summary()
                if step % args.eval_interval == 0:
                    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(label, 1),
                            predictions=tf.argmax(_logits, 1))
                    sess.run(tf.local_variables_initializer())
                    _, _acc = sess.run([acc, acc_op])
                    summary.value.add(tag="train_accuracy", simple_value=_acc)
                    print("Epochs: {}, Step: {}, Train Loss: {}, Acc: {}".format(epoch, step, _loss, _acc))

                    test_reader = DataReader(args.vocab_path, args.test_data_path, 
                            args.vocab_size, args.batch_size, args.max_seq_len)
                    sum_loss = 0.0
                    sum_acc = 0.0
                    iters = 0
                    for test_batch in test_reader.batch_generator():
                        text, label = test_batch
                        _loss, _logits = sess.run([model.loss, model.logits],
                                feed_dict={model.label_in: label, model.text_in: text})
                        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(label, 1),
                                predictions=tf.argmax(_logits, 1))
                        sess.run(tf.local_variables_initializer())
                        _, _acc = sess.run([acc, acc_op])
                        sum_acc += _acc
                        sum_loss += _loss
                        iters += 1
                    avg_loss = sum_loss / iters
                    avg_acc = sum_acc / iters
                    summary.value.add(tag="test_accuracy", simple_value=avg_acc)
                    summary.value.add(tag="test_loss", simple_value=avg_loss)
                    _writer.add_summary(summary, step)
                    print("Epochs: {}, Step: {}, Test Loss: {}, Acc: {}".format(epoch, step, avg_loss, avg_acc))
                if step % args.save_interval == 0:
                    save_path = saver.save(sess, "{}/birnn.lm.ckpt".format(args.model_path), global_step=step)
                    print("Model save to path: {}/birnn.lm.ckpt".format(args.model_path))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=48)
    parser.add_argument("--encoder_type", type=str, default="BOW")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
    parser.add_argument("--test_data_path", type=str, default="./data/test.txt")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default="100")
    parser.add_argument("--eval_interval", type=int, default="50")
    args = parser.parse_args()
    train(args)
