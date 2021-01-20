# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import math
import time
import json
import datetime
from collections import OrderedDict
import tensorflow as tf
from model import Seq2SeqModel
import codecs
from utils import get_iterator, create_vocab_tables
import argparse
import numpy as np
from logger import Logger

parser = argparse.ArgumentParser()
# Data loading parameters
parser.add_argument("--source_vocabulary", type=str, default='', help="Path to source vocabulary")
parser.add_argument("--target_vocabulary", type=str, default='', help="Path to target vocabulary")
parser.add_argument("--source_train_data", type=str, default='', help="Path to source training data")
parser.add_argument("--target_train_data", type=str, default='', help="Path to target training data")
parser.add_argument("--source_valid_data", type=str, default='', help="Path to source validation data")
parser.add_argument("--target_valid_data", type=str, default='', help="Path to source validation data")

# Network parameters
parser.add_argument('--cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')
parser.add_argument('--attention_type', type=str, default='bahdanau', help='Attention mechanism: (bahdanau, luong), default: bahdanau')
parser.add_argument('--hidden_units', type=int, default=100, help='Number of hidden units in each layer')
parser.add_argument('--depth', type=int, default=2, help='Number of layers in each encoder and decoder')
parser.add_argument('--use_residual', type=lambda x: (str(x).lower() == 'false'), default=True, help='Use residual connection between layers')
parser.add_argument('--attn_input_feeding', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, help='Use input feeding method in attentional decoder')
parser.add_argument('--use_dropout', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help='Use dropout in each rnn cell')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout probability for input/output/state units (0.0: no dropout)')
parser.add_argument('--encoder_type', type=str, default="emb", help='Bi, Uni or Just Emb')
parser.add_argument('--decoder_output_arch', type=str, default="Local", help="Local, Hybrid, Emotion (No global)")

# Training parameters
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--decay_rate', type=float, default=0.98, help='Decay rate')
parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='Clip gradients to this norm')
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument('--max_epochs', type=int, default=10, help='Maximum # of training epochs')
parser.add_argument("--max_load_batches", type=int, default=50, help="Maximum # of batches to load at one time")
parser.add_argument("--max_seq_length", type=int, default=50, help="Maximum sequence length")
parser.add_argument("--display_freq", type=int, default=50, help="Display training status every this iteration")
parser.add_argument("--save_freq", type=int, default=1000, help="Save model checkpoint every this iteration")
parser.add_argument("--valid_freq", type=int, default=1000, help="Evaluate model every this iteration: valid_data needed")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training: (adadelta, adam, rmsprop)")
parser.add_argument("--model_dir", type=str, default="", help="Path to save model checkpoints")
parser.add_argument("--model_name", type=str, default="seq2seq.ckpt", help="File name used for model checkpoints")
# Runtime parameters
parser.add_argument("--allow_soft_placement", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Allow device soft placement")
parser.add_argument("--log_device_placement", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, help="Log placement of ops on devices")
parser.add_argument("--gpu_ids", type=str, default="0", help="Id of gpu for running")


def create_model(session, saver, config, FLAGS):
    """
    Create the model with the config
    """
    model_status = ""

    ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt):
        print 'Reloading model parameters..'
        saver.restore(session, ckpt)
        model_status = "reload"
    else:
        if not os.path.exists(FLAGS.model_dir):
            date_now = datetime.datetime.fromtimestamp(
                int(time.time())).strftime('%Y%m%d%H%M')
            FLAGS.model_dir = "model_" + date_now
            if not os.path.exists(FLAGS.model_dir):
                os.makedirs(FLAGS.model_dir)
        print 'Created new model parameters..'

        with open(os.path.join(FLAGS.model_dir, "parameters.txt"), "w") as fopen:
            for k, v in config.items():
                if not isinstance(v, str):
                    v = str(v)
                fopen.write("%s %s\n" % (k, v))
                print "%s %s" %(k, v)

        with open(os.path.join(FLAGS.model_dir, "parameters.txt"), "aw") as f:
            total_parameters = 0
            for variable in tf.trainable_variables():
                f.write(str(variable) + " " + variable.device + '\n')
                shape = variable.get_shape()
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print 'number of parameters in the network', total_parameters

            for n in tf.get_default_graph().as_graph_def().node:
                f.write(n.name+"\n")
        model_status = "create"

    return model_status


def load_vocab(FLAGS):
    """
    Load vocab if needed
    """
    # get sizes of source vocab and target vocab
    encoder_symbols, decoder_symbols = {}, {}

    with codecs.open(FLAGS.source_vocabulary, "r", "utf-8") as f:
        for line in f:
            word = line.strip()
            encoder_symbols[word] = None

    with codecs.open(FLAGS.target_vocabulary, "r", "utf-8") as f:
        for line in f:
            word = line.strip()
            decoder_symbols[word] = None
    print "src vocab size ", len(encoder_symbols)
    print "trg vocab size ", len(decoder_symbols)

    return encoder_symbols, decoder_symbols


def average_gradients(tower_grads, max_gradient_norm):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    vars_list = []
    grads_list = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            tmp_g = tf.expand_dims(g, 0)
            grads.append(tmp_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        vars_list.append(v)
        grads_list.append(grad)
    clip_gradients, _ = tf.clip_by_global_norm(grads_list, max_gradient_norm)
    average_grads = zip(clip_gradients, vars_list)
    return average_grads


def init_optimizer(FLAGS, learning_rate):
    # optimizer
    if FLAGS.optimizer.lower() == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer.lower() == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer.lower() == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return opt

def init_global_step(FLAGS):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    global_epoch_step_op = tf.assign(global_epoch_step, global_epoch_step + 1)
    return global_step, global_epoch_step, global_epoch_step_op

def init_global_step_and_learning_rate(FLAGS, global_epoch_step):
    learning_rate_variable = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False, name='learning_rate')
    learning_rate_decay_op = tf.assign(learning_rate_variable, FLAGS.learning_rate * (FLAGS.decay_rate ** tf.cast(global_epoch_step, tf.float32)))
    return learning_rate_variable, learning_rate_decay_op


def train(FLAGS):
    """
    Training Process
    """
    # get num symbols of encoder and decoder for creating embedding
    encoder_symbols, decoder_symbols = load_vocab(FLAGS)
    FLAGS.num_encoder_symbols = len(encoder_symbols)
    FLAGS.num_decoder_symbols = len(decoder_symbols)
    print "num of encoder symbols is: ", FLAGS.num_encoder_symbols
    print "num of decoder symbols is: ", FLAGS.num_decoder_symbols

    # create tables
    src_vocab, trg_vocab = create_vocab_tables(FLAGS.source_vocabulary, FLAGS.target_vocabulary, False)

    # train data iterator
    source_train_dataset = tf.data.TextLineDataset(FLAGS.source_train_data)
    target_train_dataset = tf.data.TextLineDataset(FLAGS.target_train_data)
    num_gpus = len(FLAGS.gpu_ids.split(","))
    train_iterator = get_iterator(source_train_dataset, target_train_dataset, src_vocab,
                                  trg_vocab, batch_size=FLAGS.batch_size, sos="<SOS>",
                                  eos="<EOS>", random_seed=255, num_buckets=5,
                                  src_max_len=50, tgt_max_len=50, 
                                  num_shards=num_gpus, output_buffer_size=FLAGS.max_load_batches)

    # valid data iterator
    if len(FLAGS.source_valid_data) > 0 and len(FLAGS.target_valid_data) > 0:
        source_valid_dataset = tf.data.TextLineDataset(FLAGS.source_valid_data)
        target_valid_dataset = tf.data.TextLineDataset(FLAGS.target_valid_data)
        valid_iterator = get_iterator(source_valid_dataset, target_valid_dataset, src_vocab,
                                      trg_vocab, batch_size=FLAGS.batch_size, sos="<SOS>",
                                      eos="<EOS>", random_seed=255, num_buckets=5,
                                      src_max_len=50, tgt_max_len=50, output_buffer_size=FLAGS.max_load_batches)
    else:
        valid_iterator = None


    # Get global step
    global_step, global_epoch_step, global_epoch_step_op = init_global_step(FLAGS)
    # Get learning rate and decay rate
    learning_rate_variable, learning_rate_decay_op = init_global_step_and_learning_rate(FLAGS, global_epoch_step)
    # Get optimizer
    optimizer = init_optimizer(FLAGS, learning_rate_variable)
    
    config = OrderedDict(sorted(vars(FLAGS).items()))
    model = Seq2SeqModel(config, 'train')

    # Parallel train
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        """
        Assign model to different gpus
        """
        tower_losess = []
        tower_grads = []
        for i in range(num_gpus):
            name = "GPU_%d" %i
            with tf.device("/gpu:%d"%i), tf.name_scope(name):
                loss = model.build_model(train_iterator)
                with tf.name_scope("compute_grads"):
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                tower_losess.append(loss)

            if i == 0 and valid_iterator:
                valid_loss_op = model.build_model(valid_iterator)

    # ready to save
    saver = tf.train.Saver(max_to_keep=5)
     

    with tf.name_scope("apply_gradients"), tf.device("/gpu:0"):
        """
        Collect grads from each gpus and updates (norm clip)
        """
        gradients = average_gradients(tower_grads, FLAGS.max_gradient_norm)
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        average_loss = tf.reduce_mean(tower_losess)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                           log_device_placement=FLAGS.log_device_placement,
                                           gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Init tables for source vocab and target vocab
        sess.run(tf.tables_initializer())

        # Init the iterator for data streaming
        sess.run(train_iterator.initializer)
        if valid_iterator:
           sess.run(valid_iterator.initializer)


        sess.run(tf.global_variables_initializer())
        model_status = create_model(sess, saver, config, FLAGS)
        # logger
        logger = Logger(FLAGS.model_dir + '/log')
        # Before actual training, init global variables
        if model_status == "reload":
            print "loading existing model ...", FLAGS.model_dir
        elif model_status == "create":
            print "create a new model", FLAGS.model_dir
        else:
            raise Exception("Invalid intent")

        # Actual Train Process
        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()
        while global_epoch_step.eval() < FLAGS.max_epochs:
            """
            Runing steps
            """
            try:

                step_loss, _ = sess.run([average_loss, apply_gradient_op])
                loss += float(step_loss) / FLAGS.display_freq
                sents_seen += float(FLAGS.batch_size) # batch_size

                if global_step.eval() % FLAGS.display_freq == 0:
                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    time_elapsed = time.time() - start_time
                    sents_per_sec = sents_seen / time_elapsed
                    step_time = time_elapsed
                    print 'Epoch ', global_epoch_step.eval(), 'Step ', global_step.eval(), \
                          'Perplexity {0:.2f}'.format(avg_perplexity), "Loss {0:.2f}".format(loss), \
                           'Step-time ', '{0:.2f}s'.format(step_time), \
                          '{0:.2f} sents/s'.format(sents_per_sec)
                    logger.scalar_summary('Training loss', step_loss, global_step.eval())
                    logger.scalar_summary('Training Perplexity', avg_perplexity, global_step.eval())
                    logger.scalar_summary('Step time', step_time, global_step.eval())
                    loss = 0.0
                    start_time = time.time()
                    sents_seen = 0

                if valid_iterator and global_step.eval() % FLAGS.valid_freq == 0:
                    print 'Validation step'
                    valid_loss = 0.0
                    valid_sents_seen = 0
                    for _ in range(500):
                        try:
                            step_loss = sess.run(valid_loss_op)
                        except tf.errors.OutOfRangeError:
                            print "Finished going through the valid dataset"
                            ## Finished going through the valid dataset.
                            # Re-init valid iterator
                            sess.run(valid_iterator.initializer)
                            continue
                        batch_size = FLAGS.batch_size
                        valid_loss += step_loss * batch_size
                        valid_sents_seen += batch_size
                    valid_loss = valid_loss / valid_sents_seen
                    print '{} valid samples seen'.format(valid_sents_seen)
                    print "Valid loss: {0:.2f}".format(valid_loss)
                    print 'Valid perplexity: {0:.2f}'.format(math.exp(valid_loss))
                    logger.scalar_summary('Validation loss', valid_loss, global_step.eval())
                    logger.scalar_summary('Valid perplexity', math.exp(valid_loss), global_step.eval())

                if global_step.eval() % FLAGS.save_freq == 0:
                    print 'Saving the model..'
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, global_step.eval()), 'wb'),
                              indent=2)

            except tf.errors.OutOfRangeError:
                print "Finished going through the training dataset and going to the next epoch ", global_epoch_step_op.eval()
                # # Finished going through the training dataset.  Go to next epoch.
                # Re-init train iterator
                sess.run(train_iterator.initializer)
                print "learning rate updates", learning_rate_decay_op.eval()


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_ids
    train(FLAGS)


