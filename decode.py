# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import json
import sys
import os
from model import Seq2SeqModel
from utils import get_infer_iterator, create_vocab_tables, get_iterator
import codecs
import argparse
import time

parser = argparse.ArgumentParser()
# Decoding parameters
parser.add_argument('--beam_width', type=int, default=-1, help='Beam width used in beamsearch')
parser.add_argument('--decode_batch_size', type=int, default=50, help='Batch size used for decoding')
parser.add_argument('--max_decode_step', type=int, default=50, help='Maximum time step limit to decode')
parser.add_argument('--model_path', type=str, default='', help='Path to a specific model checkpoint.')
parser.add_argument('--decode_input', type=str, default='', help='Decoding input path')
parser.add_argument('--decode_output', type=str, default='', help='Decoding output path')
parser.add_argument('--decode_ignore_unk', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help='Ignore unk when decoding')
# Runtime parameters
parser.add_argument('--allow_soft_placement', type=lambda s: s.lower() in ['true', 't', 'yes', '1'] , default=True, help='Allow device soft placement')
parser.add_argument('--log_device_placement', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],  default=False, help='Log placement of ops on devices')
parser.add_argument('--gpu_ids', type=str, default="0", help="Id of gpu for training")

UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


def seq2words(seq, trg_vocab):
    """
    seq2words
    """
    words = []
    for w in seq:
        if w == EOS_ID:
            break
        if w in trg_vocab:
            words.append(trg_vocab[w])
        else:
            words.append(UNK)
    return ' '.join(words)


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def load_config(FLAGS):
    config = unicode_to_utf8(
        json.load(open('%s.json' % FLAGS.model_path, 'rb')))
    for key, value in vars(FLAGS).items():
        config[key] = value
        print key, value
    config['decode_ignore_unk'] = FLAGS.decode_ignore_unk
    return config


def load_model(session, model, model_path):
    if tf.train.checkpoint_exists(model_path):
        print 'Reloading model parameters..'
        model.restore(session, model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(model_path))
    return model


def load_word2idx(source_vocabulary, target_vocabulary):
    src_word2idx = []
    trg_word2idx = []
    with codecs.open(source_vocabulary, 'r', 'utf-8') as f:
        for word in f:
            word = word.strip()
            src_word2idx.append(word)
    with codecs.open(target_vocabulary, 'r', 'utf-8') as f:
        for word in f:
            word = word.strip()
            trg_word2idx.append(word)
    src_word2idx = {src_word2idx[idx]:idx for idx in range(len(src_word2idx))}
    trg_word2idx = {trg_word2idx[idx]:idx for idx in range(len(trg_word2idx))}
    return src_word2idx, trg_word2idx



def decode(FLAGS):
    """
    Decode ...
    """
    config = load_config(FLAGS)
    src_vocab, trg_vocab = create_vocab_tables(config["source_vocabulary"], config["target_vocabulary"], False)
    print "Loading target vocab... "
    with codecs.open(config["target_vocabulary"], "r", "utf-8") as f:
        trg_reverse_vocab = {}
        idx = 0
        for line in f:
            word = line.strip()
            trg_reverse_vocab[idx] = word
            idx += 1
    source_test_dataset = tf.data.TextLineDataset(FLAGS.decode_input)
    test_iterator = get_infer_iterator(source_test_dataset, src_vocab,
        batch_size=FLAGS.decode_batch_size, eos="<EOS>",
        src_max_len=FLAGS.max_decode_step)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.tables_initializer())
        # Init the iterator for data streaming
        sess.run(test_iterator.initializer)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            model = Seq2SeqModel(config, "decode")
            pred_decode_op = model.build_model(test_iterator)
            load_model(sess, model, FLAGS.model_path)
            print "Decoding..."
            fout = codecs.open(FLAGS.decode_output, "w", "utf-8")
            idx = 0
            while True:
                try:
                    predict_ids = sess.run(pred_decode_op)

                    for seq in predict_ids:
                        decode_tmp_result = []
                        for beam_id in range(FLAGS.beam_width):
                            decode_tmp_result.append(seq2words(seq[:, beam_id], trg_reverse_vocab))
                            idx += 1
                        fout.write(" |<->| ".join(decode_tmp_result) + "\n")

                except tf.errors.OutOfRangeError:
                    print "Finished decoding on test dataset"
                    break
            fout.close()
            print '{}th line decoded'.format(idx)

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_ids
    decode(FLAGS)
