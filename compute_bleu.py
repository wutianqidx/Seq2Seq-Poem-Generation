# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
import sys
import os
from model import Seq2SeqModel
from utils import get_infer_iterator, create_vocab_tables
from decode import seq2words, load_config, load_model, load_word2idx, rerank_by_loss
import codecs
import argparse
import nltk
import time

parser = argparse.ArgumentParser()
# Decoding parameters
parser.add_argument('--beam_width', type=int, default=-1, help='Beam width used in beamsearch')
parser.add_argument('--decode_batch_size', type=int, default=50, help='Batch size used for decoding')
parser.add_argument('--max_decode_step', type=int, default=50, help='Maximum time step limit to decode')
parser.add_argument('--model_path', type=str, default='', help='Path to a specific model checkpoint.')
parser.add_argument('--valid_src_path', type=str, default='data/weibo4m.src.valid', help='Path to validation src')
parser.add_argument('--valid_trg_path', type=str, default='data/weibo4m.trg.valid', help='Path to validation trg')
parser.add_argument('--decode_ignore_unk', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True,
                    help='Ignore unk when decoding')
parser.add_argument('--rerank', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, help='Rerank decode results by the loss')
# Runtime parameters
parser.add_argument('--allow_soft_placement', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True,
                    help='Allow device soft placement')
parser.add_argument('--log_device_placement', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                    help='Log placement of ops on devices')
parser.add_argument('--gpu_ids', type=str, default="0", help="Id of gpu for training")

UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


def load_post_references(valid_src_path, valid_trg_path):
    post_list = []
    post_resp_dict = {}
    posts = [x.strip() for x in codecs.open(valid_src_path, 'r', 'utf-8').readlines()]
    responses = [x.strip() for x in codecs.open(valid_trg_path, 'r', 'utf-8').readlines()]
    valid_data = zip(posts, responses)
    valid_data = sorted(valid_data, key=lambda x:x[0])
    for post, response in valid_data:
        if post not in post_resp_dict:
            post_resp_dict[post] = []
            post_list.append(post)
        post_resp_dict[post].append(response)
    references = [post_resp_dict[post] for post in post_list]
    references = map(lambda x:map(lambda y:y.split(), x), references)
    return post_list, references


def compute_bleu(FLAGS):
    """
  Decode ...
  """
    config = load_config(FLAGS)
    src_vocab, trg_vocab = create_vocab_tables(config["source_vocabulary"], config["target_vocabulary"], False)
    print "Loading target vocab ... "
    with codecs.open(config["target_vocabulary"], "r", "utf-8") as f:
        trg_reverse_vocab = {}
        idx = 0
        for line in f:
            word = line.strip()
            trg_reverse_vocab[idx] = word
            idx += 1
    querys, list_of_references = load_post_references(FLAGS.valid_src_path, FLAGS.valid_trg_path)
    assert len(querys) == len(list_of_references)
    source_test_dataset = tf.data.Dataset.from_tensor_slices(querys)
    print("Has {} querys needed to be decoded".format(len(querys)))
    test_iterator = get_infer_iterator(source_test_dataset, src_vocab,
                                       batch_size=FLAGS.decode_batch_size, eos="<EOS>",
                                       src_max_len=FLAGS.max_decode_step)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.tables_initializer())
        # Init the iterator for data streaming
        sess.run(test_iterator.initializer)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            model = Seq2SeqModel(config, "decode")
            pred_decode_op = model.build_model(test_iterator)
            load_model(sess, model, FLAGS.model_path)
            print "Decoding ..."
            idx = 0
            querys_decode_list = [(query, []) for query in querys]
            while True:
                try:
                    start_time = time.time()
                    predict_ids = sess.run(pred_decode_op)
                    print("decoded {} querys in {} seconds".format(len(predict_ids), time.time() - start_time))
                    if FLAGS.beam_width > 1:
                        for seq in predict_ids:
                            for k in range(FLAGS.beam_width):
                                decode_result = seq2words(seq[:, k], trg_reverse_vocab)
                                querys_decode_list[idx][1].append(decode_result)
                            idx += 1

                except tf.errors.OutOfRangeError:
                    print "Finished decoding on test dataset"
                    break
            print '{}th post decoded'.format(idx)
            if FLAGS.rerank:
                config['dropout_rate'] = 0.0
                model_rerank = Seq2SeqModel(config, 'train')
                print('starting rerank...')
                reranked_query_decode_list = rerank_by_loss(querys_decode_list, sess, model_rerank, src_vocab, trg_vocab,
                                                    len(FLAGS.gpu_ids.split(',')), FLAGS.decode_batch_size)
                querys_decode_list = [(query, map(lambda x:x[0], ranked_resp)) for query, ranked_resp in reranked_query_decode_list]
        querys_decode_list = [x[1][0] for x in querys_decode_list]
        hypothesis = map(lambda x: x.split(), querys_decode_list)
        weights = [(0.25, 0.25, 0.25, 0.25), (1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, 1.0)]
        bleus = []
        print('starting calcaulate bleu on validation set')
        for weight in weights:
          bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, hypothesis, weights=weight)
          bleus.append(bleu * 100)
        for i in range(len(bleus)):
          if i == 0:
            print('BLEU: {}'.format(bleus[i]))
          else:
            print('bleu{}: {}'.format(i, bleus[i]))


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_ids
    compute_bleu(FLAGS)
