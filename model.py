# -*- coding: utf-8 -*-

import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.ops import array_ops
import greedy_decoder_helper
import beamsearch_decoder_helper
import attention_helper


UNK = "<UNK>"  # 0
SOS = "<SOS>"  # 1
EOS = "<EOS>"  # 2
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


class Seq2SeqModel(object):
    """
    An attention-based seq2seq model
    """

    def __init__(self, config, mode):

        assert mode.lower() in ['train', 'decode']
        self.config = config
        self.mode = mode.lower()
        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units']
        self.depth = config['depth']
        self.attention_type = config['attention_type']
        self.decoder_output_arch = config['decoder_output_arch'] if 'decoder_output_arch' in config else 'Local'
        self.max_seq_length = config['max_seq_length']
        self.encoder_type = config["encoder_type"]
        # num symbols should equals to the size of the vocab
        self.num_encoder_symbols = config['num_encoder_symbols']
        self.num_decoder_symbols = config['num_decoder_symbols']
        self.use_residual = config['use_residual']
        self.attn_input_feeding = config['attn_input_feeding']
        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']
        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.decay_rate = config['decay_rate']
        self.max_gradient_norm = config['max_gradient_norm']
        self.dtype = tf.float32
        self.use_beamsearch_decode = False
        if self.mode == 'decode':
            self.beam_width = config['beam_width']
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = config['max_decode_step']
            self.keep_prob = 1.0
            self.decode_ignore_unk = config['decode_ignore_unk']


    def build_model(self, input_fn=None):
        """
        Input is a input function
        And return a loss
        """
        if input_fn:
            print "building model.."
            self.encoder_inputs = input_fn.source
            self.encoder_inputs_length = input_fn.source_sequence_length
            self.batch_size = tf.shape(self.encoder_inputs)[0]
            if self.mode == "train":
                self.decoder_inputs_train = input_fn.target_input
                self.decoder_targets_train = input_fn.target_output
                self.decoder_inputs_length_train = input_fn.target_sequence_length

        self.build_encoder()
        self.build_decoder()
        # Merge all the training summaries
        # self.summary_op = tf.summary.merge_all()

        if hasattr(self, 'loss'):
            return self.loss
        elif hasattr(self, "decoder_pred_decode"):
            return self.decoder_pred_decode
        else:
            raise Exception("Not train nor pred")

    def build_encoder(self):
        print "building encoder.."
        with tf.variable_scope('encoder'):
            # Building encoder_cell

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = 0.1  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
            self.encoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.num_encoder_symbols, self.hidden_units],
                                                      initializer=initializer, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            if self.encoder_type == "bi":
                # Bidirectional RNN
                self.encoder_cell_fw, self.encoder_cell_bw = self.build_encoder_cell()
                encoder_outputs, encoder_last_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.encoder_cell_fw,
                    cell_bw=self.encoder_cell_bw,
                    inputs=self.encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                    time_major=False
                )
                ## average
                encoder_outputs = (encoder_outputs[0] + encoder_outputs[1]) / 2
                temp_encoder_last_state = []
                if self.cell_type.lower() == "lstm":
                    for i in range(self.depth):
                        temp_c = (encoder_last_state[0][i].c + encoder_last_state[1][i].c) / 2
                        temp_h = (encoder_last_state[0][i].h + encoder_last_state[1][i].h) / 2
                        temp_tuple = tf.contrib.rnn.LSTMStateTuple(c=temp_c, h=temp_h)
                        temp_encoder_last_state.append(temp_tuple)
                elif self.cell_type.lower() == "gru":
                    for i in range(self.depth):
                        temp = (encoder_last_state[0][i] + encoder_last_state[1][i]) / 2
                        temp_encoder_last_state.append(temp)
                else:
                    raise Exception("Invalid cell type for bidirection")
                encoder_last_state = tuple(temp_encoder_last_state)

            elif self.encoder_type == "uni":
                # Unidirectional RNN
                self.encoder_cell, _ = self.build_encoder_cell()
                encoder_outputs, encoder_last_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                    time_major=False)

            else:
                # Just Embedding
                self.position_embeddings = tf.get_variable(name='position_embedding',
                    shape=[4, self.hidden_units],
                    initializer=initializer, dtype=self.dtype)
                position = tf.range(4) 
                position = tf.expand_dims(position, 0)
                position = tf.tile(position, [self.batch_size, 1])
                position_embedded = tf.nn.embedding_lookup(                                                                                                                  
                    params=self.position_embeddings, ids=position)
                encoder_outputs = self.encoder_inputs_embedded + position_embedded
                encoder_outputs = self.encoder_inputs_embedded
                encoder_last_state = [tf.reduce_mean(self.encoder_inputs_embedded, 1) for _ in range(self.depth)] 
            
            self.encoder_outputs = encoder_outputs
            self.encoder_last_state = encoder_last_state

    def build_decoder(self):
        print "building decoder and attention.."
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = 0.1  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
            self.decoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.num_decoder_symbols, self.hidden_units],
                                                      initializer=initializer, dtype=self.dtype)
            
            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            
            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols, name='output_projection')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs_train)

                # Embedded inputs having gone through input projection layer

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_length_train,
                                                         time_major=False,
                                                         name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                        helper=training_helper,
                                                        initial_state=self.decoder_initial_state,
                                                        output_layer=output_layer)

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))

                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                  targets=self.decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True, )

                # Training summary for the current batch_loss
                # tf.summary.scalar('loss', self.loss)
                # Contruct graphs for minimizing loss

            elif self.mode == 'decode':

                # Start_tokens: [batch_size,] `int32` vector
                unk_token=0
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * 1 # start_token
                end_token = 2 #

                def embed_and_input_proj(inputs):
                    return tf.nn.embedding_lookup(self.decoder_embeddings, inputs)

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    if not self.decode_ignore_unk:
                        decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                        end_token=end_token,
                                                                        embedding=embed_and_input_proj)
                    else:
                        decoding_helper = greedy_decoder_helper.GreedyIgnoreUnkEmbeddingHelper(start_tokens=start_tokens,
                                                                                               end_token=end_token,
                                                                                               unk_token=unk_token,
                                                                                               embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print "building greedy decoder.."
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print "building beamsearch decoder.."
                    if not self.decode_ignore_unk:
                        inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                                                  embedding=embed_and_input_proj,
                                                                                  start_tokens=start_tokens,
                                                                                  end_token=end_token,
                                                                                  initial_state=self.decoder_initial_state,
                                                                                  beam_width=self.beam_width,
                                                                                  output_layer=output_layer, )
                    else:
                        inference_decoder = beamsearch_decoder_helper.BeamSearchIgnoreUnkDecoder(cell=self.decoder_cell,
                                                                                             embedding=embed_and_input_proj,
                                                                                             start_tokens=start_tokens,
                                                                                             end_token=end_token,
                                                                                             initial_state=self.decoder_initial_state,
                                                                                             beam_width=self.beam_width,
                                                                                             output_layer=output_layer, )
                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True 

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #                         namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                #                                                    namedtuple(scores, predicted_ids, parent_ids)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    # impute_finished=True,	# error occurs
                    maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def build_single_cell(self):
        cell_type = LSTMCell
        if self.cell_type.lower() == 'gru':
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            # use variational recurrent
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  input_keep_prob=self.keep_prob,
                                  output_keep_prob=self.keep_prob,
                                  state_keep_prob=self.keep_prob,
                                  variational_recurrent=True,
                                  input_size=tf.TensorShape([self.hidden_units]))

        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    # Building encoder cell
    def build_encoder_cell(self):
        if self.encoder_type == "bi":
            multi_cell_fw = MultiRNNCell([self.build_single_cell() for _ in range(self.depth)])
            multi_cell_bw = MultiRNNCell([self.build_single_cell() for _ in range(self.depth)])
            return multi_cell_fw, multi_cell_bw
        elif self.encoder_type == "uni":
            return MultiRNNCell([self.build_single_cell() for _ in range(self.depth)]), None
        else:
            raise Exception("Invalid Encoder Type")

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self):

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            print ("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.hidden_units, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length, )
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length, )

        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for i in range(self.depth)]
        #decoder_initial_state = encoder_last_state
        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                # no feed the input of the cell with last attention information
                return inputs
            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer

        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
            else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)
        #decoder_initial_state = initial_state[-1]

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def save(self, sess, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        #if not hasattr(self, "saver"):
        #    self.saver = tf.train.Saver(var_list, max_to_keep=10) 
        saver = tf.train.Saver(var_list)
        # del tf.get_collection_ref('LAYER_NAME_UIDS')[0]
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print('model saved at %s' % save_path)

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        #if not hasattr(self, 'saver'):
        saver = tf.train.Saver(var_list,max_to_keep=10)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
