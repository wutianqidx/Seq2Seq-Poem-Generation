# coding=utf-8

import tensorflow.contrib.seq2seq as seq2seq
import tensorflow
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

class GreedyIgnoreUnkEmbeddingHelper(seq2seq.GreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input without UNK
  """

  def __init__(self, embedding, start_tokens, unk_token, end_token):
    """
    inherit from from seq2seq.GreedyEmbeddingHelper
    """
    super(GreedyIgnoreUnkEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)
    self._unk_token = unk_token

  def sample(self, time, outputs, state, name=None):
    """overwrite sample for GreedyEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    top_2_values, top_2_indices = nn_ops.top_k(outputs, 2)
    first_idxs = tensorflow.stop_gradient(top_2_indices[:, 0])
    second_idxs = tensorflow.stop_gradient(top_2_indices[:, 1])
    comparision = math_ops.equal(first_idxs, self._unk_token)
    final_idxs = tensorflow.stop_gradient(tensorflow.where(comparision, second_idxs, first_idxs))
    return final_idxs