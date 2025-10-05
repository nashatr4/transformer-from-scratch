# /components/utils.py
import numpy as np

def softmax(x):
  """Fungsi aktivasi softmax"""
  e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return e_x / np.sum(e_x, axis=-1, keepdims=True)

def causal_mask(seq_len):
  """Causal mask untuk decoder"""
  mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=1)
  return np.where(mask, -np.inf, 0)