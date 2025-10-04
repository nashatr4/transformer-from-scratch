import numpy as np
from components.utils import softmax

def scaled_dot_product_attention(Q, K, V, mask=None):
  """Implementasi sclaed dot product untuk attention"""
  d_k = Q.shape[-1]
  attention_scores = (Q @ K.swapaxes(-2, -1)) / np.sqrt(d_k)
  if mask is not None:
    attention_scores += mask
  attention_weights = softmax(attention_scores)
  output = attention_weights @ V
  return output

class MultiHeadAttention:
  """Implementasi multi-head attention"""
  def __init__(self, d_model, num_heads):
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.W_q = np.random.randn(d_model, d_model)
    self.W_k = np.random.randn(d_model, d_model)
    self.W_v = np.random.randn(d_model, d_model)
    self.W_o = np.random.randn(d_model, d_model)
  
  def forward(self, x, mask=None):
    batch_size, seq_len, _ = x.shape

    # Proyeksi Linear
    Q = x @ self.W_q
    K = x @ self.W_k
    V = x @ self.W_v

    # Reshape untuk Multi-Head
    Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
    K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
    V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)

    # Hitung Attention
    attention_output = scaled_dot_product_attention(Q, K, V, mask)

    # Menggabungkan dan reshape
    concatenated_output = attention_output.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

    # Proyeksi output akhir
    output = concatenated_output @ self.W_o
    return output