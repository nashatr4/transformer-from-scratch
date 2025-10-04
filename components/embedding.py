import numpy as np

class TokenEmbedding:
  """Mengubah token integer menjadi vektor embedding"""
  def __init__(self, vocab_size, d_model):
    self.embedding_matrix = np.random.randn(vocab_size, d_model)

  def forward(self, token_indices):
    return self.embedding_matrix[token_indices]
  
class PositionalEncoding:
  """Menambahkan informasi posisi ke embedding."""
  def __init__(self, seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    self.pe = pe[np.newaxis, :, :] 

  def forward(self, x):
    return x + self.pe