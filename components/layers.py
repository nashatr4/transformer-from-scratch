# /components/layers.py
import numpy as np

class LayerNormalization:
  """Implementasi layer normalization"""
  def __init__(self, d_model, epsilon=1e-5):
    self.epsilon = epsilon
    self.gamma = np.ones(d_model)
    self.beta = np.zeros(d_model)

  def forward(self, x):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
  
class FeedForwardNetwork:
  """Implementasi feed-forward network"""
  def __init__(self, d_model, d_ff):
    scale1 = 1 / np.sqrt(d_model)
    scale2 = 1 / np.sqrt(d_ff)
    
    self.W1 = np.random.randn(d_model, d_ff) * scale1
    self.b1 = np.zeros(d_ff)
    self.W2 = np.random.randn(d_ff, d_model) * scale2
    self.b2 = np.zeros(d_model)

  def relu(self, x):
    return np.maximum(0, x)
  
  def forward(self, x):
    return self.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2