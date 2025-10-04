import numpy as np
from components.embedding import TokenEmbedding, PositionalEncoding
from components.attention import MultiHeadAttention
from components.layers import FeedForwardNetwork, LayerNormalization
from components.utils import causal_mask, softmax

# Hyperparameter
vocab_size = 1000
d_model = 512
seq_len = 5
batch_size = 2
num_heads = 8
d_ff = 2048 

# Input dan mask
input_tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
causal_mask = causal_mask(seq_len)

print(f"Bentuk Input Tokens: {input_tokens.shape}\n")

# Inisialisasi seluruh komponen
token_embedding_layer = TokenEmbedding(vocab_size, d_model)
positional_encoding_layer = PositionalEncoding(seq_len, d_model)
mha_layer = MultiHeadAttention(d_model, num_heads)
ffn_layer = FeedForwardNetwork(d_model, d_ff)
norm1 = LayerNormalization(d_model)
norm2 = LayerNormalization(d_model)
final_projection_layer = np.random.randn(d_model, vocab_size)

# Menjalankan forward-pass

# Langkah A: Embedding + Positional Encoding
x = token_embedding_layer.forward(input_tokens)
x = positional_encoding_layer.forward(x)
print(f"Bentuk setelah Embedding & Positional Encoding: {x.shape}")

# Langkah B: Blok Transformer (Pre-Norm)
# Add & Norm + Multi-Head Attention
residual_1 = x
x_norm1 = norm1.forward(x)
attention_output = mha_layer.forward(x_norm1, causal_mask)
x = residual_1 + attention_output # Residual Connection
print(f"Bentuk setelah Multi-Head Attention: {x.shape}")

# Add & Norm + Feed-Forward Network
residual_2 = x
x_norm2 = norm2.forward(x)
ffn_output = ffn_layer.forward(x_norm2)
x = residual_2 + ffn_output # Residual Connection
print(f"Bentuk setelah Feed-Forward Network: {x.shape}")

# Langkah C: Output Layer
logits = x @ final_projection_layer
probabilities = softmax(logits)
print(f"\nBentuk Logits Akhir: {logits.shape}")
print(f"Bentuk Probabilitas Akhir: {probabilities.shape}") 

# Probabilitas token berikutnya 
next_token_prob = probabilities[:, -1, :]

for i in range(batch_size):
    sum_prob = np.sum(next_token_prob[i, :])
    print(f"Total Probabilitas untuk sekuens ke-{i+1}: {sum_prob}")
    
print(f"Total Probabilitas: {np.sum(next_token_prob)}")
print(f"Bentuk Probabilitas Token Berikutnya: {next_token_prob.shape}")