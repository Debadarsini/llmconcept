import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        mask: optional (batch, seq_len, seq_len) boolean mask
    Returns:
        output: (batch, seq_len, d_v)
        weights: (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Step 1: compute raw attention scores
    # Q @ K^T -> (batch, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Step 2: apply mask (set masked positions to large negative number)
    if mask is not None:
        scores = np.where(mask == 0, scores, -1e9)

    # Step 3: softmax over last dimension (across key positions)
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))  # numerically stable
    weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Step 4: weighted sum of values
    output = np.matmul(weights, V)

    return output, weights


def causal_mask(seq_len):
    """Upper triangular mask to prevent attending to future positions."""
    # Returns True where attention should be BLOCKED (upper triangle, excluding diagonal)
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


# --- Dry run ---
batch, seq_len, d_k, d_v = 1, 4, 8, 8

np.random.seed(0)
Q = np.random.randn(batch, seq_len, d_k)
K = np.random.randn(batch, seq_len, d_k)
V = np.random.randn(batch, seq_len, d_v)

mask = causal_mask(seq_len)
output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

print("Attention weights (should be lower-triangular):")
print(np.round(weights[0], 3))
print("\nOutput shape:", output.shape)