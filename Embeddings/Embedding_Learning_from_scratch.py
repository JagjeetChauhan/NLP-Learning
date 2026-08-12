import numpy as np

np.random.seed(42)

# -------------------------
# Hyperparameters
# -------------------------

vocab_size = 5
embedding_dim = 3
learning_rate = 0.1

# -------------------------
# Parameters
# -------------------------

E = np.random.randn(vocab_size, embedding_dim) * 0.01 # Embedding Matrix
W = np.random.randn(embedding_dim, vocab_size) * 0.01

print(E.shape)
print(W.shape)

# -------------------------
# Softmax
# -------------------------

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

# -------------------------
# Forward Pass
# -------------------------

def forward(imput_id, target_id, E,W):
    # 1. Embedding lookup
    e = E[input_id]

    # 2. Compute logits
    logits = W.T@e

    # 3. Softmax
    probs = softmax(logits)

    # 4. Cross entropy
    loss = -np.log(probs[target_id])

    return e, logits, probs, loss

# -------------------------
# Backpropagation
# -------------------------

def backward(input_id, target_id, e, probs, E, W):

    vocab_size = W.shape[1]

    # One-hot target
    y = np.zeros(vocab_size)
    y[target_id] = 1

    # Gradient wrt logits
    d_logits = probs - y

    # Gradient wrt W
    dW = np.outer(e, d_logits)

    # Gradient wrt embedding
    de = W @ d_logits

    # Gradient wrt embedding matrix
    dE = np.zeros_like(E)

    # Only selected row gets gradient
    dE[input_id] = de # Sparse Embedding Gradient.

    return dE, dW

# -------------------------
# SGD Update
# -------------------------

def sgd_update(E, W, dE, dW, learning_rate):

    E -= learning_rate * dE
    W -= learning_rate * dW

    return E, W

# Complete Training Step

input_id = 1
target_id = 2

# Forward
e, logits, probs, loss = forward(
    input_id,
    target_id,
    E,
    W
)

# Backward
dE, dW = backward(
    input_id,
    target_id,
    e,
    probs,
    E,
    W
)

# Update
E, W = sgd_update(
    E,
    W,
    dE,
    dW,
    learning_rate
)

print("Loss:", loss)
print("\nEmbedding gradient:")
print()
print(dE)

# --------------------
# Training Loop
# --------------------

training_data = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (3, 4),
    (4, 3),
] # input → target example: 0 -> 1 means: Given word 0, predict word 1.

# ---------------
# Training
# ---------------

for epoch in range(5000):

    total_loss = 0

    for input_id, target_id in training_data:

        # Forward
        e, logits, probs, loss = forward(
            input_id,
            target_id,
            E,
            W
        )

        # Backward
        dE, dW = backward(
            input_id,
            target_id,
            e,
            probs,
            E,
            W
        )

        # SGD
        E, W = sgd_update(
            E,
            W,
            dE,
            dW,
            learning_rate
        )

        total_loss += loss

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, "
            f"Loss = {total_loss:.4f}"
        )

print()
print(dE)