import numpy as np

# One Hot Encoding
vocab = {
    "cat": 0,
    "dog": 1,
    "apple": 2,
    "car": 3,
    "book": 4
}

V = len(vocab)

word = "apple"

one_hot = np.zeros(V)
one_hot[vocab[word]] = 1

print(one_hot)

# One-Hot Similarity
cat = np.array([1, 0, 0, 0])
dog = np.array([0, 1, 0, 0])
apple = np.array([0, 0, 1, 0])

print("cat · dog =", np.dot(cat, dog))
print("cat · apple =", np.dot(cat, apple))

print("Distance(cat, dog) =", np.linalg.norm(cat - dog))
print("Distance(cat, apple) =", np.linalg.norm(cat - apple))

# Embedding Layer
vocab_size = 10
embedding_dim = 4 # No. of Features

embedding_matrix = np.random.randn(vocab_size, embedding_dim)

print(embedding_matrix.shape)
print(embedding_matrix)

# Embedding Lookup -> Forward Pass
token = 5
vector = embedding_matrix[token]
print("Single Word")
print(vector)

tokens = np.array([4,5,6])
vectors = embedding_matrix[tokens]
print("Single Sentence")
print(vectors)

batch_sentences = np.array([[4,5,6],[7,8,9]])
vector_sentence = embedding_matrix[batch_sentences]
print("Batch of Sentences")
print(vector_sentence)

#---------------------------------------------------------Production Level Code------------------------------------------
def Embedding(vocab_size, embedding_dim, padding_idx=None):
    embedding_matrix = np.random.normal(loc=0.0, scale=0.01, size=(vocab_size, embedding_dim)).astype(np.float32)
    if padding_idx is not None:
        embedding_matrix[padding_idx] = 0.0
    
    return embedding_matrix

def forward(token_ids, embedding_matrix):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    return embedding_matrix[token_ids]

import numpy as np

embedding = Embedding(
    vocab_size=10,
    embedding_dim=4,
    padding_idx=6
)

print(embedding.shape)

batch = np.array([
    [1, 2, 3, 6],   # I love NLP <PAD>
    [4, 5, 6, 6]    # cats dogs <PAD> <PAD>
])

output = forward(batch, embedding)

print(output.shape)
print()
print(output)

# ( Batch Size, Sequence Length, Embedding Dimension )
#       ↓            ↓                ↓
#       2            4                4
#       ↓            ↓                ↓
#     No. of     each sentence   Each token in each sentence is represented by a vector containing 4 features.
#    Sentence  contains 4 tokens
