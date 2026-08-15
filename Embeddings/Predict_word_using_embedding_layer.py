with open("/workspaces/NLP-Learning/Embeddings/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text)

#-------------------------------
# Create a vocabulary
#-------------------------------

tokens = text.lower().split()

print(tokens[:30])

vocab = sorted(set(tokens))

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)

print(word_to_id["the"])

#--------------------------------------
# Create the training examples
# We want: current word → next word
#--------------------------------------

X = []
y = []

for i in range(len(tokens) - 1):
    current_word = tokens[i]
    next_word = tokens[i + 1]

    X.append(word_to_id[current_word])
    y.append(word_to_id[next_word])

print(X[0])
print(y[0])

#----------------------------------
# build the neural network
#----------------------------------

import torch
import torch.nn as nn

class NextWordModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.linear = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        # Convert word ID into embedding
        x = self.embedding(x)

        # Convert embedding into vocabulary scores
        x = self.linear(x)

        return x

#-----------------------------
#Train it: Convert the examples to tensors
#-----------------------------

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

#Create the model:
embedding_dim = 100

model = NextWordModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)

#Loss:
criterion = nn.CrossEntropyLoss()

#Optimizer:
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

for epoch in range(epochs):

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(predictions, y)

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {loss.item():.4f}"
    )

embedding_matrix = model.embedding.weight.detach()
print(embedding_matrix.shape)

cat_id = word_to_id["cat"]

cat_embedding = embedding_matrix[cat_id]

print(cat_embedding)

def predict_next_word(word):

    word_id = torch.tensor(
        [word_to_id[word]]
    )

    with torch.no_grad():
        scores = model(word_id)

    predicted_id = scores.argmax(dim=1).item()

    return id_to_word[predicted_id]

print(predict_next_word("the"))

import torch.nn.functional as F

def similar_words(word, n=10):

    word_id = word_to_id[word]

    target = model.embedding.weight[word_id]

    similarities = F.cosine_similarity(
        target.unsqueeze(0),
        model.embedding.weight
    )

    values, indices = torch.topk(
        similarities,
        n + 1
    )

    results = []

    for index in indices:
        candidate = id_to_word[index.item()]

        if candidate != word:
            results.append(candidate)

    return results[:n]

similar_words("sat")