with open("/workspaces/NLP-Learning/Embeddings/data.txt", "r", encoding="utf-8") as f:
    sentence = f.read().splitlines()

tokens = []

for j in sentence:
    tokens.extend(j.lower().split()) 

print(f"Tokens: {tokens}")

vocab = sorted(set(tokens))
print(f"Vocab: {vocab}")

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for i, word in enumerate(vocab)
}

vocab_size = len(vocab)
print(f"Vocab Length: {vocab_size}")

print(word_to_id["is"])

x = []
y = []

for s in sentence:

    tokens = s.lower().split()

    for i in range(len(tokens) - 2):

        input_context1 = tokens[i]
        input_context2 = tokens[i + 1]
        output_context = tokens[i + 2]

        x.append([
            word_to_id[input_context1],
            word_to_id[input_context2]
        ])

        y.append(
            word_to_id[output_context]
        )

print(x)
print(y)

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

        # [batch_size, 2]
        x = self.embedding(x)

        # [batch_size, 2, embedding_dim]
        x = x.mean(dim=1)

        # [batch_size, embedding_dim]
        x = self.linear(x)

        # [batch_size, vocab_size]
        return x

X = torch.tensor(x, dtype=torch.long)
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

epochs = 200

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

def predict_next_word(word1, word2):

    input_tensor = torch.tensor(
        [[
            word_to_id[word1],
            word_to_id[word2]
        ]],
        dtype=torch.long
    )

    with torch.no_grad():
        scores = model(input_tensor)

    predicted_id = scores.argmax(dim=1).item()

    return id_to_word[predicted_id]

print(predict_next_word("today", "is"))

def generate_sentence(word1, word2, model, max_words=20):

    word1 = word1.lower()
    word2 = word2.lower()

    if word1 not in word_to_id:
        return f"Word '{word1}' is not in the vocabulary."

    if word2 not in word_to_id:
        return f"Word '{word2}' is not in the vocabulary."

    words = [word1, word2]

    for _ in range(max_words - 2):

        input_tensor = torch.tensor(
            [[
                word_to_id[words[-2]],
                word_to_id[words[-1]]
            ]],
            dtype=torch.long
        )

        with torch.no_grad():
            scores = model(input_tensor)

        predicted_id = scores.argmax(dim=1).item()

        next_word = id_to_word[predicted_id]

        words.append(next_word)

    return " ".join(words)

print(generate_sentence("today", "is", model))