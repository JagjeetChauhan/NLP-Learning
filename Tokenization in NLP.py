# Word-Level Tokenizer:

def word_tokenize(text):
    return text.split()

text = "Hello I am Jagjeet, I am a software engineer at Sopra Steria"
print(text)
tokens = word_tokenize(text)
print(tokens)

# Build Vocabulary:

def build_vocab(tokens):
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

vocab = build_vocab(tokens)
print(vocab)

# Encode:

def encode(tokens, vocab):
    return [vocab[token] for token in tokens]

encoded = encode(tokens, vocab)
print(encoded)