# Word-Level Tokenizer:

print("Word Level Tokenizer: ")
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

#-----------------------------------------------------------------------#

print()
print("Character Level Tokenization: ")
# Character-Level Tokenization

def char_tokenize(text):
    return list(text)

text = "Hello I am Jagjeet"
tokens = char_tokenize(text)
print(tokens)

def char_vocab(tokens):
    char_store = {}
    for token in tokens:
        if token not in char_store:
            char_store[token] = len(char_store)
    return char_store

chars = char_vocab(tokens)
print(chars)

def encode(tokens, vocab):
    return [vocab[token] for token in tokens]

encoded = encode(tokens, chars)
print(encoded)

#---------------------------------------------------------------------#
# SUBWORD TOKENIZATION

print()
print("SUBWORD TOKENIZATION: ")
