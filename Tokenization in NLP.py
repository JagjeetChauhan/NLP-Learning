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
print("Byte Pair Tokenization: ")

# Step 0: Initialize (Character Level): We split into characters and add end token:

def get_vocab(corpus):
    vocab = {}
    for word in corpus:
        tokens = list(word) + ['</w>']
        key = tuple(tokens)
        vocab[key] = vocab.get(key, 0) + 1
    return vocab

list_corpus = ['low','lowest','newer','wider']
vocab_list = get_vocab(list_corpus)
print(vocab_list)

# STEP 1 — Count Pair Frequencies:

from collections import defaultdict
def get_pairs(vocab):
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq

    return pairs
pairs_list = get_pairs(vocab_list)
print(pairs_list)
