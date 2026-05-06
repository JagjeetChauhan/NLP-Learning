# Word-Level Tokenizer:

# print("Word Level Tokenizer: ")
def word_tokenize(text):
    return text.split()

text = "Hello I am Jagjeet, I am a software engineer at Sopra Steria"
# print(text)
tokens = word_tokenize(text)
# print(tokens)

# Build Vocabulary:

def build_vocab(tokens):
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

vocab = build_vocab(tokens)
# print(vocab)

# Encode:

def encode(tokens, vocab):
    return [vocab[token] for token in tokens]

encoded = encode(tokens, vocab)
# print(encoded)

#-----------------------------------------------------------------------#

# print()
# print("Character Level Tokenization: ")
# Character-Level Tokenization

def char_tokenize(text):
    return list(text)

text = "Hello I am Jagjeet"
tokens = char_tokenize(text)
# print(tokens)

def char_vocab(tokens):
    char_store = {}
    for token in tokens:
        if token not in char_store:
            char_store[token] = len(char_store)
    return char_store

chars = char_vocab(tokens)
# print(chars)

def encode(tokens, vocab):
    return [vocab[token] for token in tokens]

encoded = encode(tokens, chars)
# print(encoded)

#---------------------------------------------------------------------#
# SUBWORD TOKENIZATION

# print()
print("SUBWORD TOKENIZATION: ")
print("Byte Pair Tokenization: ")

# Step 0: Initialize (Character Level): We split into characters and add end token:
# print("Step 0:")
# print()
def get_vocab(corpus):
    vocab = {}
    for word in corpus:
        tokens = list(word) + ['</w>']
        key = tuple(tokens)
        vocab[key] = vocab.get(key, 0) + 1
    return vocab

list_corpus = ['low','lowest','newer','wider']
vocab_list = get_vocab(list_corpus)
print("Corpus List: ",list_corpus)
print(vocab_list)

# STEP 1 — Count Pair Frequencies:
# print()
# print("Step 1:")
# print()

from collections import defaultdict
def get_pairs(vocab):
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq

    return pairs
# pairs_list = get_pairs(vocab_list)
# print(pairs_list)

# print()

def get_best_pair(pairs):
    return max(pairs, key=pairs.get)

# print("Step 2: ")
# print()

# best_pairs = get_best_pair(pairs_list)
# print("Best pair: ", best_pairs)

print()
print("Step 3: ")
print()

def merge_vocab(pair, vocab):
    new_vocab = {}

    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word)-1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])
                i+=2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    
    return new_vocab

# merge_vocab_list = merge_vocab(best_pairs, vocab_list)
# print(merge_vocab_list)

# Store merges:
# merges = [best_pairs]

# Step 4: Build Encoder:
def encode(word, merges):
    tokens = list(word) + ['</w>']

    for pair in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens

# print()
# print("Step 4: ")
# print()
# encoded = encode("lowest", merges)
# print(encoded)

def train_bpe(corpus, num_merges=10):
    vocab = get_vocab(corpus)
    merges = []

    for step in range(num_merges):
        
        # STEP 1: count pairs
        pairs = get_pairs(vocab)
        if not pairs:
            break
        
        # STEP 2: choose best pair
        best_pair = get_best_pair(pairs)

        # STEP 3: merge
        vocab = merge_vocab(best_pair, vocab)

        merges.append(best_pair)

        print(f"Step {step+1}: Merged {best_pair}")
    
    return merges, vocab

merges, final_vocab = train_bpe(list_corpus)
print("Learned merges:", merges, "\n")

print("Encoding Examples:")
print("lowest ->", encode("lowest", merges))
print("newer  ->", encode("newer", merges))
print(encode("low", merges))
print(encode("wider", merges))