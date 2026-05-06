# Byte Pair Encoding
print("Byte Pair Encoding:")

from collections import defaultdict

def get_vocab(corpus):
    vocab = {}
    for word in corpus:
        tokens = list(word) + ['</w>']
        key = tuple(tokens)
        vocab[key] = vocab.get(key, 0) + 1
    return vocab

def get_pairs(vocab):
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq

    return pairs

def get_best_pair(pairs):
    return max(pairs, key=pairs.get)

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



list_corpus = ['low','lowest','newer','wider']
merges, final_vocab = train_bpe(list_corpus)
print("Learned merges:", merges, "\n")

print("Encoding Examples:")
print("lowest ->", encode("lowest", merges))
print("newer  ->", encode("newer", merges))
print(encode("low", merges))
print(encode("wider", merges))