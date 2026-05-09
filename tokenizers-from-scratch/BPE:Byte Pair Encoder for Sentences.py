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
        for i in range(len(word) -1):
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
            if(
                i < len(word) - 1 and
                (word[i], word[i + 1]) == pair
            ):
                new_word.append(
                    word[i] + word[i + 1]
                )
                i += 2
            else:
                    new_word.append(word[i])
                    i+=1
        new_vocab[tuple(new_word)] = (
            new_vocab.get(tuple(new_word), 0) + freq
        )
    return new_vocab

def train_bpe(corpus, num_merges):
    vocab = get_vocab(corpus)
    merges = []
    for step in range(num_merges):
        pairs = get_pairs(vocab)

        if not pairs:
            break

        best_pair = get_best_pair(pairs)

        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        print(f"Step {step + 1}: Meged {best_pair}")

    return merges, vocab

corpus = [

    "low",
    "lowest",
    "newer",
    "wider",
    "newest",
    "lower",
    "low"
]

vocab_list = get_vocab(corpus)
print(vocab_list)
print()

pairs_list = get_pairs(vocab_list)
print(pairs_list)
print()

best_pairs = get_best_pair(pairs_list)
print("Best Pair: ",best_pairs)
print()

merge_list = merge_vocab(best_pairs, vocab_list)
print(merge_list)

train_merges, train_vocab = train_bpe(corpus, 5)
print("Trained Merges: ",train_merges)
print()
print("Trained Vocab: ",train_vocab)
