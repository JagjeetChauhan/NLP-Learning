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

def encode_word(word, merges):
    tokens = list(word) + ['</w>']

    for pair in merges:
        i=0
        new_tokens = []
        while i < len(tokens):
            if(
                i<len(tokens) - 1 and
                (tokens[i], tokens[i+1]) == pair
            ):
                new_tokens.append(
                    tokens[i] + tokens[i+1]
                )
                i+=2
            else:
                new_tokens.append(tokens[i])
                i+=1
        tokens = new_tokens
    return tokens

def encode_text(sentence, merges):
    all_tokens = []

    words = sentence.split()
    for word in words:
        word_token = encode_word(word, merges)
        all_tokens.extend(word_token)

    return all_tokens

def build_token_vocab(final_vocab,special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>"]
    token_set = set()
    for word in final_vocab:
        token_set.update(word)
    all_tokens = special_tokens + sorted(token_set)

    token_to_id = {
        token: idx

        for idx, token in enumerate(all_tokens)
    }

    id_to_token = {
        idx: token
        for token, idx in token_to_id.items()
    }

    return token_to_id, id_to_token

def tokens_to_ids(tokens, token_to_id):
    unk_id = token_to_id["<unk>"]
    return [
        token_to_id.get(token, unk_id)
        for token in tokens
    ]

def ids_to_tokens(ids, id_to_token):
    return [
        id_to_token[i]
        for i in ids
    ]

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

encoded_word = encode_word("pizza", train_merges)
print(encoded_word)

sentence = "Hello I am at the lowest point on earth"
encoded_sentence = encode_text(sentence, train_merges)
print(encoded_sentence)

tokens_to_ids, ids_to_tokens = build_token_vocab(train_vocab, None)
print(tokens_to_ids)
print()
print(ids_to_tokens)


