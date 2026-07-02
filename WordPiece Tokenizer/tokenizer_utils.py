import os
import json

# Step 1: Extract Words
def extract_words(corpus):
    words = []

    for sentence in corpus:
        words.extend(sentence.split())

    return words

# Step 2: Word Frequencies
def build_word_frequencies(words):
    word_frequencies = {}

    for word in words:
        word_frequencies[word] = (
            word_frequencies.get(word, 0) + 1
        )
    return word_frequencies

# Step 3: Initial WordPiece
def build_initial_splits(word_frequencies):
    splits = {}
    for word in word_frequencies:
        splits[word] = (
            [word[0]] + [f"##{char}" for char in word[1:]]
        )
    return splits

# Step 4: Token Frequencies
def compute_token_frequencies(
    splits,
    word_frequencies
):

    token_frequencies = {}

    for word, tokens in splits.items():

        word_count = word_frequencies[word]

        for token in tokens:

            token_frequencies[token] = (
                token_frequencies.get(token, 0)
                + word_count
            )

    return token_frequencies

# Step 5: Pair Frequencies
def compute_pair_frequencies(
    splits,
    word_frequencies
):

    pair_frequencies = {}

    for word, tokens in splits.items():

        word_count = word_frequencies[word]

        for i in range(len(tokens) - 1):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            pair_frequencies[pair] = (
                pair_frequencies.get(pair, 0)
                + word_count
            )

    return pair_frequencies

# Step 6: WordPiece Scores
def compute_scores(
    pair_frequencies,
    token_frequencies
):

    scores = {}

    for pair, pair_count in pair_frequencies.items():

        left_token, right_token = pair

        score = (
            pair_count
            /
            (
                token_frequencies[left_token]
                *
                token_frequencies[right_token]
            )
        )

        scores[pair] = score

    return scores

# Step 7: Best Pair
def find_best_pair(scores):

    best_pair = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_pair]

    return best_pair, best_score

# Step 8: Merge best pair into new token
def merge_best_pair(best_pair):
    a, b = best_pair
    if b.startswith("##"):
        return a + b[2:]
    
    return a + b

# Step 9: Merge Pair
def merge_pair(best_pair, splits):
    a, b = best_pair

    for word in splits:
        tokens = splits[word]
        new_tokens = []

        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == a
                and tokens[i+1] == b
            ):
                merged = a + b.replace("##", "")
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        splits[word] = new_tokens
    return splits

# Step 11: Token to ids and ids to Token
def build_token_to_id(vocab):
    return {
        token: idx
        for idx, token in enumerate(sorted(vocab))
    }


def build_id_to_token(token_to_id):
    return {
        idx: token
        for token, idx in token_to_id.items()
    }

# Step 12: WordPiece Encode
def wordpiece_encode(word, vocab):

    tokens = []

    start = 0

    while start < len(word):

        end = len(word)

        current_token = None

        while start < end:

            piece = word[start:end]

            if start > 0:
                piece = "##" + piece

            if piece in vocab:
                current_token = piece
                break

            end -= 1

        if current_token is None:
            return ["[UNK]"]

        tokens.append(current_token)

        start = end

    return tokens

# Step 12: Encode to Ids
def encode_to_ids(word, vocab, token_to_id):
    tokens = wordpiece_encode(word, vocab)

    return [
        token_to_id[token]
        for token in tokens
    ]

# Step 13: Decode to ids
def encode_to_ids(word, vocab, token_to_id):

    tokens = wordpiece_encode(word, vocab)

    unk_id = token_to_id["[UNK]"]

    return [
        token_to_id.get(token, unk_id)
        for token in tokens
    ]

# Step 14: Save Tokenizer
def save_tokenizer(vocab, special_tokens,save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data = {
        "vocab": vocab,
        "special_tokens": special_tokens
    }

    with open(
        os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=4)

# Step 15: Load Tokenizer
def load_tokenizer(save_dir):
    with open(
        os.path.join(save_dir, "tokenizer.json"),
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data["vocab"], data["special_tokens"]