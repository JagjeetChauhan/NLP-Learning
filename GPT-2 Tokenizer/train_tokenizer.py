from collections import defaultdict
import re
import json


# =========================================================
# STEP 1: PREPROCESS SENTENCE
# =========================================================

def preprocess_text(sentence):

    sentence = sentence.lower()

    sentence = re.sub(r'([.,!?])', r' \1 ', sentence)

    sentence = re.sub(r'\s+', ' ', sentence).strip()

    words = sentence.split()

    return words


# =========================================================
# STEP 2: BUILD INITIAL VOCABULARY
# =========================================================

def get_vocab(corpus):

    vocab = {}

    for word in corpus:

        tokens = list(word) + ['</w>']

        key = tuple(tokens)

        vocab[key] = vocab.get(key, 0) + 1

    return vocab


# =========================================================
# STEP 3: COUNT TOKEN PAIRS
# =========================================================

def get_pairs(vocab):

    pairs = defaultdict(int)

    for word, freq in vocab.items():

        for i in range(len(word) - 1):

            pairs[(word[i], word[i + 1])] += freq

    return pairs


# =========================================================
# STEP 4: SELECT BEST PAIR
# =========================================================

def get_best_pair(pairs):

    return max(pairs, key=pairs.get)


# =========================================================
# STEP 5: MERGE TOKEN PAIRS
# =========================================================

def merge_vocab(pair, vocab):

    new_vocab = {}

    for word, freq in vocab.items():

        new_word = []

        i = 0

        while i < len(word):

            if (
                i < len(word) - 1 and
                (word[i], word[i + 1]) == pair
            ):

                new_word.append(
                    word[i] + word[i + 1]
                )

                i += 2

            else:

                new_word.append(word[i])

                i += 1

        new_vocab[tuple(new_word)] = (
            new_vocab.get(tuple(new_word), 0) + freq
        )

    return new_vocab


# =========================================================
# STEP 6: TRAIN BPE
# =========================================================

def train_bpe(corpus, num_merges=10):

    vocab = get_vocab(corpus)

    merge_ranks = {}

    for step in range(num_merges):

        pairs = get_pairs(vocab)

        if not pairs:
            break

        best_pair = get_best_pair(pairs)

        vocab = merge_vocab(best_pair, vocab)

        merge_ranks[best_pair] = step

        print(
            f"Step {step + 1}: "
            f"Merged {best_pair}"
        )

    return merge_ranks, vocab


# =========================================================
# STEP 7: BUILD TOKEN VOCAB
# =========================================================

def build_token_vocab(
    final_vocab,
    merge_ranks,
    special_tokens=None
):

    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>"]

    token_set = set()

    for word in final_vocab:
        token_set.update(word)

    for word in final_vocab:

        for token in word:

            chars = re.findall(
                r'</w>|.',
                token
            )

            token_set.update(chars)

    for pair in merge_ranks:

        merged_token = pair[0] + pair[1]

        token_set.add(merged_token)

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


# =========================================================
# STEP 8: SAVE TOKENIZER
# =========================================================

def save_tokenizer(
    merge_ranks,
    token_to_id,
    id_to_token,
    filepath="tokenizer.json"
):

    serializable_merges = {

        " ".join(pair): rank

        for pair, rank in merge_ranks.items()
    }

    tokenizer_data = {

        "merge_ranks": serializable_merges,

        "token_to_id": token_to_id,

        "id_to_token": id_to_token
    }

    with open(filepath, "w") as f:

        json.dump(
            tokenizer_data,
            f,
            indent=4
        )

    print(f"\nTokenizer saved to {filepath}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    list_corpus = [

        "low",
        "lowest",
        "newer",
        "wider",
        "newest",
        "lower",
        "hello",
        "world",
        "point",
        "earth",
        "the",
        "i",
        "am",
        "at",
        "on",
        "!",
        "."
    ]

    merge_ranks, final_vocab = train_bpe(
        list_corpus
    )

    token_to_id, id_to_token = build_token_vocab(
        final_vocab,
        merge_ranks
    )

    save_tokenizer(
        merge_ranks,
        token_to_id,
        id_to_token
    )