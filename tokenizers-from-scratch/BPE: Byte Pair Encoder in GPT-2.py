from collections import defaultdict
import re


# =========================================================
# STEP 1: PREPROCESS SENTENCE
# =========================================================
# Definition:
# Cleans and splits sentence into words.
#
# Arguments:
# sentence -> input text
#
# Functionality:
# - Lowercases text
# - Separates punctuation
# - Splits into words
#
# Why Needed?
# Raw text contains punctuation, mixed casing,
# and irregular spacing.
#
# Example:
# "Hello, World!"
# ->
# ['hello', ',', 'world', '!']
# =========================================================

def preprocess_text(sentence):

    # Lowercase
    sentence = sentence.lower()

    # Separate punctuation
    sentence = re.sub(r'([.,!?])', r' \1 ', sentence)

    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    # Split into words
    words = sentence.split()

    return words


# =========================================================
# STEP 2: BUILD INITIAL VOCABULARY
# =========================================================
# Definition:
# Converts words into character tokens.
#
# Arguments:
# corpus -> list of words
#
# Functionality:
# - Splits words into characters
# - Adds </w>
# - Counts frequency
#
# Why Needed?
# BPE starts from character-level vocabulary.
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
# Definition:
# Counts adjacent token pair frequency.
#
# Arguments:
# vocab -> tokenized vocabulary
#
# Functionality:
# Finds most common neighboring tokens.
#
# Why Needed?
# BPE merges the most frequent pair.
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
# Definition:
# Finds most frequent pair.
#
# Arguments:
# pairs -> pair frequency dictionary
#
# Why Needed?
# Most frequent pair becomes next merge.
# =========================================================

def get_best_pair(pairs):

    return max(pairs, key=pairs.get)

# =========================================================
# HELPER FUNCTION:
# GET ADJACENT TOKEN PAIRS
# =========================================================
# Definition:
# Finds all neighboring token pairs.
#
# Example:
# ['l', 'o', 'w']
#
# ->
#
# {
#   ('l', 'o'),
#   ('o', 'w')
# }
# =========================================================

def get_adjacent_pairs(tokens):

    pairs = set()

    for i in range(len(tokens) - 1):

        pairs.add(
            (tokens[i], tokens[i + 1])
        )

    return pairs


# =========================================================
# STEP 5: MERGE TOKEN PAIRS
# =========================================================
# Definition:
# Merges selected token pair.
#
# Arguments:
# pair -> best pair
# vocab -> vocabulary
#
# Example:
# ('l', 'o') -> 'lo'
#
# Why Needed?
# Creates larger subword units.
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
# STEP 6: TRAIN BPE WITH MERGE RANKING
# =========================================================
# Definition:
# Learns merge rules and stores merge priority.
#
# Functionality:
# Earlier merges get lower rank numbers.
#
# Example:
# ('l', 'o') -> rank 0
# ('lo', 'w') -> rank 1
# =========================================================

def train_bpe(corpus, num_merges=10):

    vocab = get_vocab(corpus)

    # Store merge rankings
    merge_ranks = {}

    for step in range(num_merges):

        # Count token pairs
        pairs = get_pairs(vocab)

        if not pairs:
            break

        # Select best pair
        best_pair = get_best_pair(pairs)

        # Merge pair
        vocab = merge_vocab(best_pair, vocab)

        # Store ranking
        merge_ranks[best_pair] = step

        print(
            f"Step {step + 1}: "
            f"Merged {best_pair} "
            f"-> Rank {step}"
        )

    return merge_ranks, vocab


# =========================================================
# STEP 7: ENCODE SINGLE WORD
# =========================================================
# Definition:
# Applies learned merges to one word.
#
# Arguments:
# word -> input word
# merges -> learned merge rules
#
# Why Needed?
# Core BPE encoding step.
# =========================================================

def encode_word(word, merges):

    tokens = list(word) + ['</w>']

    for pair in merges:

        i = 0

        new_tokens = []

        while i < len(tokens):

            if (
                i < len(tokens) - 1 and
                (tokens[i], tokens[i + 1]) == pair
            ):

                new_tokens.append(
                    tokens[i] + tokens[i + 1]
                )

                i += 2

            else:

                new_tokens.append(tokens[i])

                i += 1

        tokens = new_tokens

    return tokens


# =========================================================
# STEP 8: ENCODE SENTENCE
# =========================================================
# Definition:
# Encodes full sentence into BPE tokens.
#
# Arguments:
# sentence -> input sentence
# merges -> learned merges
#
# Functionality:
# 1. Preprocess sentence
# 2. Encode each word
# 3. Combine all tokens
#
# Why Needed?
# Real NLP works on sentences, not isolated words.
# =========================================================

def encode_sentence(sentence, merges):

    all_tokens = []

    words = preprocess_text(sentence)

    for word in words:

        word_tokens = encode_word(word, merges)

        all_tokens.extend(word_tokens)

    return all_tokens


# =========================================================
# STEP 9: BUILD TOKEN VOCABULARY
# =========================================================
# Definition:
# Creates token <-> id mappings.
#
# Arguments:
# final_vocab -> learned vocab
#
# Why Needed?
# Neural networks use numerical IDs.
# =========================================================

def build_token_vocab(
    final_vocab,
    merges,
    special_tokens=None
):

    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>"]

    token_set = set()

    # =====================================================
    # ADD ALL TOKENS FROM FINAL VOCAB
    # =====================================================

    for word in final_vocab:
        token_set.update(word)

    # =====================================================
    # ADD ALL INDIVIDUAL CHARACTERS
    # =====================================================

    for word in final_vocab:

        for token in word:

            # Split merged tokens into characters
            chars = re.findall(
                r'</w>|.',
                token
            )

            token_set.update(chars)

    # =====================================================
    # ADD MERGED TOKENS
    # =====================================================

    for pair in merges:

        merged_token = pair[0] + pair[1]

        token_set.add(merged_token)

    # =====================================================
    # FINAL VOCAB
    # =====================================================

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
# STEP 10: TOKENS -> IDS
# =========================================================
# Definition:
# Converts tokens into numerical IDs.
#
# Why Needed?
# Deep learning models consume IDs.
# =========================================================

def tokens_to_ids(tokens, token_to_id):

    unk_id = token_to_id["<unk>"]

    return [

        token_to_id.get(token, unk_id)

        for token in tokens
    ]


# =========================================================
# STEP 11: IDS -> TOKENS
# =========================================================
# Definition:
# Converts IDs back into tokens.
# =========================================================

def ids_to_tokens(ids, id_to_token):

    return [

        id_to_token[i]

        for i in ids
    ]


# =========================================================
# STEP 12: DECODE TOKENS
# =========================================================
# Definition:
# Converts tokens back into readable text.
#
# Functionality:
# Removes </w> markers.
#
# Why Needed?
# Reconstructs original sentence.
# =========================================================

def decode_tokens(tokens):

    text = ""

    for token in tokens:

        if token.endswith("</w>"):

            text += token.replace("</w>", "") + " "

        else:

            text += token

    return text.strip()


# =========================================================
# STEP 13: FULL SENTENCE -> IDS PIPELINE
# =========================================================
# Definition:
# Full tokenizer pipeline.
#
# sentence
# -> tokens
# -> ids
#
# Why Needed?
# Production tokenizers use full pipelines.
# =========================================================

def encode_sentence_ids(sentence, merges, token_to_id):

    tokens = encode_sentence(sentence, merges)

    ids = tokens_to_ids(tokens, token_to_id)

    return ids


# =========================================================
# TRAINING CORPUS
# =========================================================

corpus = [

    "low",
    "lowest",
    "newer",
    "wider",
    "newest",
    "lower",
    "low"
]


# =========================================================
# TRAIN BPE
# =========================================================

merges, final_vocab = train_bpe(corpus, 10)

print("\nMERGES:\n")

print(merges)


# =========================================================
# BUILD TOKEN VOCAB
# =========================================================

token_to_id, id_to_token = build_token_vocab(
    final_vocab,
    merges
)

print("\nTOKEN TO ID:\n")

print(token_to_id)


# =========================================================
# ENCODE SENTENCE
# =========================================================

sentence = "Hello! I am at the lowest point on earth."

tokens = encode_sentence(sentence, merges)

print("\nENCODED TOKENS:\n")

print(tokens)


# =========================================================
# TOKENS -> IDS
# =========================================================

ids = tokens_to_ids(tokens, token_to_id)

print("\nTOKEN IDS:\n")

print(ids)


# =========================================================
# IDS -> TOKENS
# =========================================================

recovered_tokens = ids_to_tokens(ids, id_to_token)

print("\nRECOVERED TOKENS:\n")

print(recovered_tokens)


# =========================================================
# DECODE BACK
# =========================================================

decoded_text = decode_tokens(recovered_tokens)

print("\nDECODED TEXT:\n")

print(decoded_text)