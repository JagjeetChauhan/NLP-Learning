# =========================================================
# BYTE PAIR ENCODING (BPE) TOKENIZER
# =========================================================

print("Byte Pair Encoding:\n")

from collections import defaultdict


# =========================================================
# STEP 1: BUILD INITIAL VOCABULARY
# =========================================================
# Definition:
# Converts each word into character-level tokens.
#
# Example:
# "low" -> ('l', 'o', 'w', '</w>')
#
# Functionality:
# - Adds end-of-word token </w>
# - Counts word frequency
# - Creates vocabulary dictionary
# =========================================================

def get_vocab(corpus):

    vocab = {}

    for word in corpus:

        # Split word into characters
        tokens = list(word) + ['</w>']

        # Convert to tuple so it can be dictionary key
        key = tuple(tokens)

        # Count frequency
        vocab[key] = vocab.get(key, 0) + 1

    return vocab


# =========================================================
# STEP 2: COUNT TOKEN PAIRS
# =========================================================
# Definition:
# Counts frequency of adjacent token pairs.
#
# Example:
# ('l', 'o')
# ('o', 'w')
#
# Functionality:
# Finds most common neighboring tokens.
# =========================================================

def get_pairs(vocab):

    pairs = defaultdict(int)

    for word, freq in vocab.items():

        for i in range(len(word) - 1):

            pairs[(word[i], word[i + 1])] += freq

    return pairs


# =========================================================
# STEP 3: SELECT BEST PAIR
# =========================================================
# Definition:
# Finds the most frequent token pair.
#
# Functionality:
# Chooses pair to merge next.
# =========================================================

def get_best_pair(pairs):

    return max(pairs, key=pairs.get)


# =========================================================
# STEP 4: MERGE TOKEN PAIRS
# =========================================================
# Definition:
# Replaces the best pair with merged token.
#
# Example:
# ('l', 'o') -> 'lo'
#
# Functionality:
# Updates vocabulary with merged tokens.
# =========================================================

def merge_vocab(pair, vocab):

    new_vocab = {}

    for word, freq in vocab.items():

        new_word = []

        i = 0

        while i < len(word):

            # Merge matching pair
            if (
                i < len(word) - 1 and
                (word[i], word[i + 1]) == pair
            ):

                new_word.append(word[i] + word[i + 1])

                i += 2

            else:

                new_word.append(word[i])

                i += 1

        new_vocab[tuple(new_word)] = (
            new_vocab.get(tuple(new_word), 0) + freq
        )

    return new_vocab


# =========================================================
# STEP 5: TRAIN BPE
# =========================================================
# Definition:
# Learns merge rules from corpus.
#
# Functionality:
# Repeatedly:
# 1. Count pairs
# 2. Select best pair
# 3. Merge pair
# =========================================================

def train_bpe(corpus, num_merges=10):

    vocab = get_vocab(corpus)

    merges = []

    for step in range(num_merges):

        # Count token pairs
        pairs = get_pairs(vocab)

        if not pairs:
            break

        # Select most frequent pair
        best_pair = get_best_pair(pairs)

        # Merge pair
        vocab = merge_vocab(best_pair, vocab)

        merges.append(best_pair)

        print(f"Step {step + 1}: Merged {best_pair}")

    return merges, vocab


# =========================================================
# STEP 6: ENCODE TEXT USING LEARNED MERGES
# =========================================================
# Definition:
# Applies learned BPE merges to new text.
#
# Functionality:
# Converts text -> subword tokens
# =========================================================

def encode(word, merges):

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
# STEP 7: BUILD TOKEN VOCABULARY
# =========================================================
# Definition:
# Creates token <-> ID mappings.
#
# Functionality:
# token -> id
# id -> token
# =========================================================

def build_token_vocab(final_vocab, special_tokens=None):

    if special_tokens is None:

        special_tokens = ["<pad>", "<unk>"]

    token_set = set()

    # Collect learned tokens
    for word in final_vocab:

        token_set.update(word)

    # Combine special + learned tokens
    all_tokens = special_tokens + sorted(token_set)

    # token -> id
    token_to_id = {

        token: idx

        for idx, token in enumerate(all_tokens)
    }

    # id -> token
    id_to_token = {

        idx: token

        for token, idx in token_to_id.items()
    }

    return token_to_id, id_to_token


# =========================================================
# STEP 8: TOKENS -> IDS
# =========================================================
# Definition:
# Converts tokens into numerical IDs.
#
# Functionality:
# Handles unknown tokens using <unk>
# =========================================================

def tokens_to_ids(tokens, token_to_id):

    unk_id = token_to_id["<unk>"]

    return [

        token_to_id.get(token, unk_id)

        for token in tokens
    ]


# =========================================================
# STEP 9: IDS -> TOKENS
# =========================================================
# Definition:
# Converts numerical IDs back into tokens.
# =========================================================

def ids_to_tokens(ids, id_to_token):

    return [

        id_to_token[i]

        for i in ids
    ]


# =========================================================
# STEP 10: ENCODE TEXT -> IDS
# =========================================================
# Definition:
# Full encoding pipeline.
#
# Functionality:
# text -> tokens -> ids
# =========================================================

def encode_ids(word, merges, token_to_id):

    tokens = encode(word, merges)

    return tokens_to_ids(tokens, token_to_id)


# =========================================================
# STEP 11: DECODE TOKENS -> TEXT
# =========================================================
# Definition:
# Converts tokens back into readable text.
#
# Functionality:
# Removes </w> markers
# =========================================================

def decode(tokens):

    text = ''.join(tokens)

    return text.replace('</w>', ' ').strip()


# =========================================================
# TRAIN TOKENIZER
# =========================================================

list_corpus = ['low', 'lowest', 'newer', 'wider']

merges, final_vocab = train_bpe(list_corpus)


# =========================================================
# BUILD TOKEN VOCABULARY
# =========================================================

token_to_id, id_to_token = build_token_vocab(final_vocab)

print("\nLearned merges:\n")

print(merges)


# =========================================================
# ENCODING EXAMPLES
# =========================================================

print("\nEncoding Examples:\n")

print("lowest ->", encode("lowest", merges))

print("newer  ->", encode("newer", merges))

print("low     ->", encode("low", merges))

print("wider   ->", encode("wider", merges))


# =========================================================
# TOKEN VOCABULARY
# =========================================================

print("\nTOKEN TO ID:\n")

print(token_to_id)

print("\nID TO TOKEN:\n")

print(id_to_token)


# =========================================================
# TOKEN ID ENCODING
# =========================================================

print("\nTOKEN ID ENCODING:\n")

print(
    "lowest ->",
    encode_ids("lowest", merges, token_to_id)
)

print(
    "newer  ->",
    encode_ids("newer", merges, token_to_id)
)

print(
    "wider  ->",
    encode_ids("wider", merges, token_to_id)
)


# =========================================================
# FULL TOKENIZER ROUNDTRIP
# =========================================================
# Text
# -> Tokens
# -> IDs
# -> Tokens
# -> Decoded Text
# =========================================================

print("\nFULL ROUNDTRIP TEST:\n")

word = "lowest"

# Text -> Tokens
tokens = encode(word, merges)

print("TOKENS:")

print(tokens)

# Tokens -> IDs
ids = tokens_to_ids(tokens, token_to_id)

print("\nTOKEN IDS:")

print(ids)

# IDs -> Tokens
recovered_tokens = ids_to_tokens(ids, id_to_token)

print("\nRECOVERED TOKENS:")

print(recovered_tokens)

# Tokens -> Text
decoded_text = decode(recovered_tokens)

print("\nDECODED TEXT:")

print(decoded_text)

