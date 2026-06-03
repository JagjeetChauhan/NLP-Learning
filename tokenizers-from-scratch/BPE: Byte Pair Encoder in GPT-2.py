from collections import defaultdict
import re
import json


# =========================================================
# GPT-2 STYLE BPE TOKENIZER
# =========================================================
#
# FEATURES:
#
# 1. TRAIN BPE
# 2. SAVE TOKENIZER
# 3. LOAD TOKENIZER
# 4. ENCODE TEXT
# 5. TOKENS -> IDS
# 6. IDS -> TOKENS
# 7. DECODE TEXT
#
# =========================================================

# =========================================================
# SPECIAL TOKENS CONFIG
# =========================================================
SPECIAL_TOKENS = {

            "pad_token": "<pad>",

            "unk_token": "<unk>",

            "bos_token": "<bos>",

            "eos_token": "<eos>"
}

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
# - Removes extra spaces
# - Splits into words
#
# Why Needed?
# Raw text contains punctuation, mixed casing,
# and irregular spacing.
#
# Example:
# "Hello, World!"
#
# ->
#
# ['hello', ',', 'world', '!']
# =========================================================

def preprocess_text(sentence):

    # Lowercase
    sentence = sentence.lower()

    # Separate punctuation
    sentence = re.sub(
        r'([.,!?])',
        r' \1 ',
        sentence
    )

    # Remove extra spaces
    sentence = re.sub(
        r'\s+',
        ' ',
        sentence
    ).strip()

    # Split into words
    words = sentence.split()

    return words


# =========================================================
# STEP 2: BUILD INITIAL VOCABULARY
# =========================================================
# Definition:
# Converts words into character-level tokens.
#
# Arguments:
# corpus -> list of words
#
# Functionality:
# - Splits words into characters
# - Adds </w> end-of-word token
# - Counts word frequency
#
# Why Needed?
# BPE starts from character-level vocabulary.
#
# Example:
# "low"
#
# ->
#
# ('l', 'o', 'w', '</w>')
# =========================================================

def get_vocab(corpus):

    vocab = {}

    for word in corpus:

        # Character tokens
        tokens = list(word) + ['</w>']

        # Convert to tuple
        key = tuple(tokens)

        # Count frequency
        vocab[key] = (
            vocab.get(key, 0) + 1
        )

    return vocab


# =========================================================
# STEP 3: COUNT TOKEN PAIRS
# =========================================================
# Definition:
# Counts frequency of adjacent token pairs.
#
# Arguments:
# vocab -> tokenized vocabulary
#
# Functionality:
# Finds neighboring token pairs.
#
# Why Needed?
# BPE merges the most frequent pair first.
#
# Example:
#
# ('l', 'o', 'w')
#
# ->
#
# ('l', 'o')
# ('o', 'w')
# =========================================================

def get_pairs(vocab):

    pairs = defaultdict(int)

    for word, freq in vocab.items():

        for i in range(len(word) - 1):

            pairs[
                (word[i], word[i + 1])
            ] += freq

    return pairs


# =========================================================
# STEP 4: SELECT BEST PAIR
# =========================================================
# Definition:
# Finds most frequent token pair.
#
# Arguments:
# pairs -> pair frequency dictionary
#
# Why Needed?
# Most frequent pair becomes next merge.
# =========================================================

def get_best_pair(pairs):

    return max(
        pairs,
        key=pairs.get
    )


# =========================================================
# HELPER FUNCTION:
# GET ADJACENT TOKEN PAIRS
# =========================================================
# Definition:
# Finds neighboring token pairs.
#
# Arguments:
# tokens -> list of tokens
#
# Example:
#
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
# Functionality:
# Combines neighboring tokens.
#
# Example:
#
# ('l', 'o')
#
# ->
#
# 'lo'
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

            # Merge pair
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
            new_vocab.get(
                tuple(new_word),
                0
            ) + freq
        )

    return new_vocab


# =========================================================
# STEP 6: TRAIN BPE WITH MERGE RANKING
# =========================================================
# Definition:
# Learns merge rules and merge priority.
#
# Arguments:
# corpus -> training corpus
# num_merges -> number of merges
#
# Functionality:
# - Counts token pairs
# - Selects best pair
# - Merges pair
# - Stores merge ranking
#
# Why Needed?
# Earlier merges get higher priority.
#
# Example:
#
# ('l', 'o') -> rank 0
# ('lo', 'w') -> rank 1
# =========================================================

def train_bpe(
    corpus,
    num_merges=10
):

    vocab = get_vocab(corpus)

    # Store merge rankings
    merge_ranks = {}

    for step in range(num_merges):

        # Count token pairs
        pairs = get_pairs(vocab)

        # Stop if no pairs remain
        if not pairs:
            break

        # Select best pair
        best_pair = get_best_pair(pairs)

        # Merge pair
        vocab = merge_vocab(
            best_pair,
            vocab
        )

        # Store merge rank
        merge_ranks[
            best_pair
        ] = step

        print(
            f"Step {step + 1}: "
            f"Merged {best_pair} "
            f"-> Rank {step}"
        )

    return merge_ranks, vocab


# =========================================================
# STEP 7: GPT-STYLE BPE ENCODING
# =========================================================
# Definition:
# Applies learned merges dynamically.
#
# Arguments:
# word -> input word
# merge_ranks -> learned merge priorities
#
# Functionality:
# 1. Find neighboring pairs
# 2. Keep valid learned merges
# 3. Select lowest-ranked pair
# 4. Merge pair
# 5. Repeat until no merges remain
#
# Why Needed?
# GPT-style tokenizers apply merges dynamically.
# =========================================================

def encode(word, merge_ranks):

    # Initial character tokens
    tokens = list(word) + ['</w>']

    while True:

        # Find neighboring pairs
        pairs = get_adjacent_pairs(tokens)

        # Keep only learned merges
        candidate_pairs = {

            pair: merge_ranks[pair]

            for pair in pairs

            if pair in merge_ranks
        }

        # Stop if no valid merges remain
        if not candidate_pairs:
            break

        # Select best-ranked pair
        best_pair = min(
            candidate_pairs,
            key=candidate_pairs.get
        )

        # Merge pair
        new_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1 and
                (tokens[i], tokens[i + 1]) == best_pair
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
# merge_ranks -> learned merges
#
# Functionality:
# 1. Preprocess sentence
# 2. Encode each word
# 3. Combine all tokens
#
# Why Needed?
# Real NLP works on sentences.
# =========================================================

def encode_sentence(
    sentence,
    merge_ranks,
    add_special_tokens=True
):

    all_tokens = []

    words = preprocess_text(sentence)

    for word in words:

        word_tokens = encode(
            word,
            merge_ranks
        )

        all_tokens.extend(word_tokens)
    if add_special_tokens:

        all_tokens = (
            [SPECIAL_TOKENS["bos_token"]]
            +
            all_tokens
            +
            [SPECIAL_TOKENS["eos_token"]]
        )

    return all_tokens


# =========================================================
# STEP 9: BUILD TOKEN VOCABULARY
# =========================================================
# Definition:
# Creates token <-> id mappings.
#
# Arguments:
# final_vocab -> learned vocabulary
# merge_ranks -> learned merges
#
# Functionality:
# - Builds token set
# - Adds merged tokens
# - Adds special tokens
# - Creates mappings
#
# Why Needed?
# Neural networks use numerical IDs.
# =========================================================

# =========================================================
# GET SPECIAL TOKEN IDS
# =========================================================

def get_special_token_ids(
    token_to_id,
    special_tokens=SPECIAL_TOKENS
):

    ids = {}

    for name, token in special_tokens.items():

        ids[name] = token_to_id[token]

    return ids

def build_token_vocab(
    final_vocab,
    merge_ranks,
    special_tokens=SPECIAL_TOKENS
):

    token_set = set()

    # =====================================================
    # ADD TOKENS FROM FINAL VOCAB
    # =====================================================

    for word in final_vocab:
        token_set.update(word)

    # =====================================================
    # ADD INDIVIDUAL CHARACTERS
    # =====================================================

    for word in final_vocab:

        for token in word:

            chars = re.findall(
                r'</w>|.',
                token
            )

            token_set.update(chars)

    # =====================================================
    # ADD MERGED TOKENS
    # =====================================================

    for pair in merge_ranks:

        merged_token = (
            pair[0] + pair[1]
        )

        token_set.add(merged_token)

    # =====================================================
    # FINAL TOKEN LIST
    # =====================================================

    all_tokens = (
        list(special_tokens.values())
        +
        sorted(token_set)
    )

    # =====================================================
    # TOKEN -> ID
    # =====================================================

    token_to_id = {

        token: idx

        for idx, token
        in enumerate(all_tokens)
    }

    # =====================================================
    # ID -> TOKEN
    # =====================================================
    id_to_token = {

        idx: token

        for token, idx
        in token_to_id.items()
    }

    return token_to_id, id_to_token

# =========================================================
# STEP 10: TOKENS -> IDS
# =========================================================
# Definition:
# Converts tokens into numerical IDs.
#
# Arguments:
# tokens -> token list
# token_to_id -> vocabulary mapping
#
# Why Needed?
# Deep learning models consume IDs.
# =========================================================

def tokens_to_ids(
    tokens,
    token_to_id
):

    unk_id = token_to_id["<unk>"]

    return [

        token_to_id.get(
            token,
            unk_id
        )

        for token in tokens
    ]


# =========================================================
# STEP 11: IDS -> TOKENS
# =========================================================
# Definition:
# Converts IDs back into tokens.
#
# Arguments:
# ids -> numerical IDs
# id_to_token -> reverse vocabulary mapping
#
# Why Needed?
# Reconstructs token sequence.
# =========================================================

def ids_to_tokens(
    ids,
    id_to_token
):

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
# Arguments:
# tokens -> token list
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

            text += (
                token.replace(
                    "</w>",
                    ""
                ) + " "
            )

        else:

            text += token

    return text.strip()

# =========================================================
# STEP 13: PAD SEQUENCE
# =========================================================
#
# Definition:
# Makes sequence fixed length.
#
# Example:
#
# [2,10,20,3]
#
# max_length=8
#
# ->
#
# [2,10,20,3,0,0,0,0]
#
# =========================================================

def pad_sequence(
    input_ids,
    max_length,
    pad_token_id
):

    current_length = len(
        input_ids
    )

    if current_length >= max_length:

        return input_ids

    padding_needed = (
        max_length
        -
        current_length
    )

    return (
        input_ids
        +
        [pad_token_id]
        *
        padding_needed
    )

# =========================================================
# STEP 14: TRUNCATE SEQUENCE
# =========================================================
#
# Definition:
# Shortens sequence when it exceeds
# max_length.
#
# Preserves EOS token.
#
# Example:
#
# [2,10,20,30,40,50,3]
#
# max_length=5
#
# ->
#
# [2,10,20,30,3]
#
# =========================================================

def truncate_sequence(
    input_ids,
    max_length,
    eos_token_id=None
):

    if len(input_ids) <= max_length:

        return input_ids

    if eos_token_id is not None:

        return (
            input_ids[
                : max_length - 1
            ]
            +
            [eos_token_id]
        )

    return input_ids[
        : max_length
    ]

# =========================================================
# STEP 15: FULL SENTENCE -> IDS PIPELINE
# =========================================================
# Definition:
# Full tokenizer pipeline.
#
# Arguments:
# sentence -> input text
# merge_ranks -> learned merges
# token_to_id -> token vocabulary
#
# Functionality:
#
# sentence
# ->
# tokens
# ->
# ids
#
# Why Needed?
# Production tokenizers use pipelines.
# =========================================================

def encode_sentence_ids(
    sentence,
    merge_ranks,
    token_to_id
):

    tokens = encode_sentence(
        sentence,
        merge_ranks
    )

    ids = tokens_to_ids(
        tokens,
        token_to_id
    )

    return ids


# =========================================================
# STEP 16: SAVE TOKENIZER
# =========================================================
# Definition:
# Saves tokenizer to disk.
#
# Arguments:
# merge_ranks -> learned merges
# token_to_id -> vocabulary mapping
# id_to_token -> reverse mapping
#
# Functionality:
# - Converts tuple keys to strings
# - Stores tokenizer data as JSON
#
# Why Needed?
# Tokenizers are trained once and reused.
# =========================================================

def save_tokenizer(
    merge_ranks,
    token_to_id,
    id_to_token,
    filepath="tokenizer.json"
):

    # =====================================================
    # CONVERT TUPLE KEYS -> STRINGS
    # =====================================================

    serializable_merges = {

        " ".join(pair): rank

        for pair, rank
        in merge_ranks.items()
    }

    # =====================================================
    # BUILD SAVE OBJECT
    # =====================================================

    tokenizer_data = {

        "merge_ranks":
        serializable_merges,

        "token_to_id":
        token_to_id,

        "id_to_token":
        id_to_token
    }

    # =====================================================
    # SAVE JSON FILE
    # =====================================================

    with open(filepath, "w") as f:

        json.dump(
            tokenizer_data,
            f,
            indent=4
        )

    print(
        f"\nTokenizer saved to {filepath}"
    )


# =========================================================
# STEP 17: LOAD TOKENIZER
# =========================================================
# Definition:
# Loads tokenizer from disk.
#
# Arguments:
# filepath -> tokenizer file
#
# Functionality:
# - Loads JSON
# - Restores merge ranks
# - Restores vocabulary mappings
#
# Why Needed?
# Allows tokenizer reuse without retraining.
# =========================================================

def load_tokenizer(
    filepath="tokenizer.json"
):

    # =====================================================
    # LOAD JSON FILE
    # =====================================================

    with open(filepath, "r") as f:

        tokenizer_data = json.load(f)

    # =====================================================
    # RESTORE MERGE RANKS
    # =====================================================

    merge_ranks = {

        tuple(pair.split()): rank

        for pair, rank
        in tokenizer_data[
            "merge_ranks"
        ].items()
    }

    # =====================================================
    # RESTORE TOKEN MAPPINGS
    # =====================================================

    token_to_id = tokenizer_data[
        "token_to_id"
    ]

    # JSON converts integer keys to strings
    id_to_token = {

        int(idx): token

        for idx, token
        in tokenizer_data[
            "id_to_token"
        ].items()
    }

    print(
        f"\nTokenizer loaded from {filepath}"
    )

    return (
        merge_ranks,
        token_to_id,
        id_to_token
    )


# =========================================================
# TRAINING CORPUS
# =========================================================

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


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # TRAIN BPE TOKENIZER
    # =====================================================

    merge_ranks, final_vocab = train_bpe(
        list_corpus
    )

    print("\nMERGE RANKS:\n")

    print(merge_ranks)

    # =====================================================
    # BUILD TOKEN VOCABULARY
    # =====================================================

    token_to_id, id_to_token = (
        build_token_vocab(
            final_vocab,
            merge_ranks
        )
    )

    print("\nTOKEN TO ID:\n")

    print(token_to_id)

    # =====================================================
    # SAVE TOKENIZER
    # =====================================================

    save_tokenizer(
        merge_ranks,
        token_to_id,
        id_to_token
    )

    # =====================================================
    # LOAD TOKENIZER
    # =====================================================

    merge_ranks, token_to_id, id_to_token = (
        load_tokenizer()
    )

    # =====================================================
    # ENCODE SENTENCE
    # =====================================================

    sentence = (
        "Hello! I am at the lowest point on earth."
    )

    tokens = encode_sentence(
        sentence,
        merge_ranks
    )

    print("\nENCODED TOKENS:\n")

    print(tokens)

    # =====================================================
    # TOKENS -> IDS
    # =====================================================

    ids = tokens_to_ids(
        tokens,
        token_to_id
    )

    print("\nTOKEN IDS:\n")

    print(ids)

    # =====================================================
    # IDS -> TOKENS
    # =====================================================

    print("\nORIGINAL LENGTH:\n")

    print(len(ids))


    padded_ids = pad_sequence(

        input_ids=ids,

        max_length=50,

        pad_token_id=token_to_id["<pad>"]
    )

    print("\nPADDED LENGTH:\n")

    print(len(padded_ids))

    print("\nPADDED IDS:\n")

    print(padded_ids)

    truncated_ids = truncate_sequence(

    input_ids=ids,

    max_length=15,

    eos_token_id=token_to_id["<eos>"]
    )

    print("\nTRUNCATED LENGTH:\n")

    print(len(truncated_ids))

    print("\nTRUNCATED IDS:\n")

    print(truncated_ids)

    recovered_tokens = ids_to_tokens(
        ids,
        id_to_token
    )

    print("\nRECOVERED TOKENS:\n")

    print(recovered_tokens)

    # =====================================================
    # DECODE BACK
    # =====================================================

    decoded_text = decode_tokens(
        recovered_tokens
    )

    print("\nDECODED TEXT:\n")

    print(decoded_text)