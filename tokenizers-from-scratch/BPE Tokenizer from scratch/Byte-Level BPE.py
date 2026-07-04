# =========================================================
# TEXT TO BYTES
# =========================================================
#
# Converts text into UTF-8 byte values.
#
# Example:
#
# hello
#
# ->
#
# [104,101,108,108,111]
#
# =========================================================

def text_to_bytes(text):

    return list(
        text.encode("utf-8")
    )

# =========================================================
# BYTES TO TEXT
# =========================================================
#
# Converts byte values back into text.
#
# Example:
#
# [104,101,108,108,111]
#
# ->
#
# hello
#
# =========================================================

def bytes_to_text(byte_values):

    return bytes(
        byte_values
    ).decode(
        "utf-8"
    )

# =========================================================
# WORD TO BYTE TOKENS
# =========================================================
#
# Converts word into UTF-8 byte tokens.
#
# Example:
#
# low
#
# ->
#
# ['108','111','119']
#
# =========================================================

def word_to_byte_tokens(word):

    byte_values = list(
        word.encode("utf-8")
    )

    byte_tokens = [

        str(byte)

        for byte in byte_values
    ]

    return byte_tokens

# =========================================================
# BUILD BYTE VOCABULARY
# =========================================================
#
# Converts corpus into byte-level
# vocabulary representation.
#
# =========================================================


def build_byte_vocab(corpus):

    vocab = defaultdict(int)

    for word in corpus:

        tokens = (
            word_to_byte_tokens(word)
            +
            ["</w>"]
        )

        vocab[
            tuple(tokens)
        ] += 1

    return vocab

# =========================================================
# GET BYTE PAIR COUNTS
# =========================================================
#
# Counts frequency of adjacent
# byte token pairs.
#
# =========================================================

from collections import defaultdict


def get_byte_pair_counts(
    vocab
):

    pair_counts = defaultdict(
        int
    )

    for word, freq in vocab.items():

        for i in range(
            len(word) - 1
        ):

            pair = (

                word[i],

                word[i + 1]
            )

            pair_counts[
                pair
            ] += freq

    return pair_counts

# =========================================================
# MERGE BYTE PAIR
# =========================================================
#
# Replaces most frequent pair
# with a new byte-level token.
#
# =========================================================

def merge_byte_pair(
    pair,
    vocab,
    new_token
):

    new_vocab = {}

    for word, freq in vocab.items():

        word = list(word)

        i = 0

        merged_word = []

        while i < len(word):

            if (

                i < len(word) - 1

                and

                word[i] == pair[0]

                and

                word[i + 1] == pair[1]

            ):

                merged_word.append(
                    str(new_token)
                )

                i += 2

            else:

                merged_word.append(
                    word[i]
                )

                i += 1

        new_vocab[
            tuple(merged_word)
        ] = freq

    return new_vocab

# =========================================================
# TRAIN BYTE LEVEL BPE
# =========================================================
#
# Learns byte-level merge rules.
#
# =========================================================

def train_byte_bpe(
    corpus,
    num_merges
):

    vocab = build_byte_vocab(
        corpus
    )

    merge_vocab = {}

    merge_ranks = {}

    next_token_id = 256


    for rank in range(
        num_merges
    ):

        pair_counts = (
            get_byte_pair_counts(
                vocab
            )
        )


        if not pair_counts:

            break


        best_pair = max(

            pair_counts,

            key=pair_counts.get
        )


        print(

            f"Step {rank+1}: "

            f"Merging "

            f"{best_pair}"

            f" -> "

            f"{next_token_id}"
        )


        merge_vocab[
            next_token_id
        ] = best_pair


        merge_ranks[
            best_pair
        ] = rank


        vocab = merge_byte_pair(

            best_pair,

            vocab,

            next_token_id
        )


        next_token_id += 1


    return (

        vocab,

        merge_vocab,

        merge_ranks
    )

# =========================================================
# GET ADJACENT PAIRS
# =========================================================

def get_adjacent_pairs(tokens):

    pairs = set()

    for i in range(
        len(tokens) - 1
    ):

        pairs.add(

            (
                tokens[i],

                tokens[i+1]
            )
        )

    return pairs

# =========================================================
# GET ADJACENT PAIRS
# =========================================================

def get_adjacent_pairs(tokens):

    pairs = set()

    for i in range(
        len(tokens) - 1
    ):

        pairs.add(

            (
                tokens[i],

                tokens[i+1]
            )
        )

    return pairs

# =========================================================
# ENCODE BYTE TOKENS
# =========================================================
#
# Applies learned merge ranks.
#
# =========================================================

def encode_byte_tokens(

    byte_tokens,

    merge_ranks

):

    tokens = byte_tokens[:]


    while True:

        pairs = get_adjacent_pairs(
            tokens
        )


        candidate_pairs = {

            pair:
            merge_ranks[pair]

            for pair in pairs

            if pair in merge_ranks
        }


        if not candidate_pairs:

            break


        best_pair = min(

            candidate_pairs,

            key=candidate_pairs.get
        )


        rank = merge_ranks[
            best_pair
        ]


        print(

            f"Merging "

            f"{best_pair}"

            f" rank={rank}"
        )


        new_token = str(
            256 + rank
        )


        new_tokens = []

        i = 0

        while i < len(tokens):

            if (

                i < len(tokens)-1

                and

                (
                    tokens[i],

                    tokens[i+1]

                ) == best_pair

            ):

                new_tokens.append(
                    new_token
                )

                i += 2

            else:

                new_tokens.append(
                    tokens[i]
                )

                i += 1


        tokens = new_tokens


    return tokens

# =========================================================
# EXPAND TOKEN
# =========================================================
#
# Recursively expands merged tokens
# back into byte tokens.
#
# =========================================================

def expand_token(
    token,
    merge_vocab
):

    try:
        token_int = int(token)

    except ValueError:
        return [token]

    if token_int not in merge_vocab:
        return [token]

    left, right = merge_vocab[token_int]

    return (
        expand_token(left, merge_vocab)
        +
        expand_token(right, merge_vocab)
    )


    left, right = (

        merge_vocab[
            token_int
        ]
    )


    return (

        expand_token(
            left,
            merge_vocab
        )

        +

        expand_token(
            right,
            merge_vocab
        )
    )

# =========================================================
# DECODE BYTE TOKENS
# =========================================================

def decode_byte_tokens(

    encoded_tokens,

    merge_vocab

):

    byte_tokens = []


    for token in encoded_tokens:

        expanded = expand_token(

            token,

            merge_vocab
        )

        byte_tokens.extend(
            expanded
        )


    return byte_tokens

#=======================================Testing======================================================
# corpus = [

#     "hello",

#     "hello",

#     "help"
# ]

# byte_vocab = build_byte_vocab(
#     corpus
# )

# print("\nCorpus:")
# print(corpus)

# for word, freq in byte_vocab.items():

#     print(
#         word,
#         "->",
#         freq
#     )

# pair_counts = (
#     get_byte_pair_counts(
#         byte_vocab
#     )
# )

# for pair, count in sorted(
#     pair_counts.items(),
#     key=lambda x: x[1],
#     reverse=True
# ):
#     print("\nPair Count:")
#     print(
#         pair,
#         "->",
#         count
#     )

# print("\nTraining Byte BPE: ")
# final_vocab, merge_vocab, merge_ranks = (

#     train_byte_bpe(

#         corpus,

#         num_merges=5
#     )
# )

# print(
#     "\nMERGE VOCAB:\n"
# )

# for token_id, pair in merge_vocab.items():

#     print(
#         token_id,
#         "->",
#         pair
#     )

# print(
#     "\nMERGE RANKS:\n"
# )

# print(
#     merge_ranks
# )

# tokens = (
#     word_to_byte_tokens("hello")
#     + ["</w>"]
# )

# print("\nEcoded Tokens:")
# encoded = encode_byte_tokens(

#     tokens,

#     merge_ranks
# )

# print(encoded)

# The First Full Round Trip

corpus = [
    "hello",
    "hello",
    "help"
]

final_vocab, merge_vocab, merge_ranks = train_byte_bpe(
    corpus,
    num_merges=5
)

text = "hello"

byte_tokens = (
    word_to_byte_tokens(text)
    + ["</w>"]
)

encoded_tokens = encode_byte_tokens(
    byte_tokens,
    merge_ranks
)

decoded_bytes = decode_byte_tokens(
    encoded_tokens,
    merge_vocab
)

decoded_bytes = [
    int(x)
    for x in decoded_bytes
    if x != "</w>"
]

decoded_text = bytes_to_text(
    decoded_bytes
)

print("Original:", text)
print("Encoded:", encoded_tokens)
print("Decoded:", decoded_text)