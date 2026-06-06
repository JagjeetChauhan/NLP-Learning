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

#=======================================Testing======================================================
corpus = [

    "low",

    "lowest",

    "newer"
]

byte_vocab = build_byte_vocab(
    corpus
)

for word, freq in byte_vocab.items():

    print(
        word,
        "->",
        freq
    )

pair_counts = (
    get_byte_pair_counts(
        byte_vocab
    )
)

for pair, count in sorted(
    pair_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print("\nPair Count:")
    print(
        pair,
        "->",
        count
    )