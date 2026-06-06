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

#=======================================Testing======================================================
corpus = [

    "hello",

    "hello",

    "help"
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

print("\nTraining Byte BPE: ")
final_vocab, merge_vocab, merge_ranks = (

    train_byte_bpe(

        corpus,

        num_merges=5
    )
)

print(
    "\nMERGE VOCAB:\n"
)

for token_id, pair in merge_vocab.items():

    print(
        token_id,
        "->",
        pair
    )

print(
    "\nMERGE RANKS:\n"
)

print(
    merge_ranks
)