import re
import json
from config import SPECIAL_TOKENS
from tokenizer_utils import pad_sequence, truncate_sequence, create_attention_mask

# =========================================================
# STEP 1: PREPROCESS TEXT
# =========================================================

def preprocess_text(sentence):

    sentence = sentence.lower()

    sentence = re.sub(r'([.,!?])', r' \1 ', sentence)

    sentence = re.sub(r'\s+', ' ', sentence).strip()

    words = sentence.split()

    return words


# =========================================================
# STEP 2: GET ADJACENT PAIRS
# =========================================================

def get_adjacent_pairs(tokens):

    pairs = set()

    for i in range(len(tokens) - 1):

        pairs.add(
            (tokens[i], tokens[i + 1])
        )

    return pairs


# =========================================================
# STEP 3: LOAD TOKENIZER
# =========================================================

def load_tokenizer(filepath="tokenizer.json"):

    with open(filepath, "r") as f:

        tokenizer_data = json.load(f)

    special_tokens = tokenizer_data[
        "special_tokens"
    ]
    merge_ranks = {

        tuple(pair.split()): rank

        for pair, rank in tokenizer_data[
            "merge_ranks"
        ].items()
    }

    token_to_id = tokenizer_data[
        "token_to_id"
    ]

    id_to_token = {

        int(idx): token

        for idx, token in tokenizer_data[
            "id_to_token"
        ].items()
    }

    print(f"\nTokenizer loaded from {filepath}")

    return (
        merge_ranks,
        token_to_id,
        id_to_token,
        special_tokens
    )


# =========================================================
# STEP 4: ENCODE WORD
# =========================================================

def encode(word, merge_ranks):

    tokens = list(word) + ['</w>']

    while True:

        pairs = get_adjacent_pairs(tokens)

        candidate_pairs = {

            pair: merge_ranks[pair]

            for pair in pairs

            if pair in merge_ranks
        }

        if not candidate_pairs:
            break

        best_pair = min(
            candidate_pairs,
            key=candidate_pairs.get
        )

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
# STEP 5: ENCODE SENTENCE
# =========================================================

def encode_sentence(sentence, merge_ranks, special_tokens, add_special_tokens=True):

    all_tokens = []

    words = preprocess_text(sentence)

    for word in words:

        word_tokens = encode(word, merge_ranks)

        all_tokens.extend(word_tokens)
    
    if add_special_tokens:

        all_tokens = (
            [
                special_tokens[
                    "bos_token"
                ]
            ]
            +
            all_tokens
            +
            [
                special_tokens[
                    "eos_token"
                ]
            ]
        )

    return all_tokens


# =========================================================
# STEP 6: TOKENS -> IDS
# =========================================================

def tokens_to_ids(tokens, token_to_id):

    unk_id = token_to_id["<unk>"]

    return [

        token_to_id.get(token, unk_id)

        for token in tokens
    ]


# =========================================================
# STEP 7: IDS -> TOKENS
# =========================================================

def ids_to_tokens(ids, id_to_token):

    return [

        id_to_token[i]

        for i in ids
    ]


# =========================================================
# STEP 8: DECODE TOKENS
# =========================================================

def decode_tokens(
    tokens
):

    text = ""

    ignore_tokens = {

        "<pad>",
        "<bos>",
        "<eos>"
    }


    for token in tokens:


        if token in ignore_tokens:

            continue


        if token.endswith("</w>"):

            text += (
                token.replace(
                    "</w>",
                    ""
                )
                +
                " "
            )

        else:

            text += token


    return text.strip()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    merge_ranks, token_to_id, id_to_token, special_tokens = (
        load_tokenizer()
    )

    sentence = (
        "Hello! I am at the lowest point on earth. "
    "This tokenizer is learning truncation."
    )

    tokens = encode_sentence(
        sentence,
        merge_ranks,
        special_tokens
    )

    print("\nTOKENS:\n")

    print(tokens)

    ids = tokens_to_ids(
    tokens,
    token_to_id
    )

    print(
        "\nORIGINAL LENGTH:\n"
    )

    print(
        len(ids)
    )

    print(
        "\nORIGINAL IDS:\n"
    )

    print(ids)


    # =========================================================
    # TRUNCATION
    # =========================================================

    ids = truncate_sequence(

        input_ids=ids,

        max_length=15,

        eos_token_id=token_to_id[
            "<eos>"
        ]
    )

    print(
        "\nLENGTH AFTER TRUNCATION:\n"
    )

    print(
        len(ids)
    )

    print(
        "\nTRUNCATED IDS:\n"
    )

    print(ids)


    # =========================================================
    # PADDING
    # =========================================================

    ids = pad_sequence(

        input_ids=ids,

        max_length=20,

        pad_token_id=token_to_id[
            "<pad>"
        ]
    )

    print(
        "\nFINAL LENGTH:\n"
    )

    print(
        len(ids)
    )

    print(
        "\nFINAL IDS:\n"
    )

    print(ids)

    # =========================================================
    # ATTENTION MASK
    # =========================================================

    attention_mask = create_attention_mask(

    input_ids=ids,

    pad_token_id=token_to_id[
        "<pad>"
    ]
    )

    print(
    "\nATTENTION MASK:\n"
    )

    print(
        attention_mask
    )

    recovered_tokens = ids_to_tokens(
        ids,
        id_to_token
    )

    print("\nRECOVERED TOKENS:\n")

    print(recovered_tokens)

    decoded_text = decode_tokens(
        recovered_tokens
    )

    print("\nDECODED TEXT:\n")

    print(decoded_text)