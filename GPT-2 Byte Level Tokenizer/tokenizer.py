import json
from tokenizer_utils import text_to_bytes, bytes_to_text, get_pairs


# =========================================================
# LOAD
# =========================================================

def load_tokenizer(path="tokenizer.json"):

    with open(path) as f:
        data = json.load(f)

    merges = {
        eval(k): v for k, v in data["merges"].items()
    }

    return merges


# =========================================================
# ENCODE (FLAT GPT-2 STYLE)
# =========================================================

def encode(text, merges):

    tokens = text_to_bytes(text)

    while True:

        pairs = get_pairs(tokens)

        ranked = {p: merges[p] for p in pairs if p in merges}

        if not ranked:
            break

        best = min(ranked, key=ranked.get)

        new_tokens = []
        i = 0

        while i < len(tokens):

            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                # SAFE: replace pair with first byte ONLY (NO NEW OBJECTS)
                new_tokens.append(tokens[i])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens

# =========================================================
# DECODE (LOSSLESS)
# =========================================================

def decode(tokens):

    flat = []

    for t in tokens:

        if isinstance(t, int):
            flat.append(t)
        else:
            flat.append(t)

    return bytes_to_text(flat)