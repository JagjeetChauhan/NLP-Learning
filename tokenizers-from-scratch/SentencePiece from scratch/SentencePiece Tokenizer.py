""" Stage 1 — Text preprocessing
Replace spaces with ▁
Treat each sentence as one sequence """

def Text_preprocess_test(sentence):
    result = ""
    for line in sentence:
        for ch in line:
            if ch ==' ':
                result += '_'
            else:
                result += ch
        result += '\n'
        
    return result

def Text_preprocess(sentences):
    result = []

    for sentence in sentences:
        result.append(sentence.replace(" ", "▁"))

    return result

"""
Stage 2 — Initial vocabulary
Build a character vocabulary (including ▁)
"""
def Initial_vocab(sentence_list):
    Initial_vocab_list = []
    for sentence in sentence_list:
        for ch in sentence:
            if ch not in Initial_vocab_list:
                Initial_vocab_list.append(ch)

    return Initial_vocab_list

"""
Stage 3 — Pair statistics
Count adjacent symbol pairs over the full sentence
"""
from collections import defaultdict

def pair_statistics(sentences):
    pair_count = defaultdict(int)

    for sentence in sentences:
        chars = list(sentence)

        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            pair_count[pair] += 1

    return dict(pair_count)

"""
Stage 4 — BPE merges
Merge the most frequent pair repeatedly until reaching the target vocabulary size
"""
def best_pair_in_vocab(pairs_list):
    best_pair = max(pairs_list, key=pairs_list.get)
    return best_pair

def merge_pair(corpus, pair):
    """
    corpus: List[List[str]]
    pair: ('a', 'm')

    returns updated corpus
    """

    merged_corpus = []
    new_token = pair[0] + pair[1]

    for sentence in corpus:
        new_sentence = []

        i = 0
        while i < len(sentence):

            if (
                i < len(sentence) - 1
                and sentence[i] == pair[0]
                and sentence[i + 1] == pair[1]
            ):
                new_sentence.append(new_token)
                i += 2
            else:
                new_sentence.append(sentence[i])
                i += 1

        merged_corpus.append(new_sentence)

    return merged_corpus

"""
Stage 5 — Encoding
Encode new sentences using the learned merges
"""
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

def encode_word(word, merges):

    tokens = list(word)

    for pair in merges:

        new_tokens = []

        i = 0

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

def encode_sentence(sentence, merges):

    sentence = sentence.replace(" ", "▁")

    return encode_word(sentence, merges)

def token_to_ids(tokens, token_to_id):
    unk_id = token_to_id["<unk>"]
    return [

        token_to_id.get(token, unk_id)

        for token in tokens
    ]

"""
Stage 6 — Decoding
Join pieces and convert ▁ back to spaces
"""

sentences = [
    "I love natural language processing",
    "I love machine learning",
    "I like deep learning",
    "She likes machine learning",
    "He loves natural language",
    "Natural language is interesting",
    "Deep learning is powerful",
    "Machine learning uses data",
    "I enjoy reading books",
    "She enjoys reading articles",
    "The cat sits on the mat",
    "The dog sits near the cat",
    "New York is a big city",
    "I visited New York last year",
    "Artificial intelligence is changing the world"
]

processed = Text_preprocess(sentences)
corpus = [list(s) for s in processed]

initial_vocab = Initial_vocab(processed)
current_vocab_size = len(initial_vocab)

merges = []

target_vocab_size = 60
while current_vocab_size < target_vocab_size:
    pairs = pair_statistics(corpus)
    best_pair = best_pair_in_vocab(pairs)
    merges.append(best_pair)
    corpus = merge_pair(corpus, best_pair)
    current_vocab_size += 1

    print(f"Merge: {best_pair}")
    print(corpus)
    print("-" * 40)

print()
print(f"Final Corpus: {corpus}")

print(f"\nFinal Merges: {merges}")

print()
token_to_id, id_to_token = build_token_vocab(corpus)

print(f"\nToken to Ids: {token_to_id}")
print(f"\nIds to Token: {id_to_token}")

sentence = "Hello! I am at the lowest point on earth."

tokens = encode_sentence(sentence, merges)

print("\nENCODED TOKENS:\n")

print(tokens)