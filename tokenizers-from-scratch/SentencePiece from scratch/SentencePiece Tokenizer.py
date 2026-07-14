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
        tokens = list(sentence)

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
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
def build_token_vocab(corpus, special_tokens=None):

    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>"]

    token_set = set()

    for sentence in corpus:
        for token in sentence:
            token_set.add(token)

    all_tokens = special_tokens + sorted(token_set)

    token_to_id = {t:i for i,t in enumerate(all_tokens)}
    id_to_token = {i:t for t,i in token_to_id.items()}

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

def ids_to_tokens(ids, id_to_token):

    return [

        id_to_token[i]

        for i in ids
    ]

"""
Stage 6 — Decoding
Join pieces and convert ▁ back to spaces
"""
def decode_tokens(tokens):
    return "".join(tokens).replace("▁", " ")



sentences = [
    "Natural language processing enables computers to understand human language.",
    "Machine learning algorithms improve with experience.",
    "Deep learning is a subset of machine learning.",
    "Artificial intelligence is transforming many industries.",
    "Large language models generate human like text.",
    "Neural networks are inspired by the human brain.",
    "Data science combines statistics programming and domain knowledge.",
    "Python is a popular programming language.",
    "Tokenization is the first step of many NLP pipelines.",
    "Byte Pair Encoding builds subword vocabularies.",
    "Language models predict the next token.",
    "Students enjoy learning artificial intelligence.",
    "Researchers publish papers on natural language processing.",
    "Computers process millions of words every day.",
    "Reading books improves vocabulary and comprehension.",
    "The weather is pleasant today.",
    "The dog chased the cat across the garden.",
    "The children played football after school.",
    "She enjoys reading novels every evening.",
    "He studies computer science at university.",
] * 200

processed = Text_preprocess(sentences)
corpus = [list(s) for s in processed]

initial_vocab = Initial_vocab(processed)
current_vocab_size = len(initial_vocab)

merges = []

def get_vocab(corpus):
    vocab = set()

    for sentence in corpus:
        vocab.update(sentence)

    return vocab

target_vocab_size = 200
vocab = get_vocab(sentences)

while len(vocab) < target_vocab_size:

    pairs = pair_statistics(corpus)

    if not pairs:
        break

    best_pair = max(pairs, key=pairs.get)

    merges.append(best_pair)

    corpus = merge_pair(corpus, best_pair)

    vocab = get_vocab(corpus)

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

ids = token_to_ids(tokens, token_to_id)

print("\nTOKEN IDS:\n")

print(ids)

recovered_tokens = ids_to_tokens(ids, id_to_token)
print("\nTOKENS:\n")
print(recovered_tokens)

decoded_text = decode_tokens(recovered_tokens)

print("\nDECODED TEXT:\n")

print(decoded_text)