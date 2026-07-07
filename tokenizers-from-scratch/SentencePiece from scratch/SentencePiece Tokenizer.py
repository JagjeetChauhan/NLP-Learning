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

vocab = ["I love NLP","I like New York","I love CV"]
updated_vocab = Text_preprocess(vocab)
corpus = [list(sentence) for sentence in updated_vocab]
print(corpus)

Initial_vocab_list = Initial_vocab(updated_vocab)
pair_stats = pair_statistics(updated_vocab)
best_pair = best_pair_in_vocab(pair_stats)

new_corpus = merge_pair(corpus, best_pair)

print()
print(Initial_vocab_list)

print()
print(pair_stats)

print()
print(best_pair)
print(new_corpus)

target_vocab_size = 25
while len(vocab) < target_vocab_size:

    pairs = pair_statistics(corpus)

    best_pair = best_pair_in_vocab(pairs)

    corpus = merge_pair(corpus, best_pair)

    vocab.append(best_pair[0] + best_pair[1])

    print(f"Merge: {best_pair}")
    print(corpus)
    print("-" * 40)

print()
print(F"Final Corpus: {corpus}")

print()
print(build_token_vocab(corpus))