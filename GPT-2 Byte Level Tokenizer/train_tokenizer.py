from collections import defaultdict
import json

from config import NUM_MERGES, TOKENIZER_TYPE, TOKENIZER_VERSION
from tokenizer_utils import text_to_bytes, get_pairs


# =========================================================
# CORPUS (TRAINING DATA)
# =========================================================

CORPUS = [
    "Hello world!",
    "GPT-2 tokenizer test 🙂",
    "Numbers 123 and symbols #@$",
    "This is a byte level BPE tokenizer",
    "NLP is fun",
    "नमस्ते दुनिया",
    "你好世界",
    "Email test: test@example.com",
    "URL: https://openai.com",
    "🔥🚀 emoji test",
    "Machine learning is powerful"
]


# =========================================================
# BUILD VOCAB (BYTE LEVEL)
# =========================================================

def get_vocab(corpus):

    vocab = defaultdict(int)

    for text in corpus:
        tokens = tuple(text_to_bytes(text))
        vocab[tokens] += 1

    return vocab


# =========================================================
# MERGE VOCAB
# =========================================================

def merge_vocab(vocab, pair):

    new_vocab = {}

    for word, freq in vocab.items():

        new_word = []
        i = 0

        while i < len(word):

            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_vocab[tuple(new_word)] = new_vocab.get(tuple(new_word), 0) + freq

    return new_vocab


# =========================================================
# TRAIN BPE
# =========================================================

def train_bpe(corpus, num_merges):

    vocab = get_vocab(corpus)
    merges = {}

    for i in range(num_merges):

        pair_counts = defaultdict(int)

        for word, freq in vocab.items():
            for p in get_pairs(word):
                pair_counts[p] += freq

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=pair_counts.get)

        vocab = merge_vocab(vocab, best_pair)
        merges[best_pair] = i

        print(f"merge {i}: {best_pair}")

    return merges, vocab


# =========================================================
# SAVE TOKENIZER
# =========================================================

def save_tokenizer(merges, vocab, path="tokenizer.json"):

    vocab_set = set()

    for word in vocab:
        vocab_set.update(word)

    data = {
        "version": TOKENIZER_VERSION,
        "type": TOKENIZER_TYPE,
        "vocab_size": len(vocab_set),
        "merges": {
            str(k): v for k, v in merges.items()
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nTokenizer saved to {path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n=== TRAINING BYTE LEVEL BPE TOKENIZER STARTED===\n")

    merges, vocab = train_bpe(CORPUS, NUM_MERGES)

    save_tokenizer(merges, vocab)

    print("\n=== TRAINING BYTE LEVEL BPE TOKENIZER ENDED===\n")