from config import *
from tokenizer_utils import *

class WordPieceTokenizer:

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        lowercase=LOWERCASE,
        special_tokens=SPECIAL_TOKENS
    ):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.special_tokens = special_tokens

        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, corpus):

        if self.lowercase:
            corpus = [sentence.lower() for sentence in corpus]

        # Step 1
        words = extract_words(corpus)

        # Step 2
        word_frequencies = build_word_frequencies(words)

        # Step 3
        splits = build_initial_splits(word_frequencies)

        # Initial vocabulary
        vocab = set(self.special_tokens)

        for tokens in splits.values():
            vocab.update(tokens)

        # Merge until desired vocabulary size
        while len(vocab) < self.vocab_size:

            token_frequencies = compute_token_frequencies(
                splits,
                word_frequencies
            )

            pair_frequencies = compute_pair_frequencies(
                splits,
                word_frequencies
            )

            if not pair_frequencies:
                break

            scores = compute_scores(
                pair_frequencies,
                token_frequencies
            )

            if not scores:
                break

            best_pair, _ = find_best_pair(scores)

            new_token = merge_best_pair(best_pair)

            vocab.add(new_token)

            splits = merge_pair(best_pair, splits)

        # Store vocabulary
        self.vocab = sorted(vocab)

        # Build lookup tables
        self.token_to_id = build_token_to_id(self.vocab)
        self.id_to_token = build_id_to_token(self.token_to_id)

        return self.token_to_id

    def encode(self, text):

        if self.lowercase:
            text = text.lower()

        words = text.split()

        ids = []

        for word in words:
            ids.extend(
                encode_to_ids(
                    word,
                    self.vocab,
                    self.token_to_id
                )
            )

        return ids

    def decode(self, ids):

        return decode_ids(
            ids,
            self.id_to_token
        )

    def save(self, save_dir):

        save_tokenizer(
            self.token_to_id,
            self.special_tokens,
            save_dir
        )

    def load(self, save_dir):

        self.token_to_id, self.special_tokens = load_tokenizer(save_dir)

        self.id_to_token = build_id_to_token(
            self.token_to_id
        )

        self.vocab = list(
            self.token_to_id.keys()
        )