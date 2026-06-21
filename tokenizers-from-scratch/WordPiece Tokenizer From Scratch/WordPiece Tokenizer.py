# corpus = [
#     "low lower lowest",
#     "new newer newest",
# ]
# def word_split(corpus):
#     list_of_words = []
#     for sentence in corpus:
#         words = sentence.split()
#         list_of_words.append(words)
#     return list_of_words

# lists = word_split(corpus)
# print(lists)
# print("\n")

# # Word frequencies
# word_freq = {}

# for count_words in lists:
#     for count_word in count_words:
#         if count_word in word_freq:
#             word_freq[count_word] += 1
#         else:
#             word_freq[count_word] = 1

# print(word_freq)
# print("\n")

# # Split words into WordPiece-style characters
# splits = {}

# for word_to_chr in word_freq:
#     splits[word_to_chr] = [word_to_chr[0]] + [f"##{c}" for c in word_to_chr[1:]]

# print(splits)
# print("\n")

# # Find adjacent pairs
# pairs = []

# for word, tokens in splits.items():
#     print(f"\n{word}: {tokens}")

#     for i in range(len(tokens) - 1):
#         pair = (tokens[i], tokens[i+1])
#         pairs.append(pair)
#         print(pair)

# print("\nAll pairs:")
# print(pairs)

# # Pair Frequency
# pair_freq = {}

# for pair in pairs:
#     if pair in pair_freq:
#         pair_freq[pair] += 1
#     else:
#         pair_freq[pair] = 1

# print("\nPair Frequency:")
# print(pair_freq)

# # Words in the Corpus
# word_group = []

# for group in lists:
#     for word_in_group in group:
#         word_group.append(word_in_group)

# print("\nList of Words in the Corpus")
# print(word_group)

# # List of Chr in the Corpus
# chr_freq = {}

# for single_word in word_group:
#     for letter in single_word:
#         if letter in chr_freq:
#             chr_freq[letter] += 1
#         else:
#             chr_freq[letter] = 1

# print("\nCharacter Frequency")
# print(chr_freq)

# # Calculate the score

# def cal_score(pair_frequency, chr_frequency):
#     scores = {}
#     for (a,b), pair_count in pair_frequency.items():
#         a_clean = a.replace('##', '')
#         b_clean = b.replace('##', '')

#         scores[(a, b)] = round(pair_count / (chr_frequency[a_clean] * chr_frequency[b_clean]), 2)
#     return scores

# final_score = cal_score(pair_freq, chr_freq)
# print(final_score)

# best_pair = max(final_score, key=final_score.get)
# best_score = final_score[best_pair]

# print("Best Pair:", best_pair)
# print("Best Score:", best_score)

# ------------------------------------------------------------------------------------------------------------------------------

# Version 2 better Architecture:
corpus = [
    "low lower lowest",
    "new newer newest",
]

# Step 1: Extract Words
def extract_words(corpus):
    words = []

    for sentence in corpus:
        words.extend(sentence.split())

    return words

# Step 2: Word Frequencies
def build_word_frequencies(words):

    word_frequencies = {}

    for word in words:
        word_frequencies[word] = (
            word_frequencies.get(word, 0) + 1
        )

    return word_frequencies

# Step 3: Initial WordPiece Splits
def build_initial_splits(word_frequencies):

    splits = {}

    for word in word_frequencies:

        splits[word] = (
            [word[0]]
            + [f"##{char}" for char in word[1:]]
        )

    return splits

# Step 4: Token Frequencies
def compute_token_frequencies(
    splits,
    word_frequencies
):

    token_frequencies = {}

    for word, tokens in splits.items():

        word_count = word_frequencies[word]

        for token in tokens:

            token_frequencies[token] = (
                token_frequencies.get(token, 0)
                + word_count
            )

    return token_frequencies

# Step 5: Pair Frequencies
def compute_pair_frequencies(
    splits,
    word_frequencies
):

    pair_frequencies = {}

    for word, tokens in splits.items():

        word_count = word_frequencies[word]

        for i in range(len(tokens) - 1):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            pair_frequencies[pair] = (
                pair_frequencies.get(pair, 0)
                + word_count
            )

    return pair_frequencies

# Step 6: WordPiece Scores
def compute_scores(
    pair_frequencies,
    token_frequencies
):

    scores = {}

    for pair, pair_count in pair_frequencies.items():

        left_token, right_token = pair

        score = (
            pair_count
            /
            (
                token_frequencies[left_token]
                *
                token_frequencies[right_token]
            )
        )

        scores[pair] = score

    return scores

# Step 7: Best Pair
def find_best_pair(scores):

    best_pair = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_pair]

    return best_pair, best_score

# Step 8: Merge Pair
def merge_pair(best_pair, splits):
    a, b = best_pair

    for word in splits:
        tokens = splits[word]
        new_tokens = []

        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == a
                and tokens[i+1] == b
            ):
                merged = a + b.replace("##", "")
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        splits[word] = new_tokens
    return splits

# Driver Code
words = extract_words(corpus)

word_frequencies = (
    build_word_frequencies(words)
)

splits = (
    build_initial_splits(word_frequencies)
)

token_frequencies = (
    compute_token_frequencies(
        splits,
        word_frequencies
    )
)

pair_frequencies = (
    compute_pair_frequencies(
        splits,
        word_frequencies
    )
)

scores = compute_scores(
    pair_frequencies,
    token_frequencies
)

best_pair, best_score = (
    find_best_pair(scores)
)

print("Word Frequencies")
print(word_frequencies)

print("\nSplits")
print(splits)

print("\nToken Frequencies")
print(token_frequencies)

print("\nPair Frequencies")
print(pair_frequencies)

print("\nScores")
print(scores)

print("\nBest Pair")
print(best_pair)

print("\nBest Score")
print(best_score)

print("\nMerged Pair")
print(merge_pair(best_pair, splits))
    