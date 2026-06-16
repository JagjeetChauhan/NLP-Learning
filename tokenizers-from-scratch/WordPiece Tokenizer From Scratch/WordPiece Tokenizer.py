corpus = [
    "low lower lowest",
    "new newer newest",
]
def word_split(corpus):
    list_of_words = []
    for sentence in corpus:
        words = sentence.split()
        list_of_words.append(words)
    return list_of_words

lists = word_split(corpus)
print(lists)
print("\n")

# Word frequencies
word_freq = {}

for count_words in lists:
    for count_word in count_words:
        if count_word in word_freq:
            word_freq[count_word] += 1
        else:
            word_freq[count_word] = 1

print(word_freq)
print("\n")

# Split words into WordPiece-style characters
splits = {}

for word_to_chr in word_freq:
    splits[word_to_chr] = [word_to_chr[0]] + [f"##{c}" for c in word_to_chr[1:]]

print(splits)
print("\n")

# Find adjacent pairs
pairs = []

for word, tokens in splits.items():
    print(f"\n{word}: {tokens}")

    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        pairs.append(pair)
        print(pair)

print("\nAll pairs:")
print(pairs)