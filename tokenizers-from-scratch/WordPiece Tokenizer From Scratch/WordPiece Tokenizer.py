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

for words in lists:
    for word in words:
        print(word)