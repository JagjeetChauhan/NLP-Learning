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

sentence = ["Hello I am Luv",
           "I am studing"]
vocab = Text_preprocess_test(sentence)
print(vocab)

def Text_preprocess(sentences):
    result = []

    for sentence in sentences:
        result.append(sentence.replace(" ", "▁"))

    return result

vocab = Text_preprocess(sentence)
print(vocab)

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

vocab_list = Initial_vocab(vocab)

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

pairs = pair_statistics(vocab)

for pair, count in pairs.items():
    print(pair, ":", count)