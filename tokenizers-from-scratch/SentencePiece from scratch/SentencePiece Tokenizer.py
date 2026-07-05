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

print(Initial_vocab(vocab))
