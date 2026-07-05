""" Stage 1 — Text preprocessing
Replace spaces with ▁
Treat each sentence as one sequence """

def Text_preprocess(sentence):
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
vocab = Text_preprocess(sentence)
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

