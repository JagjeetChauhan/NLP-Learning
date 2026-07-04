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
                    result+= ch
            result += '\n'
        
        return result

    sentence = ["Hello I am Luv",
                "I am studing"]
    vocab = Text_preprocess(sentence)
    print(vocab)

    def add_to_storage(updated_sentences):
        vocab_storage = []
        for i in updated_sentences:
            vocab_storage.append(i)
        
        return vocab_storage

    vocab_storage = add_to_storage(vocab)
    print(vocab_storage)

"""
Stage 2 — Initial vocabulary
Build a character vocabulary (including ▁)
"""

