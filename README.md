# NLP-Learning
Day 1: Howe computers represent Text?
    Imagine a computer as a very stubborn machine that only understands two symbols: 0 and 1. Those are called bits. Everything—text, images, videos, programs—eventually becomes long patterns of these bits.

    Now the puzzle:
    Humans write letters, numbers, emojis, and symbols. Computers only understand 0 and 1.
    So we need a dictionary that says:
    “When you see this pattern of bits, it means this character.”
    UTF-8 is one such dictionary.

    Why Variable-Length Encoding?
    Most text is ASCII-heavy → saves memory
    Efficient for common characters
    Flexible for complex/global characters

    Text Representation Pipeline:
        Text → Unicode Code Point → UTF-8 Bytes → Binary → Memory
    
    Impact on NLP Pipelines
    Models do not understand text directly
    Everything becomes numbers
    Text → Encoding → Tokens → Token IDs → Embeddings

    Key insights:

    Character length ≠ byte length
    Different languages consume different memory/token budgets
    Encoding affects preprocessing and model efficiency