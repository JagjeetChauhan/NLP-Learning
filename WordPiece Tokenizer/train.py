from tokenizer import WordPieceTokenizer

# Sample corpus
corpus = [
    "This is a simple WordPiece tokenizer",
    "This tokenizer is built from scratch",
    "WordPiece tokenization is interesting",
    "I love natural language processing"
]

# Create tokenizer
tokenizer = WordPieceTokenizer(
    vocab_size=100
)

# Train tokenizer
tokenizer.train(corpus)

# Save tokenizer
tokenizer.save("saved_tokenizer")

print("Tokenizer trained successfully!")
print(f"Vocabulary size: {len(tokenizer.token_to_id)}")