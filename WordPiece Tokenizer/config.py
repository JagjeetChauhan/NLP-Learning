"""
Configuration constants for the WordPiece tokenizer.
"""

VOCAB_SIZE = 8000
LOWERCASE = True

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN
]