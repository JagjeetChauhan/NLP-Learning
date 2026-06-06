# =========================================================
# TOKENIZER UTILITIES
# =========================================================
#
# Functions:
#
# 1. Padding
# 2. Truncation
# 3. Attention Mask
#
# Used after:
#
# text
# ↓
# tokens
# ↓
# input ids
# ↓
# truncation
# ↓
# padding
# ↓
# Attention Mask
# =========================================================

# =========================================================
# PAD SEQUENCE
# =========================================================
# Definition:
# Makes token ID sequence fixed length.
#
# Arguments:
#
# input_ids:
# Numerical token ids
#
# max_length:
# Desired sequence length
#
# pad_token_id:
# ID of <pad>
#
#
# Example:
#
# [2, 10, 20, 3]
#
# max_length = 8
#
# ->
#
# [2,10,20,3,0,0,0,0]
#
# =========================================================

import re

def pad_sequence(
    input_ids,
    max_length,
    pad_token_id
):
    current_length = len(input_ids)

    # Already correct size
    if current_length >= max_length:
        print(
            "No padding required"
        )
        return input_ids

    # Padding needed
    if current_length < max_length:
        padded_needed = (max_length - current_length)
        input_ids = (
            input_ids + [pad_token_id] * padded_needed
        )
        print(
        f"Adding {padded_needed} padding tokens"
        )
    
    return input_ids

# =========================================================
# TRUNCATE SEQUENCE
# =========================================================
# Definition:
# Reduces sequence length when it exceeds
# max_length.
#
# Arguments:
#
# input_ids:
# Numerical token ids
#
# max_length:
# Desired sequence length
#
# eos_token_id:
# ID of <eos>
#
#
# Example:
#
# [2,10,20,30,40,50,3]
#
# max_length = 5
#
# ->
#
# [2,10,20,30,3]
#
# Why Needed?
#
# Transformers have fixed context windows.
#
# Example:
#
# GPT-2:
# 1024 tokens
#
# If input exceeds max_length,
# sequence must be shortened.
#
# Important:
#
# Preserve EOS token whenever possible.
#
# =========================================================

def truncate_sequence(
    input_ids,
    max_length,
    eos_token_id=None
):

    current_length = len(
        input_ids
    )

    # No truncation required
    if current_length <= max_length:

        print(
            "No truncation required"
        )

        return input_ids


    print(
        f"Truncating sequence from "
        f"{current_length} to "
        f"{max_length}"
    )


    # Preserve EOS token
    if eos_token_id is not None:

        truncated_ids = (

            input_ids[
                : max_length - 1
            ]

            +

            [eos_token_id]
        )

        return truncated_ids


    # Fallback truncation
    return input_ids[
        : max_length
    ]

# =========================================================
# CREATE ATTENTION MASK
# =========================================================
# Definition:
# Creates binary attention mask.
#
# Real Token:
# 1
#
# Padding Token:
# 0
#
#
# Example:
#
# Input:
#
# [2,10,20,3,0,0]
#
# Output:
#
# [1,1,1,1,0,0]
#
# =========================================================

def create_attention_mask(
    input_ids,
    pad_token_id
):

    attention_mask = [

        0 if token_id == pad_token_id

        else 1

        for token_id in input_ids
    ]

    return attention_mask

# =========================================================
# REGEX PRE-TOKENIZATION
# =========================================================
#
# Definition:
# Splits raw text into meaningful chunks
# before BPE encoding.
#
# Examples:
#
# "Hello!!!"
#
# ->
#
# ["Hello", "!", "!", "!"]
#
#
# "I'm learning NLP in 2026!"
#
# ->
#
# ["I", "'", "m", "learning",
#  "NLP", "in", "2026", "!"]
#
# =========================================================

def regex_pre_tokenize(text):

    pattern = (

        r"\d+"

        r"|[A-Za-z]+"

        r"|[^\w\s]"
    )

    tokens = re.findall(
        pattern,
        text
    )

    return tokens