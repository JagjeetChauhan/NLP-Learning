# =========================================================
# TOKENIZER UTILITIES
# =========================================================
#
# Functions:
#
# 1. Padding
#
# Used after:
#
# text
# ↓
# tokens
# ↓
# input ids
# ↓
# padding
#
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