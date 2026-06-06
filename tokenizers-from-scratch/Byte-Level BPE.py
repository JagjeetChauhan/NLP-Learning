# =========================================================
# TEXT TO BYTES
# =========================================================
#
# Converts text into UTF-8 byte values.
#
# Example:
#
# hello
#
# ->
#
# [104,101,108,108,111]
#
# =========================================================

def text_to_bytes(text):

    return list(
        text.encode("utf-8")
    )

# =========================================================
# BYTES TO TEXT
# =========================================================
#
# Converts byte values back into text.
#
# Example:
#
# [104,101,108,108,111]
#
# ->
#
# hello
#
# =========================================================

def bytes_to_text(byte_values):

    return bytes(
        byte_values
    ).decode(
        "utf-8"
    )

# =========================================================
# WORD TO BYTE TOKENS
# =========================================================
#
# Converts word into UTF-8 byte tokens.
#
# Example:
#
# low
#
# ->
#
# ['108','111','119']
#
# =========================================================

def word_to_byte_tokens(word):

    byte_values = list(
        word.encode("utf-8")
    )

    byte_tokens = [

        str(byte)

        for byte in byte_values
    ]

    return byte_tokens

#=======================================Testing======================================================
text = "Hello"
print("Text: ",text)
byte = text_to_bytes(text)
print("\nText to Byte:",byte)

print("\nByte to Text: ",bytes_to_text(byte))