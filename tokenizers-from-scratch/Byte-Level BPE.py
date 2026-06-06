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


#=======================================Testing======================================================
text = "Hello"
print("Text: ",text)
print("\nText to Byte:",text_to_bytes(text))