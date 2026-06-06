def text_to_bytes(text):
    return list(text.encode("utf-8"))


def bytes_to_text(byte_list):
    # ENSURE CLEAN INTS ONLY
    byte_list = [int(x) for x in byte_list]
    return bytes(byte_list).decode("utf-8", errors="replace")


def get_pairs(tokens):
    return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}