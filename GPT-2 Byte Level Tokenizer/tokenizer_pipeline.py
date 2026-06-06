import json
from datetime import datetime

from tokenizer import load_tokenizer, encode, decode
from tokenizer_utils import text_to_bytes, bytes_to_text


# =========================================================
# LOAD TOKENIZER
# =========================================================

print("\n=== LOADING TOKENIZER ===\n")

merge_ranks = load_tokenizer("tokenizer.json")


# =========================================================
# TEST DATA
# =========================================================

test_sentences = [
    "Hello world!",
    "GPT-2 tokenizer test 🙂",
    "I love NLP 🚀🔥",
    "Numbers 456 and 789",
    "नमस्ते दुनिया",
    "你好世界",
    "This is a full round trip test."
]


# =========================================================
# STORAGE FOR RESULTS
# =========================================================

results = {
    "timestamp": str(datetime.now()),
    "tests": []
}


# =========================================================
# ROUND TRIP EVALUATION
# =========================================================

print("\n=== ROUND TRIP TEST ===\n")

for sent in test_sentences:

    print("\n----------------------------")
    print("INPUT:", sent)

    tokens = encode(sent, merge_ranks)
    decoded = decode(tokens)

    status = "PASS" if decoded == sent else "FAIL"

    print("TOKENS:", tokens)
    print("DECODED:", decoded)
    print("STATUS:", status)

    results["tests"].append({
        "input": sent,
        "tokens": tokens,
        "decoded": decoded,
        "status": status
    })


# =========================================================
# BYTE SANITY CHECK
# =========================================================

print("\n=== BYTE SANITY CHECK ===\n")

sample = "Hello 🙂"

b = text_to_bytes(sample)
recovered = bytes_to_text(b)

print("BYTES:", b)
print("RECOVERED:", recovered)


# =========================================================
# SAVE RESULTS TO FILE
# =========================================================

output_file = "tokenizer_evaluation.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nSaved evaluation to {output_file}")


print("\nDONE")