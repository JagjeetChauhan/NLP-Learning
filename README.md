# NLP-Learning

A hands-on journey through Natural Language Processing (NLP), focused on understanding how text moves from raw human-readable data to numerical representations that machine learning models can process.

The goal of this repository is **learning NLP from first principles** by implementing important concepts from scratch and then comparing them with real-world implementations.

---

## 📚 Learning Roadmap

```text
Text Representation
       ↓
Tokenization
       ↓
BPE / Byte-Level BPE
       ↓
WordPiece
       ↓
SentencePiece
       ↓
GPT-2 Tokenizer
       ↓
Embeddings
       ↓
Language Modeling
```

---

# Day 1 — How Computers Represent Text

Imagine a computer as a very stubborn machine that only understands two symbols:

```text
0 and 1
```

These are called **bits**.

Everything we work with—text, images, videos, programs, etc.—eventually has to be represented as patterns of these bits.

The problem is that humans work with:

```text
A B C
hello
नमस्ते
你好
🙂
```

while computers ultimately operate on numerical/binary representations.

So we need a standardized way of mapping human-readable characters to numbers and then to bytes.

---

## Unicode

**Unicode** assigns a unique code point to characters across different writing systems.

For example:

```text
Character → Unicode Code Point
A         → U+0041
a         → U+0061
```

Unicode allows computers to represent characters from languages and symbol systems around the world.

However, a Unicode code point is not itself the final byte representation stored or transmitted by a system.

That's where **UTF-8** comes in.

---

## UTF-8

UTF-8 is a variable-length encoding used to represent Unicode characters as bytes.

A simplified pipeline looks like this:

```text
Human-readable text
        ↓
Unicode code points
        ↓
UTF-8 encoding
        ↓
Bytes
        ↓
Binary
        ↓
Computer memory / storage
```

### Why Variable-Length Encoding?

UTF-8 uses between **1 and 4 bytes** for a Unicode code point.

This is useful because:

* Most common English characters are represented using 1 byte.
* ASCII-compatible text remains compact.
* Characters from other languages can still be represented.
* Emojis and other Unicode symbols can also be represented.

For example:

```text
A
↓
U+0041
↓
UTF-8 byte
↓
0x41
↓
01000001
```

---

## Character Length ≠ Byte Length

One of the most important lessons is:

> A character and a byte are not necessarily the same thing.

For example, an English character such as:

```text
A
```

requires one UTF-8 byte.

But many characters outside ASCII require multiple bytes.

Therefore:

```text
number of characters
        ≠
number of bytes
```

This becomes especially important when working with NLP systems.

---

# Why This Matters for NLP

Machine learning models do not directly understand text.

A model ultimately operates on numbers.

The text-processing pipeline therefore becomes something like:

```text
Text
 ↓
Encoding
 ↓
Tokens
 ↓
Token IDs
 ↓
Embeddings
 ↓
Model
```

Each stage solves a different problem.

### Encoding

Converts characters into bytes/numerical representations.

### Tokenization

Breaks text into units that a model can process.

```text
"Natural Language Processing"

        ↓

["Natural", " Language", " Processing"]
```

Depending on the tokenizer, these units may be:

* characters
* words
* subwords
* bytes
* byte-level subwords

### Token IDs

Each token is mapped to an integer.

```text
Token        → Token ID

hello        → 15496
world        → 995
```

### Embeddings

Token IDs are converted into dense numerical vectors.

```text
Token ID
   ↓
Embedding Layer
   ↓
Vector

[0.21, -0.48, 0.73, ...]
```

These vectors are what neural networks can actually operate on.

---

# 🧩 Repository Structure

The repository is organized by concepts rather than simply by chronological lessons.

```text
NLP-Learning/
│
├── Embeddings/
│   ├── data.txt
│   ├── Embedding_Learning_from_scratch.py
│   ├── Embeddings_from_scratch.py
│   └── Predict_word_using_embedding_layer_and_bigram_model.py
│
├── GPT-2 Byte Level Tokenizer/
│   ├── config.py
│   ├── tokenizer_evaluation.json
│   ├── tokenizer_pipeline.py
│   ├── tokenizer_utils.py
│   ├── tokenizer.json
│   ├── tokenizer.py
│   └── train_tokenizer.py
│
├── GPT-2 Tokenizer/
│   ├── config.py
│   ├── inference.py
│   ├── tokenizer_utils.py
│   ├── tokenizer.json
│   ├── tokenizer.py
│   └── train_tokenizer.py
│
├── tokenizers-from-scratch/
│   │
│   ├── BPE Tokenizer from scratch/
│   │   ├── BPE: Byte Pair Encoder in GPT-2.py
│   │   ├── BPE: Byte Pair Encoder for Sentences.py
│   │   ├── BPE: Byte Pair Encoder.py
│   │   └── Byte-Level BPE.py
│   │
│   ├── SentencePiece from scratch/
│   │   └── SentencePiece Tokenizer.py
│   │
│   └── WordPiece Tokenizer From Scratch/
│       └── WordPiece Tokenizer.py
│
├── WordPiece Tokenizer/
│   ├── config.py
│   ├── tokenizer_utils.py
│   ├── tokenizer.py
│   ├── train.py
│   └── saved_tokenizer/
│
├── daily tasks and ideas.txt
├── How computers represent text.py
├── README.md
└── Tokenization in NLP.py
```

---

# 🔤 Tokenization in NLP

Tokenization is the process of converting text into smaller units called **tokens**.

For example:

```text
"Learning NLP is fun!"

        ↓

["Learning", " NLP", " is", " fun", "!"]
```

However, modern NLP models generally do not rely only on simple word-level tokenization.

Instead, many systems use **subword tokenization**.

---

# Why Subword Tokenization?

Consider an unknown word:

```text
unhappiness
```

A word-level tokenizer may treat the entire word as an unknown token.

A subword tokenizer can potentially decompose it into pieces:

```text
un + happiness
```

or:

```text
un + happ + iness
```

This allows models to handle words they have never explicitly seen during training.

Subword tokenization provides a useful compromise between:

```text
Character-level
        ↕
Word-level
```

---

# 🧠 Tokenizers Implemented From Scratch

This repository contains implementations of several important tokenization algorithms.

## 1. BPE — Byte Pair Encoding

Directory:

```text
tokenizers-from-scratch/
└── BPE Tokenizer from scratch/
```

Implementations include:

```text
BPE: Byte Pair Encoder.py
BPE: Byte Pair Encoder for Sentences.py
BPE: Byte Pair Encoder in GPT-2.py
Byte-Level BPE.py
```

The goal is to understand how BPE progressively learns frequent symbol/subword pairs.

Simplified idea:

```text
Initial symbols

a b c a b c

      ↓

Find frequent pair

a b

      ↓

Merge

ab c ab c

      ↓

Continue merging
```

BPE forms the conceptual foundation for several modern subword tokenization systems.

---

# 2. Byte-Level BPE

File:

```text
tokenizers-from-scratch/
└── BPE Tokenizer from scratch/
    └── Byte-Level BPE.py
```

Byte-Level BPE operates on bytes rather than directly starting from Unicode characters.

This is particularly useful for understanding the approach used by GPT-2-style tokenizers.

Conceptually:

```text
Text
 ↓
UTF-8 bytes
 ↓
Byte-level representation
 ↓
BPE merges
 ↓
Tokens
```

---

# 3. GPT-2 Byte-Level Tokenizer

Directory:

```text
GPT-2 Byte Level Tokenizer/
```

This contains a more complete tokenizer pipeline, including:

```text
train_tokenizer.py
tokenizer.py
tokenizer_utils.py
tokenizer_pipeline.py
config.py
tokenizer.json
tokenizer_evaluation.json
```

The purpose of this section is to move from understanding the algorithm to building a tokenizer pipeline closer to a practical GPT-2-style tokenizer.

---

# 4. GPT-2 Tokenizer

Directory:

```text
GPT-2 Tokenizer/
```

Contains:

```text
train_tokenizer.py
tokenizer.py
tokenizer_utils.py
inference.py
config.py
tokenizer.json
```

This section focuses on training/loading the tokenizer and using it during inference.

A typical flow is:

```text
Training text
     ↓
Tokenizer training
     ↓
Vocabulary + merge rules
     ↓
Tokenizer
     ↓
Input text
     ↓
Token IDs
```

---

# 5. WordPiece

Directory:

```text
tokenizers-from-scratch/
└── WordPiece Tokenizer From Scratch/
    └── WordPiece Tokenizer.py
```

There is also a more complete implementation under:

```text
WordPiece Tokenizer/
```

with:

```text
train.py
tokenizer.py
tokenizer_utils.py
config.py
saved_tokenizer/
```

WordPiece is another important subword tokenization approach and is strongly associated with models such as BERT.

The key learning goal here is to understand how WordPiece differs from BPE in the way vocabulary/subword units are learned and selected.

---

# 6. SentencePiece

Directory:

```text
tokenizers-from-scratch/
└── SentencePiece from scratch/
    └── SentencePiece Tokenizer.py
```

SentencePiece is designed to tokenize raw text without requiring traditional whitespace-based preprocessing.

Conceptually:

```text
Raw text
   ↓
SentencePiece model
   ↓
Subword tokens
   ↓
Token IDs
```

This makes it particularly useful for multilingual NLP systems and languages where whitespace does not naturally separate words.

---

# 🧮 Embeddings

Directory:

```text
Embeddings/
```

Files:

```text
Embedding_Learning_from_scratch.py
Embeddings_from_scratch.py
Predict_word_using_embedding_layer_and_bigram_model.py
```

After tokenization, text has been converted into token IDs.

But token IDs themselves do not contain meaningful semantic information.

For example:

```text
"cat" → 421
"dog" → 892
```

The numbers are simply identifiers.

An embedding layer maps these IDs to vectors:

```text
Token ID
   ↓
Embedding Matrix
   ↓
Dense Vector
```

Example:

```text
cat

↓

[0.21, -0.14, 0.73, 0.42, ...]
```

The model can learn useful relationships between these vectors during training.

---

# 🔗 Tokenization → Embeddings

The important connection between the two topics is:

```text
Raw Text
   ↓
Encoding
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
   ↓
Embedding Layer
   ↓
Dense Vectors
   ↓
Neural Network
```

This is one of the fundamental pipelines behind modern NLP systems.

---

# 📈 Learning Progression

My learning progression in this repository is:

### Phase 1 — Understand Text

```text
How computers represent text
        ↓
Unicode
        ↓
UTF-8
        ↓
Bytes
        ↓
Binary
```

### Phase 2 — Understand Tokenization

```text
Text
 ↓
Characters / Words
 ↓
Subwords
 ↓
Token IDs
```

### Phase 3 — Build Tokenizers

```text
BPE
 ↓
Byte-Level BPE
 ↓
GPT-2-style tokenizer
 ↓
WordPiece
 ↓
SentencePiece
```

### Phase 4 — Understand Representations

```text
Token IDs
 ↓
Embedding Layer
 ↓
Dense Vectors
```

### Phase 5 — Connect Everything

```text
Text
 ↓
UTF-8 / Bytes
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Language Model
```

---

# 🔑 Key Insights

Some of the most important concepts learned so far:

* Computers ultimately represent information using numerical/binary representations.
* Unicode provides a universal character representation.
* UTF-8 converts Unicode code points into bytes.
* **Character length is not necessarily equal to byte length.**
* Different languages and symbols can require different numbers of bytes.
* Tokenization converts text into model-processable units.
* Modern NLP systems commonly use subword tokenization.
* BPE, Byte-Level BPE, WordPiece, and SentencePiece are different approaches to subword tokenization.
* Token IDs are identifiers, not semantic representations.
* Embeddings transform token IDs into dense vectors.
* Tokenization directly affects vocabulary size, sequence length, memory usage, and ultimately model efficiency.

---

# 🚧 Current Learning Goal

The main goal of this repository is not just to use NLP libraries.

It is to understand:

> **What actually happens underneath an NLP model?**

Instead of treating tokenizers and embeddings as black boxes, I am implementing them from scratch and progressively building toward a deeper understanding of modern language models.

```text
Understand the fundamentals
          ↓
Implement from scratch
          ↓
Compare with real implementations
          ↓
Build intuition
          ↓
Understand modern LLMs
```

---

# 📌 Repository Philosophy

> **Don't just use the abstraction. Understand what the abstraction is hiding.**

This repository is a collection of experiments, implementations, notes, and daily learning tasks while exploring NLP from the fundamentals upward.
