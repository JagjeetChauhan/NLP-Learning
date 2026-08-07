import numpy as np

vocab = {
    "cat": 0,
    "dog": 1,
    "apple": 2,
    "car": 3,
    "book": 4
}

V = len(vocab)

word = "apple"

one_hot = np.zeros(V)
one_hot[vocab[word]] = 1

print(one_hot)

# One-Hot Similarity
cat = np.array([1, 0, 0, 0])
dog = np.array([0, 1, 0, 0])
apple = np.array([0, 0, 1, 0])

print("cat · dog =", np.dot(cat, dog))
print("cat · apple =", np.dot(cat, apple))

print("Distance(cat, dog) =", np.linalg.norm(cat - dog))
print("Distance(cat, apple) =", np.linalg.norm(cat - apple))