import numpy as np;

words = ["cat", "mat", "it"]


embeddings = {
    "cat": np.array([0.6, 0.1, 0.0, 0.1]),
    "mat": np.array([0.7, 0.2, 0.0, 0.4]),
    "it": np.array([0.4, 0.2, 0.3, 0.8])
}

def softmax(x):
    e = np.exp(x - np.max(x))
    print(f" check e: {e}")
    print(f" check e sum: {e.sum()}")
    return e / e.sum()

query = embeddings["it"]

scores = []
for w in words:
    key = embeddings[w]
    score = np.dot(query, key)
    scores.append(score)

print("Raw similarity scores:")
for w, s in zip(words, scores):
    print(f"  it <-> {w}: {s:.3f}")

attention_weights = softmax(np.array(scores))
print("\nAttention weights:")
for w, weight in zip(words, attention_weights):
    print(f" {w}: {weight: .3f}")
