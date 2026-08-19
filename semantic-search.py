from dotenv import load_dotenv
# from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()      # read .env file and loads it into environment
# client = OpenAI()  # automatically reads OPENAI_API_KEY from environment variables


SIMILARITY_THRESHOLD = 0.3  # tune this threshold based on testing

model = SentenceTransformer("all-MiniLM-L6-v2")

# A small "knowledge base" -- pretent these are documents/facts we want to search over
documents = [
    "The cat sat on the mat.",
    "Python is a populat programming language for AI.",
    "The Eiffel Tower is located in Paris, France.",
    "Redis is an in-momory data store ofter used for caching.",
    "Dhaka is the capital city of Bangladesh."
]

# def get_embeddings(text):
#     return model.encode(text)


# step 1: convert every document into a real embedding vector
print("Generating embeddings for documents...")
doc_embeddings = model.encode(documents)

def cosine_similariy(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# step 2: Take a user query, embed it, compare against all documents
query = "What is my name?"
query_embedding = model.encode(query)

print(f"\n Query: {query}\n")
print("Similarity scores")

similarities = []
for doc, doc_emb in zip(documents, doc_embeddings):
    sim  =  cosine_similariy(query_embedding, doc_emb)
    similarities.append((doc, sim))
    print(f" {sim:.4f} - {doc}")

# step 3 : Find the best match
best_doc, best_score = max(similarities, key=lambda x: x[1])
if best_score < SIMILARITY_THRESHOLD:
    print(f"\n No relevant document found (best score {best_score:.4f} is below threshold {SIMILARITY_THRESHOLD:.4f}).")
else:
    print(f"\n Best match: \"{best_doc}\" (score: {best_score:.4f})")