# ==========================================
# Import - bringing in the tools we need
# ==========================================

# SentenceTransformer: convert text into embeddings ventors of numbers
# This run locally on Machine, no API call needed for this part
from sentence_transformers import SentenceTransformer

# Anthropic: the official library to talk to Claude's API
# from anthropic import Anthropic

# Groq: the official library to talk to Groq's API
from groq import Groq

# numpy; used here for math operations (dot product, vector length)
import numpy as np

# load_dotenv: reads .env file and loads ANTHROPIC_API_KEY into mamory
# so the Anthropic() client below can find it automatically
from dotenv import load_dotenv
load_dotenv()


# ===================================================
# SETUP - load our two "engines"
# ===================================================

# Load the embedding model ONCE (loading it is slow, using it is fast)
# 'all-MiniLM-l6-v2' is a smaill, free model that turns text into 384 numbers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a Clause client. It automatically reads ANTHROPIC_API_KEY
# from the environment (which load_dotenv()) just loaded form us
# client = Anthropic()


# Create a Groq client. It automatically reads Groq_API_KEY
# from the environment (which load_dotenv()) just loaded form us
client = Groq()

# ===================================================
# KNOWLEDGE BASE - our smail "database" of facts
# ===================================================

# In a real RAG system, these would come from a database, PDF files
# website content, etc. Here we hardcode them for learning purpose.
documents = [
    "The cat sat on the mat.",
    "Python is a popular programming language for AI.",
    "The Eiffel Tower is located in Paris, France.",
    "Redis is an in-momory data store often used for caching.",
    "Dhaka is the capital city of Bangladesh"
]


# Convert all documents into embeddings ONE TIME.
# We do this once and reuse it, instead of re-calculating every search
# (in real system, this is usually pre-computed and stored )
doc_embeddings = model.encode(documents)


# =============================================================
# HELPER FUNCTION 1: measure similarity between two vectors
# =============================================================

def cosine_similarity(a, b):
    # np.dot(a, b) = multiply matching positions and sum them up (higher = more similar direction)
    # np.linalg.norm(x) = the length of a vector
    # Dividing by both lengths makes this pure "direction similarity"
    # ignoring how long/short the vectors are (this is the standard way embeddings are compared in real systems)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==============================================================
# HELPER FUNCTION 2: RETRIEVAL step (the "R" in RAG)
# ==============================================================

def retrieve(query, threshold=0.3):
    # Convert the user's question into an embedding too.
    # so we can compare it against our document embeddings
    query_embedding = model.encode(query)

    # For every document, calculate how similar it is to the query
    # This creates a list like : [("The cat sat ...", 0.02), ("Python is ...", 0.61), ...]
    similarities = [
        (doc, cosine_similarity(query_embedding, emb))
        for doc, emb in zip(documents, doc_embeddings)
    ]

    # Pick the document with the HIGHEST similarity score
    # key=lambda x: x[1] means "compare using the score (2nd item), not the text"
    best_doc, best_score = max(similarities, key=lambda x: x[1])

    # SAFETY CHECK: if even the "best" match is weak (below threshold)
    # treat it as "nothing relevent found" instand of forcing a bad answer
    if best_score < threshold:
        return None # signals "no good mathc"
    return best_doc # this is our retrieved context


# ===============================================================
# HELPER FUNCTION 3: GENERATION step ( The "G" in RAG)
# ===============================================================
def generate_answer(query, context):
    # If retrieval found nothing relevant , don't even call the LLM
    # This saves cost AND prevents hallucination (LLM making things up).
    if context is None:
        return "I don't have relevant information to answer that."

    # Build the prompt we'll send to Claude
    # We explicitly instruct it to only use the given context
    # and to be honest if the context isn't enough.
    # This instruction is critical - without it, the LLM may just
    # answer from its own general knowledge, ignoring our document.

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't fully answer it, say so honestly.
Context: {context}

Question: {query}

Answer: """
    # send the prompt to claude via the API
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",          # which claude model to use
    #     max_tokens=200,             # limit how long the answer can be
    #     messages=[{"role": "user", "content": prompt}]          # the conversation
    # )

    # # The actual text reply is inside response.content[0].text
    # return response.content[0].text

    # Groq's API follows the same "chat" style as OpenAI
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # a strong, free open-source model
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    # Groq's response structure: response.choices[0].message.content
    return response.choices[0].message.content

# ================================================================
# THE FULL RAG PIPELINE - combining retrieval + generation
# ================================================================

def rag_pipeline(query):
    print(f"\nQuery: {query}")

    # Step 1: RETREIVE the most relevant document ( or None if nothing fits)
    context = retrieve(query)

    # Step 2: GENERATE a real answer using the context
    answer = generate_answer(query, context)
    print(f"Answer: {answer}")


# =================================================================
# TEST IT with a few different questions
# =================================================================

rag_pipeline("What programming language is good for machine learning?")
rag_pipeline("What is my name?")
rag_pipeline("Tell me about caching")


