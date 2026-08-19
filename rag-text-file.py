# =====================================================
# Imports libraries
# =====================================================

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import numpy as np

load_dotenv()


model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq()


# ======================================================
# STEP 1: LOAD THE RAW DOCUMENT FROM FILE
# ======================================================

# Instand of hardcoding sentences in Python, we now read real text from a file
# much closer to how real RAG systems get thier data
# (PDFs, websites, databases, etc. all eventually become plain text)

with open("knowledge.txt", "r") as f:
    raw_text = f.read()


# ======================================================
# STEP 2: CHUNKING - splitting text into pieces
# ======================================================

# Why chunk at all? Beacuse:
#   1. Embedding models have a max input size they can handle well 
#   2. Smaller chunks = more precise retrieval ( a 500 word paragraph might only have ONE relevant sentence, diluting its embedding)
#   3. We don't want to feed the ENTIRE document to the LLM every time (wastes tekens = wastes money, and confuses the model with noise)

# Simple strategy here: split by paragraph (using blank lines as sepparator)
# In real system, you'd often use smarter chunking (by sentence count, fixed character length with overlap, etc.) - but this is a great start.

chunks = [
    chunk.strip()                           # remove extra whitespace
    for chunk in raw_text.split("\n\n")     # split wherever there's a blank line
    if chunk.split()                        # skip any empty chunks
]

print(f"Document split into {len(chunks)} chunks: \n")

for i, c in enumerate(chunks):
    print(f"    Chunk {i+1}: {c[:60]}...")      # Print first 60 chars as a preview



# ================================================================
# STEP 3: EMBED ALL CHUNKS (once upfront)
# ================================================================

chunk_embeddings = model.encode(chunks)


# ================================================================
# STEP 4: COSINE SIMILARITY
# ================================================================

def cosige_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================================================================
# STEP 5: TOP-K RETRIEVAL (upgraded from "best match only")
# ================================================================

def retrieve(query, top_k=3, threshold=0.3):

    query_embedding = model.encode(query)

    # Calculate similarity of query against EVERY chunk
    similarities = [
        (chunk, cosige_similarity(query_embedding, emb)) for chunk, emb in zip(chunks, chunk_embeddings)
    ]


    # Sort all chunks by score, HIGHEST First
    # key=lambda x: x[1] tell sort to compare using the score (2nd item)
    # reverse = True means desending order (biggest score first)

    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take only the top_k results (default: top 3)
    top_results = similarities[:top_k]


    # Filter out any that fall below our confidence threshold
    # This keeps our hallucination guard from before, but now applied
    # to a LIST of results instead of just one

    relevant_chunks = [chunk for chunk, score in top_results if score >= threshold]

    print(f" \n Top {top_k} candidates for query: '{query}' ")

    for chunk, score in top_results:
       marker = " KEPT" if score >= threshold else " REJECTED (below threshold)"
       print(f"  [{score:.4f}] {marker} - {chunk[:60]}...")

    return relevant_chunks          # could be an empty list if nothing was relevant


# ===================================================================
# STEP 6: GENERATION - now using MULTIPLE chunks as context
# ===================================================================

def generate_answer(query, context_chunks):
    if not context_chunks:          # empty list = nothing relevant found
        return "I don't have relevant information to answer that."

    # Combine multiple chunks into one context block relevant found
    # So the LLM tell where one piece of information ends and another begins

    combined_context = "\n\n".join(
        f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't fully answer it, say so honestly.
You may combine information from multiple sources if needed.

Context:
{combined_context}

Question: {query}

Answer: """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# =================================================================
# STEP 7: RAG PIPELINE
# =================================================================

def rag_pipeline(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    context_chunks = retrieve(query)

    answer = generate_answer(query, context_chunks)

    print(f"\nFINAL ANSWER: {answer}")


# ================================================================
# TEST IT
# ================================================================

rag_pipeline("When was the Eiffel Tower build and how tall is it?")
rag_pipeline("What is Docker used for?")
rag_pipeline("What's the weather today?")


