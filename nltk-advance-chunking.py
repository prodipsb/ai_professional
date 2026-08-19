# ============================================
# IMPORTS
# ============================================
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

# nltk needs a one-time download of its sentence-splitting rules
# (punkt = a pre-trained model that knows how to detect sentence boundaries,
# handling tricky cases like "Mr. Smith" not being treated as a sentence end)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

model = SentenceTransformer('all-MiniLM-L6-v2')


# ============================================
# LOAD DOCUMENT
# ============================================
with open("knowledge.txt", "r") as f:
    raw_text = f.read()


# ============================================
# STRATEGY 1: FIXED-SIZE CHUNKING WITH OVERLAP
# ============================================
def fixed_size_chunks(text, chunk_size=200, overlap=50):
    """
    Splits text into chunks of `chunk_size` CHARACTERS,
    but each new chunk starts `overlap` characters before
    the previous one ended.

    WHY OVERLAP MATTERS:
    Imagine chunk 1 ends mid-sentence: "...the Eiffel Tower was built"
    and chunk 2 starts: "in 1889 by Gustave Eiffel..."
    Without overlap, neither chunk alone tells the full story.
    Overlap ensures important sentences aren't cut in half and lost.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)

        # Move the start position forward, but LESS than chunk_size,
        # so the next chunk overlaps with the tail of this one
        start += (chunk_size - overlap)

    return chunks


# ============================================
# STRATEGY 2: SENTENCE-BASED CHUNKING
# ============================================
def sentence_based_chunks(text, sentences_per_chunk=2):
    """
    Splits text into individual sentences first (using NLTK's
    smart sentence detector), then GROUPS a fixed number of
    sentences together into each chunk.

    WHY THIS IS OFTEN BETTER:
    It never cuts a sentence in half (unlike fixed-size chunking).
    Each chunk is a complete, grammatically whole thought.
    """
    # Clean up the text first — replace newlines with spaces so
    # sentence detection isn't confused by paragraph breaks
    # clean_text = text.replace("\n", " ")

    # # sent_tokenize splits text into a list of individual sentences
    # # e.g. "Python is great. It is used in AI." 
    # #   -> ["Python is great.", "It is used in AI."]
    # sentences = sent_tokenize(clean_text)

    # chunks = []
    # # Loop through sentences, N at a time (step size = sentences_per_chunk)
    # for i in range(0, len(sentences), sentences_per_chunk):
    #     # Grab the next N sentences and join them into one chunk
    #     group = sentences[i:i + sentences_per_chunk]
    #     chunks.append(" ".join(group))

    # return chunks

    # Best practice: use NLTK's sent_tokenize directly on the raw text, which handles edge cases better
    paragraphs = [ p.strip() for p in text.split("\n\n") if p.strip()]

    all_chunks = []
    for para in paragraphs:
        sentences = sent_tokenize(para)
        for i in range(0, len(sentences), sentences_per_chunk):
            group = sentences[i:i + sentences_per_chunk]
            all_chunks.append(" ".join(group))
    return all_chunks

# ============================================
# COMPARE BOTH STRATEGIES
# ============================================
print("=" * 60)
print("STRATEGY 1: Fixed-size chunking (200 chars, 50 overlap)")
print("=" * 60)
fixed_chunks = fixed_size_chunks(raw_text)
for i, c in enumerate(fixed_chunks):
    print(f"\nChunk {i+1} ({len(c)} chars):")
    print(f"  {c}")

print("\n\n" + "=" * 60)
print("STRATEGY 2: Sentence-based chunking (2 sentences per chunk)")
print("=" * 60)
sent_chunks = sentence_based_chunks(raw_text)
for i, c in enumerate(sent_chunks):
    print(f"\nChunk {i+1}:")
    print(f"  {c}")