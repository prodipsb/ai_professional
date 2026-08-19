# ================================================
# import libraries
# ================================================

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()


# =============================================================
# NON-STREAMING (what we've done so far) - for comparison
# =============================================================

def ask_normal(question):

    print("Sending request... (noting shows until it's ALL ready)")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{ "role": "user", "content": question }]
    )

    print(response.choices[0].message.content)

# =============================================================
# NON-STREAMING (what we've done so far) - for comparison ENDED
# =============================================================



# =============================================================
# STREAMING VERSION - the new part
# =============================================================

def ask_streaming(question):

    print("Sending request... (text will appear as it's generated)")

    # THE KEY CHANGE: stream=True
    # This tells the API "don't wait until you're done" -
    # send me small pieces AS SOON AS each one is ready"

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{ "role": "user", "content": question }],
        stream=True
    ) 

    # stream is NOT a normal response object anymore.
    # It's a special iterable - think of it like a box that hands you
    # ONE small piece of text every time you ask for the "next" one.

    # for chunk in stream: loops through these pieces AS THEY ARRIVE,
    # not after everything is ready.

    full_answer = ""            # we'll build the complete answer here as we go

    for chunk in stream:
        # Each chunk is a small object. Most chunks contain a tiny
        # piece of new text, but SOME chunks might be empty (e.g. the 
        # very first/last chunk, which just signals start/end).
        # That's why we safely check before using it.

        delta = chunk.choices[0].delta          # "delta" = "the NEW piece added this time"

        if delta.content:                       # only proceed if this chunk actually has new text
            piece = delta.content

            # priint without a newline, and flush immediately so it
            # appears on screen right away instead of being buffered
            print(piece, end="", flush=True)

            full_answer += piece                # keep building the complete text

    print()                                     # final newline after streaming finished
    print("\n-- Streaming complete---")
    print(f"Total characters received: {len(full_answer)}")

    return full_answer




# =================================================================
# TEST BOTH - SEE THE DIFFERENCE YOURSELF
# =================================================================

question = "Explain what RAG (Retrieval-Augemented Generation) is, in 20 sentences."


print("=" * 60)
print("STREAMING:")
print("=" * 60)
ask_streaming(question)


print("\n" + "=" * 60)
print("NORMAL (non-streaming):")
print("=" * 60)
ask_normal(question)


