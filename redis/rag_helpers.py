"""
rag_helpers.py — the database-agnostic parts of a RAG app.

RAG = Retrieval-Augmented Generation:
    1. RETRIEVAL   find the stored text most related to a question
    2. AUGMENTED   paste that text into the prompt we send to an LLM
    3. GENERATION  the LLM writes an answer using that text

Everything in this file is the SAME for MongoDB, Redis, and Postgres.
The database-specific code (storing + searching vectors) lives in db.py.
"""

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

# find_dotenv() searches the current folder and its parents for a ".env"
# file, so the single .env at the repo root works for every tutorial.
load_dotenv(find_dotenv(usecwd=True))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY is not set.")
    print("   Copy .env.example to .env in the repo root and paste your key in it.")
    sys.exit(1)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna")

# Every embedding from text-embedding-3-small is a list of exactly 1536
# numbers. The database index must be told this size up front.
EMBEDDING_DIMENSIONS = 1536

# The sample documents shared by all three tutorials (../data).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# One client for both embeddings and answer generation.
# It reads OPENAI_API_KEY from the environment automatically.
openai_client = OpenAI()


# ---------------------------------------------------------------------
# 2. Pretty console output (so you can follow along in the terminal)
# ---------------------------------------------------------------------

def step(number, total, message):
    """Print a step banner like:  [STEP 2/3] Creating embeddings..."""
    print(f"\n[STEP {number}/{total}] {message}")


def info(message):
    """Print an indented detail line under a step."""
    print(f"   {message}")


def success(message):
    print(f"   ✅ {message}")


def divider(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def shorten(text, max_chars=80):
    """Collapse whitespace and cut long text to one readable line."""
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def vector_preview(vector, count=5):
    """Show a long vector as e.g. [0.0123, -0.0441, ... (1536 values)]."""
    first = ", ".join(f"{x:.4f}" for x in vector[:count])
    return f"[{first}, ... ({len(vector)} values)]"


# ---------------------------------------------------------------------
# 3. Loading and chunking documents
# ---------------------------------------------------------------------

def load_documents(folder=DATA_DIR):
    """Read every .md / .txt file in a folder. Returns [{source, text}]."""
    documents = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
            documents.append({"source": path.stem, "text": text})
            info(f"📄 {path.name:<32} {len(text):>5} characters")
    return documents


def chunk_text(text, max_chars=800, overlap=100):
    """
    Split a long text into smaller pieces ("chunks").

    WHY CHUNK?
      - An embedding captures the overall meaning of the text you give it.
        A whole handbook squeezed into ONE vector gets blurry; a single
        paragraph gets a sharp, specific vector that matches questions well.
      - We only send the few most relevant chunks to the LLM, which keeps
        the prompt small, fast, and cheap.

    HOW:
      We keep whole paragraphs together and pack them into chunks of at
      most `max_chars` characters. Each new chunk starts with the last
      `overlap` characters of the previous one, so a sentence that sits
      on a boundary isn't lost.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""
    for paragraph in paragraphs:
        # Does this paragraph still fit in the current chunk?
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            # Start a new chunk with the tail of the previous one (whole words only).
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:]
            current = "..." + tail + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)
    return chunks


def make_chunk_id(source, chunk_index):
    """
    A predictable ID like "03_vacation_and_pto::0".

    Because the ID is always the same for the same source + position,
    saving a document twice OVERWRITES it ("upsert") instead of creating
    duplicates.
    """
    return f"{source}::{chunk_index}"


# ---------------------------------------------------------------------
# 4. Embeddings — turning text into numbers
# ---------------------------------------------------------------------

def embed_texts(texts):
    """
    Turn a list of texts into a list of embeddings (one per text).

    An embedding is a list of 1536 numbers that represents the MEANING of
    the text. Texts with similar meaning get similar numbers, even if they
    use different words ("time off" ≈ "vacation"). That's what lets a
    database find related content by meaning instead of by keyword.

    We send all texts in ONE API call (batching), which is much faster
    than one call per text.
    """
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_text(text):
    """Embed a single text (e.g. the user's question)."""
    return embed_texts([text])[0]


def print_embedding_preview(vector):
    info(f"🔢 Embedding: {vector_preview(vector, count=8)}")
    info("   ↳ these numbers are the *meaning* of the text, as the model sees it")


# ---------------------------------------------------------------------
# 5. Generation — asking the LLM to answer using the retrieved chunks
# ---------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a helpful assistant for Brewly Coffee Co. employees.
Answer the question using ONLY the context below.
If the context does not contain the answer, reply only: "I don't know based on the stored documents."
When you do answer, end with the source name(s) you used, like: (source: 03_vacation_and_pto)"""


def build_prompt(question, chunks):
    """
    The "Augmented" in RAG: paste the retrieved chunks into the prompt.

    The LLM has never seen our company documents. We hand it the relevant
    pieces right inside the prompt, and tell it to answer only from those.
    """
    context_parts = []
    for number, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{number}] (source: {chunk['source']})\n{chunk['content']}")
    context = "\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


def generate_answer(question, chunks):
    """Send the question + retrieved chunks to the chat model, return its answer."""
    prompt = build_prompt(question, chunks)

    print("\n   ----- Prompt sent to the LLM (this is the 'augmented' part) -----")
    for line in prompt.splitlines():
        print(f"   │ {line}")
    print("   ------------------------------------------------------------------")

    response = openai_client.responses.create(
        model=CHAT_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,  # the "rules" for the model
        input=prompt,                      # context + question
    )
    return response.output_text
