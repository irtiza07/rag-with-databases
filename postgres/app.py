"""
app.py — an interactive RAG playground in your terminal.

Run it from this folder:
    python app.py

Try this route:
    2) Ask a question the documents can't answer  → "I don't know"
    1) Add a new document that contains the answer → watch it get
       chunked, embedded, and stored
    2) Ask the same question again                 → answered from
       your new document (marked 🆕 NEW)
"""

from pathlib import Path

import db
from rag_helpers import (
    CHAT_MODEL, EMBEDDING_MODEL, chunk_text, divider, embed_text, embed_texts,
    generate_answer, info, make_chunk_id, print_embedding_preview, shorten,
    step, success, vector_preview,
)

TOP_K = 3  # how many chunks to retrieve for each question

# Sources added during this session, so we can highlight them in results.
new_sources = set()
last_added_chunk_id = None


# ---------------------------------------------------------------------
# Small input helpers
# ---------------------------------------------------------------------

def read_multiline_text():
    """
    Read text the user types or pastes, until they enter an empty line.
    Typing @path/to/file.txt loads the text from that file instead.
    """
    print("Paste or type the document text. Press Enter on an empty line to finish.")
    print("(Or type @path/to/file.txt to load a file.)")
    lines = []
    while True:
        line = input()
        if not lines and line.startswith("@"):
            return Path(line[1:].strip()).expanduser().read_text(encoding="utf-8")
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def print_record(record):
    """Print a stored record, shortening the (very long) embedding."""
    for key, value in record.items():
        if isinstance(value, list) and len(value) > 10:  # the embedding
            value = vector_preview(value)
        elif isinstance(value, str) and len(value) > 70:
            value = shorten(value, 70)
        info(f"   {key:<12}: {value}")


# ---------------------------------------------------------------------
# Menu option 1: WRITE — add a new document
# ---------------------------------------------------------------------

def add_document():
    global last_added_chunk_id

    divider("WRITE: Add a new document")
    source = input("Source name (e.g. friday-policy): ").strip()
    if not source:
        print("❌ A source name is required.")
        return
    text = read_multiline_text().strip()
    if not text:
        print("❌ No text entered.")
        return

    # Step 1: chunk
    step(1, 3, "Splitting the text into chunks...")
    pieces = chunk_text(text)
    for index, piece in enumerate(pieces):
        info(f"✂️  chunk {index}: \"{shorten(piece, 60)}\" ({len(piece)} chars)")
    success(f"{len(pieces)} chunk(s)")

    # Step 2: embed
    step(2, 3, f"Creating embeddings with '{EMBEDDING_MODEL}'...")
    embeddings = embed_texts(pieces)
    for index, embedding in enumerate(embeddings):
        info(f"chunk {index}:")
        print_embedding_preview(embedding)
    success(f"{len(embeddings)} embedding(s), {len(embeddings[0])} numbers each")

    # Step 3: store
    step(3, 3, f"Storing in {db.DB_NAME}...")
    db.delete_source(source)  # replace an older version of this source, if any
    for index, (piece, embedding) in enumerate(zip(pieces, embeddings)):
        chunk_id = make_chunk_id(source, index)
        db.store_chunk(chunk_id, source, index, piece, embedding)
        success(f"Saved {chunk_id}")
        last_added_chunk_id = chunk_id
    db.wait_until_searchable(source, embeddings[-1])

    # Read it back so we can see EXACTLY what the database stored.
    info("")
    info(f"🔎 Here is the record as {db.DB_NAME} stored it:")
    print_record(db.get_chunk(make_chunk_id(source, 0)))

    new_sources.add(source)
    print(f"\n🎉 '{source}' is stored. Ask a question about it with option 2!")


# ---------------------------------------------------------------------
# Menu option 2: READ — ask a question
# ---------------------------------------------------------------------

def ask_question():
    divider("READ: Ask a question")
    question = input("Your question: ").strip()
    if not question:
        return

    # Step 1: embed the question with the SAME model we used for the documents.
    step(1, 3, "Turning your question into an embedding...")
    question_embedding = embed_text(question)
    print_embedding_preview(question_embedding)

    # Step 2: vector search — find the stored chunks closest in meaning.
    step(2, 3, f"Searching {db.DB_NAME} for the {TOP_K} most similar chunks...")
    results = db.search(question_embedding, k=TOP_K)
    if not results:
        print("❌ Nothing found. Did you run `python seed.py` or add a document?")
        return

    info(f"{'#':<3}{'score':<8}{'source':<28}text")
    for rank, r in enumerate(results, start=1):
        tag = "🆕 NEW " if r["source"] in new_sources else ""
        info(f"{rank:<3}{r['score']:<8.3f}{tag + r['source']:<28}{shorten(r['content'], 45)}")
    info("(higher score = more similar in meaning; 1.0 would be identical)")

    # Step 3: generation — the LLM answers using only the retrieved chunks.
    step(3, 3, f"Asking '{CHAT_MODEL}' to answer using these chunks...")
    answer = generate_answer(question, results)

    divider("ANSWER")
    print(answer)

    # Did the answer come from a document you added in this session?
    used_new = [s for s in new_sources if s in answer and s in {r["source"] for r in results}]
    for source in used_new:
        print(f"\n✅ This answer used your NEW document '{source}'.")


# ---------------------------------------------------------------------
# Menu options 3, 4, 5
# ---------------------------------------------------------------------

def list_documents():
    divider(f"Documents stored in {db.DB_NAME}")
    sources = db.list_sources()
    if not sources:
        print("(empty — run `python seed.py` or add a document)")
    for source, count in sources:
        tag = "  🆕 NEW" if source in new_sources else ""
        print(f"   {source:<32} {count} chunk(s){tag}")


def inspect_chunk():
    divider("Inspect a stored chunk")
    default = last_added_chunk_id or "03_vacation_and_pto::0"
    chunk_id = input(f"Chunk ID [{default}]: ").strip() or default
    record = db.get_chunk(chunk_id)
    if record is None:
        print(f"❌ No chunk with ID '{chunk_id}'. IDs look like <source>::<number>.")
        return
    print_record(record)


def reset_database():
    answer = input("⚠️  Delete ALL stored chunks? Type 'yes' to confirm: ")
    if answer.strip().lower() == "yes":
        db.reset()
        new_sources.clear()
        print("🧹 Database emptied. Run `python seed.py` to load the handbook again.")


# ---------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------

def status_line():
    sources = db.list_sources()
    total = sum(count for _, count in sources)
    return f"connected ✅ | {total} chunks from {len(sources)} sources"


def main():
    print(f"Connecting to {db.DB_NAME}...")
    db.setup()

    actions = {
        "1": add_document,
        "2": ask_question,
        "3": list_documents,
        "4": inspect_chunk,
        "5": reset_database,
    }
    while True:
        divider(f"RAG with {db.DB_NAME}   ({status_line()})")
        print(" 1) Add a new document        (WRITE)")
        print(" 2) Ask a question            (READ)")
        print(" 3) List stored documents")
        print(" 4) Inspect a stored chunk")
        print(" 5) Reset database")
        print(" 0) Exit")
        choice = input("> ").strip()
        if choice == "0":
            print("Bye! 👋")
            break
        action = actions.get(choice)
        if action is None:
            print("Please type a number from the menu.")
            continue
        try:
            action()
        except Exception as error:  # keep the menu alive on errors
            print(f"❌ Something went wrong: {error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye! 👋")
