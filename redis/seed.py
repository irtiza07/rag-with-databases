"""
seed.py — bulk WRITE: load the sample handbook (../data) into the database.

Run it from this folder:
    python seed.py

Safe to run as many times as you like: each document's old chunks are
replaced, so nothing gets duplicated.
"""

import db
from rag_helpers import (
    chunk_text, divider, embed_texts, info, load_documents,
    make_chunk_id, step, success, vector_preview,
)

TOTAL_STEPS = 5

divider(f"Seeding {db.DB_NAME} with the sample handbook")

# 1. Make sure the database is ready (index / table exists).
step(1, TOTAL_STEPS, f"Connecting to {db.DB_NAME} and preparing storage...")
db.setup()

# 2. Read the text files.
step(2, TOTAL_STEPS, "Loading documents from ../data ...")
documents = load_documents()
success(f"Loaded {len(documents)} documents")

# 3. Split each document into chunks.
step(3, TOTAL_STEPS, "Splitting documents into chunks...")
chunks = []
for doc in documents:
    pieces = chunk_text(doc["text"])
    info(f"✂️  {doc['source']:<28} → {len(pieces)} chunk(s)")
    for index, piece in enumerate(pieces):
        chunks.append({
            "id": make_chunk_id(doc["source"], index),
            "source": doc["source"],
            "chunk_index": index,
            "content": piece,
        })
success(f"Created {len(chunks)} chunks in total")

# 4. Turn every chunk into an embedding (one batched API call).
step(4, TOTAL_STEPS, f"Creating embeddings for {len(chunks)} chunks (one API call)...")
embeddings = embed_texts([c["content"] for c in chunks])
info(f"Example — first chunk '{chunks[0]['id']}':")
info(f"   text:      \"{chunks[0]['content'][:60]}...\"")
info(f"   embedding: {vector_preview(embeddings[0])}")
success(f"Got {len(embeddings)} embeddings, each with {len(embeddings[0])} numbers")

# 5. Store chunk text + embedding in the database.
step(5, TOTAL_STEPS, f"Storing chunks in {db.DB_NAME}...")
for doc in documents:
    db.delete_source(doc["source"])  # remove an older version, if any
for chunk, embedding in zip(chunks, embeddings):
    db.store_chunk(chunk["id"], chunk["source"], chunk["chunk_index"], chunk["content"], embedding)
    info(f"💾 saved {chunk['id']}")
db.wait_until_searchable(chunks[-1]["source"], embeddings[-1])
success(f"Stored {len(chunks)} chunks")

divider("Done! Now run:  python app.py")
