"""
db.py — everything MongoDB-specific for our RAG app.

How MongoDB stores our data:
    Each chunk is ONE document in the "chunks" collection. The text, its
    metadata, and its embedding all live side by side in that document:

        {
          "_id":         "03_vacation_and_pto::0",
          "source":      "03_vacation_and_pto",
          "chunk_index": 0,
          "content":     "Full-time employees receive 20 days...",
          "embedding":   [0.0123, -0.0441, ... 1536 numbers],
          "created_at":  2026-11-24T20:15:00
        }

    A "vector search index" on the embedding field lets us find the
    documents whose embeddings are closest to a question's embedding,
    using the $vectorSearch aggregation stage.
"""

import time
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.operations import SearchIndexModel

from rag_helpers import EMBEDDING_DIMENSIONS, info, success

DB_NAME = "MongoDB"

# directConnection=true: talk straight to our single local server.
MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
DATABASE = "rag_demo"
COLLECTION = "chunks"
INDEX_NAME = "vector_index"

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
collection = client[DATABASE][COLLECTION]


def setup():
    """Create the collection and the vector search index (only if missing)."""
    # 1. Is MongoDB running?
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        raise SystemExit(
            "❌ Can't reach MongoDB on localhost:27017.\n"
            "   Did you run `docker compose up -d mongodb` from the repo root?"
        )

    # 2. Create the collection (a search index needs an existing collection).
    if COLLECTION not in client[DATABASE].list_collection_names():
        client[DATABASE].create_collection(COLLECTION)
        info(f"Created collection '{DATABASE}.{COLLECTION}'")

    # 3. Create the vector search index.
    existing = [index["name"] for index in collection.list_search_indexes()]
    if INDEX_NAME not in existing:
        index = SearchIndexModel(
            name=INDEX_NAME,
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",                    # field holding the vector
                        "numDimensions": EMBEDDING_DIMENSIONS,  # 1536 numbers
                        "similarity": "cosine",                 # how "closeness" is measured
                    },
                    # A "filter" field lets $vectorSearch narrow results by source.
                    {"type": "filter", "path": "source"},
                ]
            },
        )
        collection.create_search_index(index)
        info(f"Creating vector search index '{INDEX_NAME}'...")

    # 4. Search indexes are built in the BACKGROUND. Wait until it's ready.
    while True:
        index = next(iter(collection.list_search_indexes(INDEX_NAME)), None)
        if index and index.get("queryable"):
            break
        info("⏳ waiting for the vector index to become queryable...")
        time.sleep(2)
    success(f"MongoDB ready (collection '{COLLECTION}', index '{INDEX_NAME}')")


def store_chunk(chunk_id, source, chunk_index, content, embedding):
    """Save one chunk. replace_one + upsert=True means insert-or-overwrite."""
    collection.replace_one(
        {"_id": chunk_id},
        {
            "_id": chunk_id,
            "source": source,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": embedding,  # a plain Python list of floats — no conversion needed
            "created_at": datetime.now(timezone.utc),
        },
        upsert=True,
    )


def wait_until_searchable(source, embedding, timeout_seconds=30):
    """
    MongoDB-specific: the vector index is updated a moment AFTER a write
    (it syncs in the background). Wait until the new chunk shows up in
    vector search so the very next question can find it.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        found = list(collection.aggregate([
            {"$vectorSearch": {
                "index": INDEX_NAME, "path": "embedding", "queryVector": embedding,
                "numCandidates": 10, "limit": 1, "filter": {"source": source},
            }},
        ]))
        if found:
            info("🔄 vector index has picked up the new data")
            return
        time.sleep(0.5)
    info("⚠️  index is still syncing — results may take a few more seconds")


def get_chunk(chunk_id):
    """Fetch one stored document exactly as MongoDB has it (or None)."""
    return collection.find_one({"_id": chunk_id})


def search(query_embedding, k=3):
    """
    Vector search with the $vectorSearch aggregation stage.

    numCandidates vs limit:
      The index is approximate (HNSW). It first gathers `numCandidates`
      likely neighbours, then returns the best `limit` of them. More
      candidates = more accurate but slower. ~10-20x `limit` is typical.

    Score: MongoDB normalizes cosine similarity to 0..1, where 1 = identical.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "id": "$_id",
                "source": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def list_sources():
    """Return [(source, number_of_chunks), ...] using an aggregation."""
    pipeline = [
        {"$group": {"_id": "$source", "chunks": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    return [(row["_id"], row["chunks"]) for row in collection.aggregate(pipeline)]


def delete_source(source):
    """Delete all chunks of one source (used before re-adding it)."""
    collection.delete_many({"source": source})


def reset():
    """Delete every chunk (the collection and index stay)."""
    collection.delete_many({})

