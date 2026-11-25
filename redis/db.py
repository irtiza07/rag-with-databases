"""
db.py — everything Redis-specific for our RAG app.

How Redis stores our data:
    Redis is a key-value store that keeps data IN MEMORY (very fast).
    Each chunk is one HASH (a small dictionary) stored under a key that
    starts with "chunk:":

        key:  chunk:03_vacation_and_pto::0
        value (hash fields):
            source       "03_vacation_and_pto"
            chunk_index  0
            content      "Full-time employees receive 20 days..."
            embedding    <6144 raw bytes = 1536 float32 numbers>
            created_at   1764015300

    A search index (created with FT.CREATE) WATCHES every key starting
    with "chunk:". Whenever we write such a key, Redis indexes it right
    away — including its embedding, for vector (KNN) search.

    👉 Browse the stored keys in RedisInsight: http://localhost:8001
"""

import time

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from rag_helpers import EMBEDDING_DIMENSIONS, info, success

DB_NAME = "Redis"

INDEX_NAME = "idx:chunks"
KEY_PREFIX = "chunk:"

# decode_responses=False because the embedding field holds raw bytes.
client = redis.Redis(host="localhost", port=6379, decode_responses=False)


def to_bytes(embedding):
    """
    Redis stores strings/bytes, not lists. We pack the 1536 numbers into
    raw float32 bytes (4 bytes each → 6144 bytes). The vector index knows
    how to read this format (TYPE FLOAT32 below).
    """
    return np.array(embedding, dtype=np.float32).tobytes()


def from_bytes(raw):
    """The reverse: raw bytes → a list of 1536 floats."""
    return np.frombuffer(raw, dtype=np.float32).tolist()


def setup():
    """Create the search index (only if it doesn't exist yet)."""
    # 1. Is Redis running?
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        raise SystemExit(
            "❌ Can't reach Redis on localhost:6379.\n"
            "   Did you run `docker compose up -d redis` from the repo root?"
        )

    # 2. Create the index if missing.
    try:
        client.ft(INDEX_NAME).info()
    except redis.exceptions.ResponseError:
        schema = [
            TextField("content"),         # full-text searchable
            TagField("source"),           # exact-match filter, e.g. @source:{friday-policy}
            NumericField("chunk_index"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",                   # fast approximate nearest-neighbour algorithm
                {
                    "TYPE": "FLOAT32",                 # matches to_bytes()
                    "DIM": EMBEDDING_DIMENSIONS,       # 1536 numbers
                    "DISTANCE_METRIC": "COSINE",       # how "closeness" is measured
                },
            ),
        ]
        definition = IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH)
        client.ft(INDEX_NAME).create_index(schema, definition=definition)
        info(f"Created search index '{INDEX_NAME}' on keys '{KEY_PREFIX}*'")

    success(f"Redis ready (index '{INDEX_NAME}')")


def store_chunk(chunk_id, source, chunk_index, content, embedding):
    """Save one chunk as a hash. HSET overwrites existing fields (upsert)."""
    client.hset(
        KEY_PREFIX + chunk_id,
        mapping={
            "source": source,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": to_bytes(embedding),
            "created_at": int(time.time()),
        },
    )


def wait_until_searchable(source, embedding):
    """Redis indexes hashes as soon as they're written — nothing to wait for."""


def get_chunk(chunk_id):
    """Fetch one stored hash exactly as Redis has it (or None)."""
    raw = client.hgetall(KEY_PREFIX + chunk_id)
    if not raw:
        return None
    record = {"key": KEY_PREFIX + chunk_id}
    for field, value in raw.items():
        field = field.decode()
        record[field] = from_bytes(value) if field == "embedding" else value.decode()
    return record


def search(query_embedding, k=3):
    """
    KNN vector search with FT.SEARCH.

    "*=>[KNN 3 @embedding $vec AS distance]" reads as:
        take all documents (*), then find the 3 nearest neighbours of
        $vec in the "embedding" field, and call the result "distance".

    Redis returns cosine DISTANCE (0 = identical, 2 = opposite).
    We convert it to a similarity score: score = 1 - distance.
    """
    query = (
        Query(f"*=>[KNN {k} @embedding $vec AS distance]")
        .sort_by("distance")
        .return_fields("source", "content", "distance")
        .dialect(2)
    )
    results = client.ft(INDEX_NAME).search(query, query_params={"vec": to_bytes(query_embedding)})

    return [
        {
            "id": _text(doc.id).removeprefix(KEY_PREFIX),
            "source": _text(doc.source),
            "content": _text(doc.content),
            "score": 1 - float(doc.distance),
        }
        for doc in results.docs
    ]


def list_sources():
    """Return [(source, number_of_chunks), ...] by scanning the chunk keys."""
    counts = {}
    for key in client.scan_iter(match=KEY_PREFIX + "*"):
        source = client.hget(key, "source").decode()
        counts[source] = counts.get(source, 0) + 1
    return sorted(counts.items())


def delete_source(source):
    """Delete all chunks of one source (used before re-adding it)."""
    keys = list(client.scan_iter(match=f"{KEY_PREFIX}{source}::*"))
    if keys:
        client.delete(*keys)


def reset():
    """Delete every chunk key (the index stays and will be empty)."""
    keys = list(client.scan_iter(match=KEY_PREFIX + "*"))
    if keys:
        client.delete(*keys)


def _text(value):
    return value.decode() if isinstance(value, bytes) else value
