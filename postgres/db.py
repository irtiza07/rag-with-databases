"""
db.py — everything Postgres-specific for our RAG app.

How Postgres stores our data:
    Postgres is a regular SQL database. The "pgvector" extension adds a
    new column type, vector(N), so an embedding is just ANOTHER COLUMN in
    a normal table:

        id                       | source              | chunk_index | content          | embedding           | created_at
        -------------------------+---------------------+-------------+------------------+---------------------+-----------
        03_vacation_and_pto::0   | 03_vacation_and_pto | 0           | Full-time emp... | [0.0123,-0.0441,...] | 2026-11-24

    An HNSW index on the embedding column makes "find the closest
    vectors" fast, and you can still use plain SQL (WHERE, JOIN, ...)
    around it.
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from rag_helpers import EMBEDDING_DIMENSIONS, info, success

DB_NAME = "Postgres"

# These must match POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
# in the docker-compose.yml at the repo root.
CONNECTION_STRING = "postgresql://rag:rag@localhost:5432/rag_demo"

conn = None  # opened in setup()


def setup():
    """Enable pgvector, then create the table and index (only if missing)."""
    global conn

    # 1. Is Postgres running?
    try:
        # autocommit=True: every statement is saved immediately (simplest for a demo).
        conn = psycopg.connect(CONNECTION_STRING, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        raise SystemExit(
            "❌ Can't reach Postgres on localhost:5432.\n"
            "   Did you run `docker compose up -d postgres` from the repo root?"
        )

    # 2. Turn on the pgvector extension (adds the "vector" column type).
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 3. Teach psycopg to convert Python lists <-> Postgres vectors.
    register_vector(conn)

    # 4. The table: the embedding is simply a column of type vector(1536).
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({EMBEDDING_DIMENSIONS}) NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # 5. An HNSW index for fast approximate nearest-neighbour search.
    #    "vector_cosine_ops" = this index is for COSINE distance (the <=> operator).
    conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
    """)
    success("Postgres ready (table 'chunks', HNSW index 'chunks_embedding_idx')")


def store_chunk(chunk_id, source, chunk_index, content, embedding):
    """
    Save one chunk. ON CONFLICT ... DO UPDATE = "upsert":
    insert a new row, or update the existing row with the same id.
    """
    conn.execute(
        """
        INSERT INTO chunks (id, source, chunk_index, content, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            source      = EXCLUDED.source,
            chunk_index = EXCLUDED.chunk_index,
            content     = EXCLUDED.content,
            embedding   = EXCLUDED.embedding,
            created_at  = now()
        """,
        (chunk_id, source, chunk_index, content, _vector(embedding)),
    )


def wait_until_searchable(source, embedding):
    """Postgres updates the index inside the same INSERT — nothing to wait for."""


def get_chunk(chunk_id):
    """Fetch one row exactly as Postgres has it (or None)."""
    cursor = conn.execute("SELECT * FROM chunks WHERE id = %s", (chunk_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column.name for column in cursor.description]
    record = dict(zip(columns, row))
    record["embedding"] = record["embedding"].to_list()  # pgvector Vector -> Python list
    return record


def search(query_embedding, k=3):
    """
    Vector search in plain SQL.

    pgvector distance operators:
        <=>  cosine distance          (what we use; matches our index)
        <->  Euclidean (L2) distance
        <#>  negative inner product

    Cosine distance is 0 for identical meaning, so we ORDER BY it
    ascending and report similarity = 1 - distance.

    Because it's SQL, you could easily add filters, e.g.
        WHERE source = '03_vacation_and_pto'
    """
    rows = conn.execute(
        """
        SELECT id, source, content, 1 - (embedding <=> %s) AS score
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (_vector(query_embedding), _vector(query_embedding), k),
    ).fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2], "score": float(r[3])} for r in rows]


def list_sources():
    """Return [(source, number_of_chunks), ...] with a GROUP BY."""
    return conn.execute(
        "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
    ).fetchall()


def delete_source(source):
    """Delete all chunks of one source (used before re-adding it)."""
    conn.execute("DELETE FROM chunks WHERE source = %s", (source,))


def reset():
    """Delete every row (the table and index stay)."""
    conn.execute("TRUNCATE chunks")


def _vector(embedding):
    """Wrap a Python list so psycopg sends it as a 'vector' value."""
    return Vector(embedding)
