# Building RAG Apps with Postgres

Part of the [RAG tutorial series](../README.md). This folder builds a complete RAG app on **Postgres**: we **WRITE** documents as embeddings, then **READ** them back by meaning to answer questions.

## How Postgres stores our data

Postgres is a regular SQL database. The **pgvector** extension adds a `vector(N)` column type, so an embedding is just **another column** in a normal table:

```
 id                     | source              | chunk_index | content          | embedding          | created_at
------------------------+---------------------+-------------+------------------+--------------------+-----------
 03_vacation_and_pto::0 | 03_vacation_and_pto | 0           | Full-time emp... | [0.0123,-0.04,...] | 2026-11-24
```

An **HNSW index** makes "find the closest vectors" fast, and you can still use all of SQL (`WHERE`, `JOIN`, ...) around it.
We use the `pgvector/pgvector:pg17` image: official Postgres 17 with pgvector pre-installed.

## Files in this folder

| File | What it does |
|---|---|
| `db.py` | ⭐ **All the Postgres-specific code**: connect, create the index, store, search. This is the file the video walks through. |
| `rag_helpers.py` | Database-agnostic helpers: chunking, OpenAI embeddings, answer generation, console output. Same in every tutorial. |
| `seed.py` | Bulk WRITE: loads the sample handbook from `../data` into Postgres. |
| `app.py` | Interactive menu to add your own documents and ask questions. |

---

## Step-by-step

> First time? Do the one-time **Setup** in the [root README](../README.md#setup-once-for-all-three-videos): create the `.env` file and the virtual environment.

### 1. Start Postgres (from the repo root)

```bash
cd ..                           # repo root, where docker-compose.yml lives
docker compose up -d postgres
docker compose ps               # wait for "healthy"
```
`docker compose ps` should show `rag-postgres` as **healthy**.

### 2. Activate the virtual environment and enter this folder

```bash
source .venv/bin/activate       # Windows: .venv\Scripts\activate
cd postgres
```

### 3. WRITE: load the sample handbook

```bash
python seed.py
```

You'll watch 5 steps: connect → load 6 files → split them into chunks → create embeddings (one API call) → store every chunk.
It's safe to run again: each document's old chunks are replaced, never duplicated.

### 4. Open the interactive app

```bash
python app.py
```

```
======================================================================
 RAG with Postgres   (connected ✅ | 12 chunks from 6 sources)
======================================================================
 1) Add a new document        (WRITE)
 2) Ask a question            (READ)
 3) List stored documents
 4) Inspect a stored chunk
 5) Reset database
 0) Exit
```

### 5. 🎬 The demo route: prove the answer comes from YOUR new document

**a) Ask something the handbook doesn't know.** Choose `2` and ask:
```
Are there meetings on Fridays?
```
The handbook never mentions it, so the answer is *"I don't know based on the stored documents."*

**b) WRITE a new document.** Choose `1`:
```
Source name: friday-policy
Starting March 2026, Fridays are no-meeting days at Brewly. Use Fridays for focused, deep work. Urgent issues can still be raised in Slack.
<press Enter on an empty line>
```
You'll see each step:
- **chunking**: the text becomes 1 chunk
- **embedding**: the 1536 numbers that represent its meaning
- **storing**: and then the record **read back from Postgres**, exactly as it was stored

**c) READ: ask the same question again.** Choose `2`:
```
Are there meetings on Fridays?
```
Now you'll see:
- the embedding of your question
- the top 3 most similar chunks, with `🆕 NEW friday-policy` ranked #1
- the full prompt sent to the LLM (the *augmented* part of RAG)
- the answer, citing `(source: friday-policy)`, then `✅ This answer used your NEW document 'friday-policy'.`

**d) Explore.** `3` lists every stored source with its chunk count. `4` shows any chunk's raw record (try `03_vacation_and_pto::0`).

More questions to try: *"How many PTO days do I get after 4 years?"*, *"What's the hotel limit when travelling?"*, *"Which coffee is best for lattes?"*

### 6. Data survives restarts

Exit the app (`0`), then from the repo root:
```bash
docker compose restart postgres
```
Run `python app.py` again and choose `3`: everything is still there, because the data lives in a Docker **named volume**.

### 7. Clean up

```bash
# Empty the database but keep it running: choose 5 in app.py
# Or stop it and delete ALL its data (run from the repo root):
docker compose down -v
```

---

## Code walkthrough: `db.py`

Every tutorial's `db.py` has the same functions, so only the Postgres-specific lines change:

- **`setup()`**: `CREATE EXTENSION vector`, a table with an `embedding vector(1536)` column, and an HNSW index using `vector_cosine_ops`.
- **`register_vector()`**: teaches psycopg to convert between Python lists and Postgres vectors.
- **`store_chunk()`**: `INSERT ... ON CONFLICT (id) DO UPDATE` is SQL's upsert.
- **`search()`**: `ORDER BY embedding <=> %s LIMIT 3`. `<=>` is cosine distance (`<->` = Euclidean, `<#>` = negative inner product), so we report `1 - distance`. Add a `WHERE` clause to filter like any SQL query!

### Peek at the data yourself (optional)
```bash
docker exec -it rag-postgres psql -U rag -d rag_demo \
  -c "SELECT id, left(content, 40), left(embedding::text, 40) FROM chunks;"
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Can't reach Postgres` | Run `docker compose up -d postgres` from the repo root and wait for `healthy`. |
| `OPENAI_API_KEY is not set` | Copy `.env.example` to `.env` in the repo root and add your key. |
| `ModuleNotFoundError` | Activate the virtual environment: `source .venv/bin/activate` (from the repo root). |
| Search returns nothing | Run `python seed.py` first, or add a document with option `1`. |
