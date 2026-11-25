# Building RAG Apps with Redis

Part of the [RAG tutorial series](../README.md). This folder builds a complete RAG app on **Redis**: we **WRITE** documents as embeddings, then **READ** them back by meaning to answer questions.

## How Redis stores our data

Redis keeps data **in memory**, which makes it very fast. Each chunk is a **hash** (a small dictionary) under a key starting with `chunk:`:

```
key:  chunk:03_vacation_and_pto::0
  source       03_vacation_and_pto
  chunk_index  0
  content      Full-time employees receive 20 days...
  embedding    <6144 raw bytes = 1536 float32 numbers>
  created_at   1795570541
```

A **search index** created with `FT.CREATE` watches every `chunk:*` key and indexes it the moment it's written, embeddings included.
We use the `redis/redis-stack` image, which includes the Query Engine (vector search) and **RedisInsight**, a web UI.

## Files in this folder

| File | What it does |
|---|---|
| `db.py` | ⭐ **All the Redis-specific code**: connect, create the index, store, search. This is the file the video walks through. |
| `rag_helpers.py` | Database-agnostic helpers: chunking, OpenAI embeddings, answer generation, console output. Same in every tutorial. |
| `seed.py` | Bulk WRITE: loads the sample handbook from `../data` into Redis. |
| `app.py` | Interactive menu to add your own documents and ask questions. |

---

## Step-by-step

> First time? Do the one-time **Setup** in the [root README](../README.md#setup-once-for-all-three-videos): create the `.env` file and the virtual environment.

### 1. Start Redis (from the repo root)

```bash
cd ..                           # repo root, where docker-compose.yml lives
docker compose up -d redis
docker compose ps               # wait for "healthy"
```
`docker compose ps` should show `rag-redis` as **healthy**. Open **http://localhost:8001** to see RedisInsight.

### 2. Activate the virtual environment and enter this folder

```bash
source .venv/bin/activate       # Windows: .venv\Scripts\activate
cd redis
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
 RAG with Redis   (connected ✅ | 12 chunks from 6 sources)
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
- **storing**: and then the record **read back from Redis**, exactly as it was stored

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
docker compose restart redis
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

Every tutorial's `db.py` has the same functions, so only the Redis-specific lines change:

- **`setup()`**: `FT.CREATE idx:chunks` with a `VectorField("embedding", "HNSW", {TYPE: FLOAT32, DIM: 1536, DISTANCE_METRIC: COSINE})` on prefix `chunk:`.
- **`to_bytes()`**: Redis stores bytes, not lists, so the 1536 numbers are packed as raw float32 bytes with NumPy.
- **`store_chunk()`**: a plain `HSET`. There's no separate "insert into index" step: the index picks up the key automatically.
- **`search()`**: `*=>[KNN 3 @embedding $vec AS distance]`. Redis returns cosine *distance* (0 = identical), so we report `1 - distance`.

### Peek at the data yourself (optional)
Open **http://localhost:8001** (RedisInsight) and browse the `chunk:*` keys, or:
```bash
docker exec -it rag-redis redis-cli HGET "chunk:friday-policy::0" content
docker exec -it rag-redis redis-cli FT.INFO idx:chunks
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Can't reach Redis` | Run `docker compose up -d redis` from the repo root and wait for `healthy`. |
| `OPENAI_API_KEY is not set` | Copy `.env.example` to `.env` in the repo root and add your key. |
| `ModuleNotFoundError` | Activate the virtual environment: `source .venv/bin/activate` (from the repo root). |
| Search returns nothing | Run `python seed.py` first, or add a document with option `1`. |
