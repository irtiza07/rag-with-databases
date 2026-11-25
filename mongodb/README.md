# Building RAG Apps with MongoDB

Part of the [RAG tutorial series](../README.md). This folder builds a complete RAG app on **MongoDB**: we **WRITE** documents as embeddings, then **READ** them back by meaning to answer questions.

## How MongoDB stores our data

MongoDB stores each chunk as **one document**: the text, its metadata, and its embedding live side by side:

```json
{
  "_id": "03_vacation_and_pto::0",
  "source": "03_vacation_and_pto",
  "chunk_index": 0,
  "content": "Full-time employees receive 20 days...",
  "embedding": [0.0123, -0.0441, "... 1536 numbers"],
  "created_at": "2026-11-24T20:15:00Z"
}
```

A **vector search index** on the `embedding` field powers the `$vectorSearch` aggregation stage.
We use the `mongodb/mongodb-atlas-local` image because the plain `mongo` image has no vector search.

## Files in this folder

| File | What it does |
|---|---|
| `db.py` | ⭐ **All the MongoDB-specific code**: connect, create the index, store, search. This is the file the video walks through. |
| `rag_helpers.py` | Database-agnostic helpers: chunking, OpenAI embeddings, answer generation, console output. Same in every tutorial. |
| `seed.py` | Bulk WRITE: loads the sample handbook from `../data` into MongoDB. |
| `app.py` | Interactive menu to add your own documents and ask questions. |

---

## Step-by-step

> First time? Do the one-time **Setup** in the [root README](../README.md#setup-once-for-all-three-videos): create the `.env` file and the virtual environment.

### 1. Start MongoDB (from the repo root)

```bash
cd ..                           # repo root, where docker-compose.yml lives
docker compose up -d mongodb
docker compose ps               # wait for "healthy"
```
`docker compose ps` should show `rag-mongodb` as **healthy** (takes ~20 seconds the first time).

### 2. Activate the virtual environment and enter this folder

```bash
source .venv/bin/activate       # Windows: .venv\Scripts\activate
cd mongodb
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
 RAG with MongoDB   (connected ✅ | 12 chunks from 6 sources)
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
- **storing**: and then the record **read back from MongoDB**, exactly as it was stored

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
docker compose restart mongodb
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

Every tutorial's `db.py` has the same functions, so only the MongoDB-specific lines change:

- **`setup()`**: the vector index is created with `SearchIndexModel(type="vectorSearch")` and builds *in the background*, so we poll until it's `queryable`.
- **`store_chunk()`**: `replace_one(..., upsert=True)`; the embedding is just a Python list, no conversion needed.
- **`wait_until_searchable()`**: MongoDB-specific! The search index syncs a moment *after* a write, so we wait until the new chunk is findable.
- **`search()`**: `$vectorSearch` gathers `numCandidates` (50) likely neighbours, then returns the best `limit` (3). The score is cosine similarity normalized to 0–1.

### Peek at the data yourself (optional)
```bash
docker exec -it rag-mongodb mongosh rag_demo --eval 'db.chunks.findOne({}, {embedding: {$slice: 5}})'
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Can't reach MongoDB` | Run `docker compose up -d mongodb` from the repo root and wait for `healthy`. |
| `OPENAI_API_KEY is not set` | Copy `.env.example` to `.env` in the repo root and add your key. |
| `ModuleNotFoundError` | Activate the virtual environment: `source .venv/bin/activate` (from the repo root). |
| Search returns nothing | Run `python seed.py` first, or add a document with option `1`. |
