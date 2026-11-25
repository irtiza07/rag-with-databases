# Building RAG Apps with MongoDB, Redis & Postgres

Companion code for a three-part YouTube tutorial series. Each video builds **the same RAG app** on a different database, so you can compare them side by side.

| Video | Folder | Database feature used |
|---|---|---|
| 1. Building RAG Apps with MongoDB | [`mongodb/`](mongodb/) | Atlas Vector Search (`$vectorSearch`) |
| 2. Building RAG Apps with Redis | [`redis/`](redis/) | Redis Query Engine (`FT.CREATE` + KNN) |
| 3. Building RAG Apps with Postgres | [`postgres/`](postgres/) | `pgvector` extension |

## What is RAG?

**R**etrieval-**A**ugmented **G**eneration lets an LLM answer questions about *your* data, which it has never seen. It has two halves:

```
 WRITE (store knowledge)                         READ (answer a question)
 ───────────────────────                         ────────────────────────
 your documents                                  "How many PTO days do I get?"
      │                                                     │
      ▼                                                     ▼
 ✂️  split into chunks                            🔢 embed the question
      │                                                     │
      ▼                                                     ▼
 🔢 embed each chunk  ──►  💾 DATABASE  ◄──────── 🔍 vector search: find the
    (text → 1536 numbers)   (text + vector)          most similar chunks
                                                            │
                                                            ▼
                                                 🤖 LLM writes an answer using
                                                    ONLY those chunks
```

An **embedding** is a list of numbers that captures the *meaning* of a piece of text. Texts with similar meaning get similar numbers, so the database can find related content by meaning instead of by exact keywords ("time off" finds "vacation").

## Repo layout

```
rag-with-databases/
├── docker-compose.yml   # all three databases (start only the one you need)
├── requirements.txt     # one set of Python dependencies for all three videos
├── .env.example         # copy to .env and add your OpenAI key
├── data/                # sample docs: a fictional coffee company's handbook
├── mongodb/  redis/  postgres/
│   ├── rag_helpers.py   # chunking, embeddings, answer generation (same in all 3)
│   ├── db.py            # ⭐ the database-specific code (what each video teaches)
│   ├── seed.py          # WRITE the sample handbook into the database
│   └── app.py           # interactive menu: add documents, ask questions
```

## Setup (once, for all three videos)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/), Python 3.11+, and an [OpenAI API key](https://platform.openai.com/api-keys).

```bash
# 1. Your API key
cp .env.example .env          # then open .env and paste your key

# 2. A virtual environment: a private folder of Python packages for this project
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then open the folder for the video you're following and continue with its README.

## Models used
- **Embeddings:** `text-embedding-3-small` (1536 dimensions)
- **Answers:** `gpt-6-luna` (change `OPENAI_CHAT_MODEL` in `.env` to use another model)

Running a whole tutorial costs a fraction of a cent.
