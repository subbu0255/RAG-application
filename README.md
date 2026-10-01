# RAG FAQ Application

A beginner-friendly Retrieval-Augmented Generation (RAG) FAQ application using:

- Python
- Streamlit
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Groq API for answer generation

## Architecture

User Question
    |
    v
Sentence Transformer
    |
    v
Query Embedding
    |
    v
ChromaDB
    |
    v
Top-K FAQ Chunks
    |
    v
Groq LLM
    |
    v
Final Answer

## Prerequisites

Recommended:
- Python 3.10 or 3.11
- pip
- A Groq API key

## 1. Create a virtual environment

macOS/Linux:

    python3 -m venv .venv
    source .venv/bin/activate

Windows:

    python -m venv .venv
    .venv\Scripts\activate

## 2. Install dependencies

    pip install --upgrade pip
    pip install -r requirements.txt

## 3. Configure Groq

Copy `.env.example` to `.env`:

macOS/Linux:

    cp .env.example .env

Windows:

    copy .env.example .env

Then open `.env` and replace:

    GROQ_API_KEY=your_groq_api_key_here

with your actual key.

Do NOT commit `.env` to GitHub.

## 4. Build the ChromaDB index

Run this from the project root:

    python -m src.ingest

You should see something similar to:

    Ingested 11 chunks into 'faq_collection'.

The exact number can change if you modify the FAQ file or chunk settings.

## 5. Start the application

    streamlit run app.py

Streamlit will print a local URL, normally:

    http://localhost:8501

Open it in your browser.

## 6. Test questions

Try:

- How long does delivery take?
- Can I return an item?
- What payment methods do you accept?
- How can I track my order?
- Do you ship internationally?

Also test an unknown question:

- What is your employee vacation policy?

The application should say that the information is not available in the FAQ knowledge base instead of inventing an answer.

## Adding your own FAQs

Replace or edit:

    data/faq.txt

Then rebuild the vector index:

    python -m src.ingest

Restart Streamlit if necessary.

## Important

ChromaDB data is stored locally in:

    chroma_db/

The first run of Sentence Transformers downloads the embedding model from Hugging Face. After it is cached locally, subsequent runs are faster.

The Groq API key is only used for LLM generation. ChromaDB and the embedding model handle local retrieval/embedding.

## Troubleshooting

### `GROQ_API_KEY is missing`

Check that `.env` exists in the project root and contains:

    GROQ_API_KEY=your_real_key

### `ChromaDB collection not found`

Run:

    python -m src.ingest

before starting Streamlit.

### Port already in use

Run:

    streamlit run app.py --server.port 8502

### Rebuild everything

Delete the contents of `chroma_db/`, then run:

    python -m src.ingest

## Security

Never put your actual Groq API key inside Python source code.
Never upload `.env` to GitHub.
