# 📄 PDF Chatbot using RAG and LangChain

A fully local, privacy-first chatbot that lets you have a real conversation with your own PDF
documents. Ask it anything about your files, and it retrieves the exact relevant passages and
generates a grounded, cited answer — powered entirely by open-source tools running on your own
machine, with zero API keys, zero cloud costs, and zero rate limits.

Built to explore Retrieval-Augmented Generation (RAG) hands-on: chunking strategy, embedding-based
retrieval, prompt design for grounded answers, and honest evaluation of where a small local model
succeeds and where it doesn't.

## Features

- 🔍 **Retrieval-Augmented Generation (RAG)** — answers are grounded in your actual documents
  instead of the model's memory, cutting down on hallucination
- 📚 **Source citations** — every answer shows which file and page it came from, so you can verify it
- 🎯 **MMR-based retrieval** — reduces redundant, near-duplicate context chunks compared to plain
  similarity search
- 🚫 **Refusal handling** — the model is explicitly instructed to say "I don't know" for
  out-of-scope questions rather than guessing
- 🔒 **100% offline and free** — powered by [Ollama](https://ollama.com) for generation and local
  `sentence-transformers` embeddings; nothing ever leaves your machine
- ✅ **Built-in evaluation** — a lightweight test harness checks retrieval accuracy and refusal
  behavior against a real question set

## How it works

```
                    ┌──────────────┐
   PDF / TXT   ───► │   ingest.py  │  chunk + embed (local, sentence-transformers)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Chroma DB   │  persistent vector store
                    └──────┬───────┘
                           ▼
   Question    ───► ┌──────────────┐      ┌───────────────┐
                     │  Retriever   │ ───► │  Ollama (LLM) │ ───► Answer + sources
                     │  (MMR search)│      │  local model  │
                     └──────────────┘      └───────────────┘
```

## Tech stack

| Layer | Tool |
|---|---|
| Architecture | Retrieval-Augmented Generation (RAG) |
| Orchestration | LangChain (LCEL) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | Chroma |
| LLM | Ollama (`qwen2:0.5b`, swappable for larger models) |
| Evaluation | Custom retrieval-accuracy + refusal-behavior test harness |

## Setup

### 1. Install Ollama
Download from [ollama.com/download](https://ollama.com/download), then pull a model:
```bash
ollama pull qwen2:0.5b
```
For better answer quality on a capable machine, use a larger model instead:
```bash
ollama pull llama3.2:3b
```
(update `LLM_MODEL` in `query.py` to match)

### 2. Set up the Python environment
```bash
git clone https://github.com/<your-username>/pdf-chatbot-using-rag-and-langchain.git
cd pdf-chatbot-using-rag-and-langchain
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your documents
```bash
mkdir data
# copy your PDFs/TXT files into data/
```

## Usage

**Build the vector store** (run once, or whenever documents change):
```bash
python ingest.py
```

**Resetting the vector store** (if you swap out documents in `data/` and want a
clean rebuild instead of adding to the existing one):
```powershell
# Windows PowerShell
Remove-Item -Recurse -Force vectorstore

# Windows Command Prompt (cmd.exe)
rmdir /s /q vectorstore

# macOS/Linux
rm -rf vectorstore
```
Then re-run `python ingest.py`.

**Ask questions via CLI:**
```bash
python query.py
```

## Evaluation

```bash
python evaluate.py
```
Runs a set of test questions (`eval_questions.json`) against the pipeline and checks two things
independently:
- **Retrieval accuracy** — did the expected page actually get retrieved?
- **Refusal accuracy** — does the bot correctly say "I don't know" for out-of-scope questions
  instead of hallucinating?

Separating these two checks makes it easier to tell *where* a failure is coming from — bad
retrieval/chunking versus a prompt or model issue.

## Project structure

```
.
├── ingest.py             # Load → chunk → embed → persist
├── query.py               # CLI Q&A interface
├── evaluate.py            # Retrieval-accuracy + refusal-behavior evaluation
├── eval_questions.json    # Test set for evaluate.py
├── requirements.txt
└── data/                  # Your source PDFs/TXT (not tracked in git)
```

## Design notes

- **Why fully offline?** Cloud LLM free tiers (Hugging Face, Gemini, etc.) have restrictive
  and frequently-changing quotas that made rapid iteration painful. Running locally via
  Ollama removes that entirely, at the cost of slower inference and a smaller model.
- **MMR retrieval** is used instead of plain similarity search to reduce redundant,
  near-duplicate chunks in the retrieved context.
- **Refusal handling**: the prompt explicitly instructs the model to say "I don't know"
  for out-of-scope questions rather than hallucinate. This works reliably with larger models
  but can still fail with very small local models (sub-1B parameters) on detail-heavy factual
  questions — a real, observed limitation documented here rather than glossed over.
- **Model size is a deliberate, visible tradeoff** in this project: `qwen2:0.5b` was chosen to
  keep everything running comfortably offline on modest hardware. Swapping in a larger model
  (e.g. `llama3.2:3b`) is a one-line change and noticeably improves answer quality and
  faithfulness at the cost of speed and disk space.

## License

MIT — see [LICENSE](LICENSE)
