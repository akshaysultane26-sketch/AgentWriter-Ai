# AgentWriter AI

A real-time, multi-agent blog writing platform. Enter a topic and a team of AI agents researches, plans, writes, and illustrates a complete blog post, streaming progress live.

## How it works

1. **Router** decides whether the topic needs web research.
2. **Research** gathers sources with DuckDuckGo search.
3. **Orchestrator** creates a structured plan (sections, goals, word targets).
4. **Workers** write each section in parallel.
5. **Reducer** merges the sections into one Markdown blog and plans images.
6. **Image agent** generates images, falling back across free providers.

## Tech stack

- LangGraph for the multi-agent workflow
- FastAPI backend with a simple HTML/JS frontend
- Groq (Llama) for text generation
- DuckDuckGo for web search
- PostgreSQL (Neon) for checkpoints and history
- Free image generation with provider fallback

## Run locally

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    uvicorn app:app --reload

Copy `.env.example` to `.env` and fill in your own keys. Then open http://127.0.0.1:8000

## Status

Work in progress.