from __future__ import annotations

import operator
import os
import re
import threading
import time
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from dotenv import load_dotenv
from ddgs import DDGS

from image_gen import generate_image

load_dotenv()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Neon connection string to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # DuckDuckGo gives no dates; keep null
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class ImageSlot(BaseModel):
    section_title: str = Field(
        ...,
        description="Exact title of the section this image should appear at the END of. Copy it from the provided list.",
    )
    filename: str = Field(..., description="Short file name without spaces, e.g. memory_overview.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt for the image model.")


class ImagePlanLite(BaseModel):
    images: List[ImageSlot] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM (Groq, free hosted)
# -----------------------------
# Free-tier Groq limits are counted per model and per minute, so we:
#  - keep a backup model (its limits are counted separately),
#  - allow only a few LLM calls at the same time,
#  - wait and retry when a rate limit (429) happens.
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
BACKUP_MODEL = os.getenv("GROQ_MODEL_BACKUP", "openai/gpt-oss-20b")


def _make_llm(model_name: str):
    return ChatGroq(model=model_name, temperature=0, max_retries=0)


LLMS = [_make_llm(PRIMARY_MODEL)]
if BACKUP_MODEL and BACKUP_MODEL != PRIMARY_MODEL:
    LLMS.append(_make_llm(BACKUP_MODEL))

_LLM_GATE = threading.Semaphore(int(os.getenv("LLM_CONCURRENCY", "2")))


def _is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "rate_limit_exceeded" in text
        or "rate limit" in text
        or "error code: 429" in text
    )


def _retry_after_seconds(error: Exception) -> float:
    match = re.search(r"try again in (?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", str(error))
    if not match:
        return 10.0
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    return minutes * 60 + seconds + 1.0


def call_llm(messages, schema=None):
    """Call Groq with model fallback and wait-and-retry on rate limits."""
    last_error = None
    with _LLM_GATE:
        for _ in range(8):
            for model_llm in LLMS:
                try:
                    runnable = model_llm.with_structured_output(schema) if schema else model_llm
                    return runnable.invoke(messages)
                except Exception as e:
                    if not _is_rate_limit(e):
                        raise
                    last_error = e
            wait = min(_retry_after_seconds(last_error), 30.0)
            print(f"Rate limited on all models, waiting {wait:.0f}s")
            time.sleep(wait)
    raise last_error


# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""


def router_node(state: State) -> dict:
    topic = state["topic"]
    decision = call_llm(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ],
        RouterDecision,
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 4) Research (DuckDuckGo, no API key)
# -----------------------------
def _web_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []

    normalized: List[dict] = []
    for r in results or []:
        snippet = r.get("body") or r.get("snippet") or ""
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("href") or r.get("url") or "",
                "snippet": snippet[:300],
                "published_at": None,  # DuckDuckGo gives no dates
                "source": None,
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    # keep it small so free-tier limits are not hit
    queries = (state.get("queries", []) or [])[:6]
    max_results = 5

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_web_search(q, max_results=max_results))
        time.sleep(1)  # be gentle, DuckDuckGo can rate-limit

    if not raw_results:
        return {"evidence": []}

    pack = call_llm(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ],
        EvidencePack,
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = call_llm(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:10]}"
                )
            ),
        ],
        Plan,
    )

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:8]
        )

    section_md = call_llm(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Choose where illustrations would help the reader of THIS blog.

Rules:
- Choose at most 3 sections. Choose none if images would not help.
- section_title must be copied EXACTLY from the provided list of section titles.
- Do NOT return the article text. Only return the image list.
- Each prompt must describe a clean illustration or visual metaphor (simple shapes,
  flat style, consistent colors). Image models cannot draw accurate text or labels,
  so say "no text, no labels" in every prompt.
- Keep alt text and captions short and factual.
"""


def _insert_after_section(md: str, section_title: str, placeholder: str):
    """Insert a placeholder at the END of the section whose '## ' heading matches."""
    title = section_title.strip().lstrip("#").strip()
    if not title:
        return md, False

    heading = re.compile(
        r"^##\s+.*" + re.escape(title) + r".*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = heading.search(md)
    if not match:
        return md, False

    next_heading = re.compile(r"^##\s+", re.MULTILINE).search(md, match.end())
    pos = next_heading.start() if next_heading else len(md)

    md = md[:pos].rstrip() + f"\n\n{placeholder}\n\n" + md[pos:]
    return md, True


def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]

    # images switched off in .env -> skip this LLM call entirely
    if os.getenv("ENABLE_IMAGES", "true").lower() != "true":
        return {"md_with_placeholders": merged_md, "image_specs": []}

    plan = state["plan"]
    assert plan is not None

    headings = re.findall(r"^##\s+(.+)$", merged_md, flags=re.MULTILINE)
    if not headings:
        return {"md_with_placeholders": merged_md, "image_specs": []}

    section_list = "\n".join(f"- {h.strip()}" for h in headings)

    try:
        result = call_llm(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        f"Section titles:\n{section_list}"
                    )
                ),
            ],
            ImagePlanLite,
        )
    except Exception as e:
        # never let image planning break the whole blog
        print(f"Image planning failed, continuing without images: {e}")
        return {"md_with_placeholders": merged_md, "image_specs": []}

    md = merged_md
    specs: List[dict] = []

    for slot in result.images[:3]:
        placeholder = f"[[IMAGE_{len(specs) + 1}]]"
        new_md, inserted = _insert_after_section(md, slot.section_title, placeholder)
        if not inserted:
            continue

        md = new_md
        stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(slot.filename).stem) or f"image_{len(specs) + 1}"

        specs.append(
            {
                "placeholder": placeholder,
                "filename": f"{stem}.png",
                "alt": slot.alt,
                "caption": slot.caption,
                "prompt": slot.prompt,
                "size": "1024x1024",
                "quality": "medium",
            }
        )

    return {"md_with_placeholders": md, "image_specs": specs}


def _safe_filename(title: str) -> str:
    # blog titles often contain : ? / which Windows does not allow in file names
    name = re.sub(r'[\\/:*?"<>|]', "", title).strip()
    return (name or "blog")[:80]


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    out_file = f"{_safe_filename(plan.blog_title)}.md"

    for spec in image_specs:
        placeholder = spec["placeholder"]
        stem = Path(spec["filename"]).stem

        # tries the free providers in order (see image_gen.py)
        saved = generate_image(spec["prompt"], stem)

        if saved:
            rel = Path(saved).as_posix()
            md = md.replace(placeholder, f"![{spec['alt']}]({rel})\n*{spec['caption']}*")
        else:
            # no provider worked: drop the placeholder so the blog stays clean
            md = md.replace(placeholder, "")

    Path(out_file).write_text(md, encoding="utf-8")
    return {"final": md}


# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)


# =========================
# PostgreSQL Checkpointer (Neon)
# A connection pool is used because Neon's free database pauses when idle,
# and the pool checks connections before using them.
# =========================
DATABASE_URL = get_database_url()

pool = ConnectionPool(
    conninfo=DATABASE_URL,
    max_size=5,
    kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    check=ConnectionPool.check_connection,
    open=True,
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

app = g.compile(checkpointer=checkpointer)