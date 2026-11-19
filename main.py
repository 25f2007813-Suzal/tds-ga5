import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).parent
PASSAGES_PATH = APP_DIR / "passages.json"
ABSTENTION_THRESHOLD = 0.38

app = FastAPI(title="RAG Colosseum Probe Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: SentenceTransformer
passages: list[str]
passage_embeddings: np.ndarray


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float


@app.on_event("startup")
def load_kb() -> None:
    global model, passages, passage_embeddings
    raw_data = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict) and "passages" in raw_data:
        passages = [str(p) for p in raw_data["passages"]]
    elif isinstance(raw_data, list):
        passages = [
            str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
            for item in raw_data
        ]
    else:
        raise ValueError("passages.json must be a list of 50 strings or {'passages': [...]}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    passage_embeddings = model.encode(passages, normalize_embeddings=True)


def extract_concise_answer(question: str, top_passages: list[str]) -> str:
    q_words = {
        w.strip("?.,!\"\'()").lower()
        for w in question.split()
        if len(w.strip("?.,!\"\'()")) > 2
    }
    scored_sentences: list[tuple[int, int, str]] = []

    for p_idx, passage in enumerate(top_passages):
        sentences = [s.strip() for s in passage.split(". ") if s.strip()]
        for s_idx, sent in enumerate(sentences):
            clean_sent = sent if sent.endswith(".") else f"{sent}."
            s_words = {w.strip("?.,!\"\'()").lower() for w in clean_sent.split()}
            overlap = len(q_words & s_words)
            priority = overlap * 10 - p_idx * 3 - s_idx
            scored_sentences.append((priority, p_idx * 100 + s_idx, clean_sent))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = [item[2] for item in scored_sentences[:2]]
    return " ".join(selected) if selected else top_passages[0]


@app.post("/query", response_model=QueryResponse)
def query_kb(req: QueryRequest) -> QueryResponse:
    q_emb = model.encode([req.question], normalize_embeddings=True)[0]
    similarities = np.clip(passage_embeddings @ q_emb, 0.0, 1.0)

    ranked_indices = np.argsort(-similarities)[: req.top_k]
    best_score = float(similarities[ranked_indices[0]])

    if best_score < ABSTENTION_THRESHOLD:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=round(min(0.35, max(0.05, best_score * 0.7)), 4),
        )

    sources = [int(idx) for idx in ranked_indices]
    scores = [round(float(similarities[idx]), 4) for idx in ranked_indices]
    top_texts = [passages[idx] for idx in sources[:2]]
    answer_text = extract_concise_answer(req.question, top_texts)
    confidence = round(min(0.95, max(0.65, best_score)), 4)

    return QueryResponse(
        answer=answer_text,
        sources=sources,
        scores=scores,
        confidence=confidence,
    )
