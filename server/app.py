"""
Minimal web API around RetrievalEngine (retrieval.py / plan.md Phase 3).

Run with:
  pip install fastapi uvicorn
  uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints:
  GET /health
  GET /climbs/{climb_id}/recommendations?mode=similar&top_k=5

Master Chief: 28EFC798ECE24D55A15AC0FE23FA986B
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.generate_training_data_v2.retrieval import RetrievalEngine

# ---------------------------------------------------------------------------
# Configuration (env vars so this doesn't need code changes to point at
# different DB/index paths in different environments)
# ---------------------------------------------------------------------------

DB_PATH = "./server/db/db.sqlite"
INDEX_PREFIX = "./server/db/subpattern_index"
MODEL_DIR =  "./server/generate_training_data_v2/movement_encoder_model/stage2"

engine: RetrievalEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Loaded once here (model + memmap + DB connections), reused for every
    # request -- this is the expensive part (model load, memmap open) and
    # it must not happen per-request.
    engine = RetrievalEngine(DB_PATH, INDEX_PREFIX, model_dir=MODEL_DIR)
    yield
    engine.close()


app = FastAPI(title="Subpattern Retrieval API", lifespan=lifespan)

# Local dev frontend (client/) is served statically / opened as a file, so
# allow any origin rather than trying to enumerate localhost ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AngleGrade(BaseModel):
    angle: float
    grade: str | None


class ClimbMatch(BaseModel):
    climb_id: str
    climb_name: str | None
    angles: list[AngleGrade]
    is_mirrored: bool
    score: float
    matched_window_count: int


class RecommendationResponse(BaseModel):
    reference_climb_id: str
    mode: str
    matches: list[ClimbMatch]
    context: str  # pre-formatted text, straight from assemble_context()


class ClimbSummary(BaseModel):
    climb_id: str
    climb_name: str
    setter_username: str
    created_at: str


class UnindexedSearchResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.get("/climbs/lookup", response_model=list[ClimbSummary])
def lookup_climbs(name: str = Query(..., min_length=1)):
    """
    Name -> id lookup (climbs.name isn't unique -- setters reuse names across
    genuinely different climbs), used by the frontend before it can call
    /climbs/{climb_id}/recommendations. Returns every exact (case-insensitive)
    match so the caller can disambiguate when there's more than one.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not initialized")

    matches = engine.find_climbs_by_name(name)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No climb found with name '{name}'")

    return [ClimbSummary(**m) for m in matches]


@app.get("/climbs/unindexed-search", response_model=UnindexedSearchResponse)
def search_unindexed_climb(name: str = Query(..., min_length=1)):
    """
    Stub for the "climb not found in climbs/index" branch (see plan.md --
    querying an un-indexed climb, e.g. one submitted as an image, needs the
    embedding model at query time instead of a plain index lookup). The
    real flow isn't built yet, so this only confirms the fork exists and
    gives the frontend a distinct, non-error response to branch on.
    """
    return UnindexedSearchResponse(
        status="not_implemented",
        message=f"'{name}' wasn't found in the database. Searching for climbs "
                 f"outside the database isn't supported yet.",
    )


@app.get("/climbs/{climb_id}/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    climb_id: str,
    mode: Literal["similar", "opposite"] = "similar",
    top_k: int = Query(5, ge=1, le=50),
):
    if engine is None:
        # Should be unreachable given lifespan setup, but fail loudly
        # rather than silently if it ever is.
        raise HTTPException(status_code=503, detail="Retrieval engine not initialized")

    try:
        matches = engine.recommend_for_climb(climb_id, mode=mode, top_k=top_k)
    except ValueError as e:
        # RetrievalEngine raises bare ValueError for "climb not found" /
        # "no subpattern windows" / "no embedding for occurrence" -- all of
        # these are client-facing 404s, not server errors.
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendationResponse(
        reference_climb_id=climb_id,
        mode=mode,
        matches=[
            ClimbMatch(
                climb_id=m["climb_id"],
                climb_name=m["climb_name"],
                angles=[AngleGrade(**a) for a in m["angles"]],
                is_mirrored=m["is_mirrored"],
                score=m["score"],
                matched_window_count=m["matched_window_count"],
            )
            for m in matches
        ],
        context=engine.assemble_context(matches),
    )