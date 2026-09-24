# Kilter Climb Finder

Given a Kilter Board climb, finds other climbs with similar (or opposite)
hold layouts, using a contrastive embedding model over local hold
neighborhoods rather than grade/name search.

Matching is **pattern-only**: angle and grade are metadata attached to the
final results, never used to filter or rank the search.

## How it works

### Representation

Each climb is broken into many small **windows**:

- **Anchors** are hand holds only (start / middle / finish / hand).
- A window is the anchor's `k` nearest neighbors by straight-line distance
  (`k` ∈ {3, 4, 5}, stored as `sequence_length`). Neighbors may also be
  foot-only holds.
- Each neighbor is a `dx=..,dy=..,c=H|F` token (offset from the anchor, plus
  whether it is a hand or foot hold), e.g.
  `dx=8,dy=40,c=H | dx=-24,dy=0,c=F | ...`.
- `canonical_key` is mirror-normalized (on the nearest neighbor's `dx` sign),
  so a layout and its mirror image collapse to the same key. `raw_key` keeps
  the un-normalized deltas.

This is spatial ("what is near this hold") rather than sequential ("what is
the next move"), and hand/foot windows get distinct keys automatically.

### Embedding model

A `sentence-transformers` bi-encoder (base: `all-MiniLM-L6-v2`) is trained
on the `raw_key` strings in two stages:

1. `MultipleNegativesRankingLoss` on (anchor, positive) pairs.
2. `TripletLoss` with synthetic hard negatives (one delta shifted by one grid
   step) and hand/foot type-flip negatives.

Train/val splits are by climb, so no climb leaks across the split. Every
window in the database is then embedded into a normalized-vector memmap.

### Retrieval

`RetrievalEngine` (`retrieval.py`) loads the model, memmap index and SQLite
DB once. For a reference climb it:

1. Searches the index with each of the climb's windows (same `k`).
2. Scores matches as cosine similarity discounted by **pattern rarity** (how
   many distinct climbs share that `canonical_key`), so a generic pattern
   shared by dozens of climbs isn't strong evidence of similarity.
3. Aggregates window matches into climb-level results, with mirror flag,
   matched window count, and angles/grades attached at the end.

`mode=opposite` ranks the same query vector farthest-first instead of
nearest, using raw cosine similarity.

### Serving

`server/app.py` is a FastAPI service that also serves the static frontend in
`client/`. The frontend looks up a climb by name (exact, with fuzzy fallback),
shows its nearest matches, and draws boards from the climb's holds.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness + whether the engine is loaded |
| `GET /climbs/lookup?name=` | Name → climb(s); names aren't unique |
| `GET /climbs/{id}/holds` | Hold coordinates for board rendering |
| `GET /climbs/{id}/recommendations?mode=similar\|opposite&top_k=5` | Ranked matches |
| `GET /climbs/unindexed-search?name=` | Stub for climbs outside the database |

## Repository layout

```
client/                     static frontend (index.html, app.js, style.css, board.png)
server/app.py               FastAPI app
server/db/                  (gitignored) db.sqlite + subpattern_index.{dat,meta.json,sqlite}
server/generate_training_data_v2/
  backfill_subpatterns.py   build subpattern_occurrences (k-NN windows) from climbs
  s1_generate_contrastive_training_pairs.py   training_pairs.jsonl
  s2_train_embedding_model.py                 two-stage training
  s3_eval_embedding_model.py                  eval gate (positive > hard neg > easy neg)
  s4_build_embedding_index.py                 embed all windows -> memmap index
  retrieval.py              RetrievalEngine (query-time logic)
  generate_answer.py        optional LLM answer layer (LM Studio, not used by the app)
export_climbing_data.py     legacy grid-based training-set export
```

## Building the data and model

The raw Kilter Board SQLite database goes in `server/db/db.sqlite`. The Kilter Board 
SQLite database can be retrieved from BoardLib: https://github.com/lemeryfertitta/BoardLib
Scripts use paths relative to their own directory, so run them from
`server/generate_training_data_v2/`:

```
python backfill_subpatterns.py
python s1_generate_contrastive_training_pairs.py
python s2_train_embedding_model.py
python s3_eval_embedding_model.py
python s4_build_embedding_index.py
```

Order matters: re-run from `backfill_subpatterns.py` whenever the window
representation changes, since the training pairs, model and index all depend
on it. Model output lands in `movement_encoder_model/` (gitignored).

## Running locally

```
pip install -r requirements.txt
uvicorn server.app:app --reload
```

Run from the repo root, then open http://localhost:8000/. The app expects
`server/db/db.sqlite`, `server/db/subpattern_index.*`, and
`server/generate_training_data_v2/movement_encoder_model/stage2` to exist.

Note: `requirements.txt` pins CUDA 12.8 PyTorch wheels; swap the
`--extra-index-url` and torch pin for a CPU build if you have no GPU.
