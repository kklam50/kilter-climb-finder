# Climb Movement-Matching RAG System — Implementation Plan

> Audience: this document is written for an autonomous coding agent (Claude Code) implementing the system. It assumes no prior conversation context. Background, schema, existing code, and open decisions are stated explicitly rather than referenced implicitly.

## 0. Project summary

Goal: given a climbing-gym database of ~344,503 climbs (hold coordinates + metadata), build a system that can recommend/compare climbs based on **movement similarity** — specifically the spacing (`dx,dy`) between consecutive hand holds, independent of hold type. The system must also support **negation queries** ("find something opposite to climb X").

Two working pieces already exist (Section 3). The core remaining work is: (1) fix a broken model-training objective, (2) build a vector index for similarity search at scale, (3) build the query-time application layer, (4) wire up generation.

**Do not reuse the existing fine-tuned causal-LM "verifier" model in the final pipeline as currently trained.** Section 2 explains why; Section 4 explains the replacement.

## 1. Data schema (verify before running anything at scale)

Source: SQLite database. Confirmed via user-provided query and parsing code:

```sql
-- Known columns (confirmed)
climbs(uuid, layout_id, name, description, frames, created_at)
climb_stats(climb_uuid, ascensionist_count, angle, difficulty_average)
difficulty_grades(difficulty, boulder_name)

-- Assumed, NOT yet confirmed against real schema — verify first:
holes(id, product_id, x, y)
placement_roles(product_id, id, ascii_symbol, full_name)
layouts(id, product_id)   -- exact join path from layout_id -> product_id is unconfirmed
```

**Action before any bulk run:** inspect the actual schema (`.schema holes`, `.schema placement_roles`, `.schema layouts` in sqlite3, or equivalent) and correct `backfill_subpatterns.py` (Section 3) if column/table names differ from the above.

`frames` format: a string of concatenated `p<hole_id>r<role_id>` segments, e.g. `"p1103r15p1234r12..."`. `hole_id` looks up into `holes` (→ x, y, product_id). `role_id` + `product_id` looks up into `placement_roles` (→ ascii_symbol, e.g. hand/foot/start/finish/excluded).

**Filter note:** current hold-filtering logic is `symbol != "X"`, which keeps *all* non-excluded holds (hands, feet, start, finish) — not hand holds exclusively, despite a stale code comment implying otherwise. Confirm whether foot holds should be included in movement-spacing calculations before treating this as final. If foot holds should be excluded, filter must change to an explicit allow-list of hand-hold symbols instead of a deny-list of `"X"`.

## 2. Why the existing fine-tuned model must not be reused as-is

The existing model was fine-tuned (via Unsloth, causal LM, instruction/input/output format) on pairs of climbs that **already share an identical `canonical_key`** — i.e., 100% positive examples, no negatives, no near-misses. Its training target (`"share a {sequence_length}-move subpattern... {mirrored/translated}"`) is fully computable from its own input via a single sign check and a string length — no judgment is required to produce it, so none was learned. Using this model as a "verifier" at query time is strictly worse than computing the same string with an f-string: same output, added latency, added failure surface (LLMs are unreliable at exact coordinate arithmetic), zero added information.

**Implication:** any similarity judgment beyond deterministic exact-key lookup (i.e., any *actual* model-driven comparison capability) requires retraining on a properly constructed dataset — see Section 4.

## 3. Existing code (already written, reuse verbatim unless schema corrections require changes)

### 3.1 `parse_frames_to_piece_map` — frames string → hold list

```python
def parse_frames_to_piece_map(frames_str, holes_table, placement_roles_table):
    """
    holes_table: dict[int, dict] mapping hole_id -> {id, product_id, x, y}
    placement_roles_table: dict[tuple[int,int], dict] mapping (product_id, role_id) -> {ascii_symbol, full_name}
    Returns list[dict] of {x, y, type_id, symbol}
    """
    import re
    if not frames_str:
        return []
    segments = re.findall(r'p(\d+)r(\d+)', frames_str)
    if not segments:
        return []
    piece_map = []
    for hole_id_str, piece_type_id_str in segments:
        hole_id = int(hole_id_str)
        piece_type_id = int(piece_type_id_str)
        if hole_id not in holes_table:
            continue
        hole_data = holes_table[hole_id]
        x, y = int(hole_data["x"]), int(hole_data["y"])
        product_id = hole_data["product_id"]
        role_key = (product_id, piece_type_id)
        symbol = "."
        if role_key in placement_roles_table:
            symbol = placement_roles_table[role_key].get("ascii_symbol", ".")
        piece_map.append({"x": x, "y": y, "type_id": piece_type_id, "symbol": symbol})
    return piece_map
```

### 3.2 `extract_canonical_windows` — hold list → canonical movement windows

```python
def extract_canonical_windows(piece_map, window_size=3):
    """
    Filters holds (symbol != "X"), sorts bottom-to-top by y, computes consecutive
    dx,dy deltas, slides a window of `window_size` deltas across the sequence.
    Mirror-normalizes each window (flips dx sign if first move goes left).
    Rounds deltas to nearest even number for +/-1 tolerance canonicalization.
    Returns list[dict] of {canonical_key, raw_key, is_mirrored, start_hold, end_hold}
    """
    m_holds = [hold for hold in piece_map if hold.get("symbol") != "X"]
    if len(m_holds) < window_size:
        return []
    sorted_holds = sorted(m_holds, key=lambda h: h['y'])
    deltas = []
    for i in range(len(sorted_holds) - 1):
        dx = sorted_holds[i+1]['x'] - sorted_holds[i]['x']
        dy = sorted_holds[i+1]['y'] - sorted_holds[i]['y']
        deltas.append((dx, dy))
    windows = []
    for i in range(len(deltas) - window_size + 1):
        win = deltas[i:i+window_size]
        if win[0][0] < 0:
            norm_win = [(-dx, dy) for dx, dy in win]
            is_mirrored = True
        else:
            norm_win = win
            is_mirrored = False
        loose_parts, raw_parts = [], []
        for dx, dy in norm_win:
            ndx = round(dx / 2) * 2
            ndy = round(dy / 2) * 2
            loose_parts.append(f"dx={ndx},dy={ndy}")
            raw_parts.append(f"dx={dx},dy={dy}")
        windows.append({
            "canonical_key": " | ".join(loose_parts),
            "raw_key": " | ".join(raw_parts),
            "is_mirrored": is_mirrored,
            "start_hold": sorted_holds[i],
            "end_hold": sorted_holds[i+window_size],
        })
    return windows
```

### 3.3 Source query (confirmed correct, already deduplicated for join semantics)

```sql
SELECT c.uuid, c.layout_id, c.name, c.description, c.frames, c.created_at,
       cs.ascensionist_count as ascensionist_count,
       cs.angle as angle,
       dg.boulder_name as climb_grade
FROM climbs c
INNER JOIN climb_stats cs ON c.uuid = cs.climb_uuid
    AND cs.ascensionist_count > 5
    AND cs.angle IS NOT NULL
LEFT JOIN difficulty_grades dg ON CAST(ROUND(cs.difficulty_average) AS INTEGER) = dg.difficulty
WHERE c.layout_id = ?
```
(Note: changed outer `LEFT JOIN cs` + redundant `WHERE cs.angle IS NOT NULL` to `INNER JOIN cs` — behaviorally identical, clearer intent.)

## 4. Phase 1 — Retrain the movement-comparison model as a contrastive embedding model

**Do not use Unsloth/causal-LM instruction-tuning for this.** Use `sentence-transformers` (bi-encoder / metric learning). Rationale: the task is "map similar movement windows to nearby vectors," not "generate text" — a bi-encoder is the correct tool, is far cheaper to run at query time (single forward pass + cosine similarity, not autoregressive generation per comparison), and produces a vector space that supports nearest-neighbor search directly.

### 4.1 Training data construction (new — nothing here exists yet)

Build a labeled pair dataset with three classes, using `subpattern_occurrences` (Section 3.4) as the source:

- **Positive pairs:** two window occurrences sharing the same `canonical_key`. (Already producible directly from the existing table — group by `canonical_key`.)
- **Hard negative pairs:** two windows whose *unrounded* deltas differ from each other by roughly 2–4 units per component (i.e., just outside the existing `±1`/round-to-even tolerance) — same `sequence_length`, different `canonical_key`, but numerically close. Requires generating these from `raw_key` values, not `canonical_key`.
- **Easy negative pairs:** randomly sampled windows with unrelated deltas, same `sequence_length`.

Target class balance: do not let positives dominate — current registry data is 100% positive by construction; explicitly downsample or upweight negatives during dataset assembly so the model faces a real discrimination task.

**Open question requiring a decision before implementation:** exact numeric threshold defining "hard negative" (e.g., is delta-distance of 2 "hard," or does it need to be 2–4, or a percentage of typical move distance?). Pick a reasonable default (e.g., Euclidean distance between raw delta vectors in range [2, 6]), document the choice in code comments, and treat as tunable — this is not a value to derive from first principles, it should be validated empirically in 4.3.

### 4.2 Training script requirements

- Input representation: the `raw_key`-style string (`"dx=8,dy=40 | dx=48,dy=24 | ..."`), tokenized as text — same representation already used, no new preprocessing needed.
- Base model: a small transformer encoder (does not need to be a large LLM — this is short symbolic text, not natural language). Default to a compact `sentence-transformers` base checkpoint unless a specific reason exists to choose otherwise.
- Loss: start with `MultipleNegativesRankingLoss` (only needs positive pairs + in-batch negatives, simplest to implement first). Once explicit hard negatives (4.1) are available, upgrade to `TripletLoss` or a contrastive loss that consumes them directly.
- Output artifact: a saved `sentence-transformers` model directory, loadable for both training-time evaluation and later batch inference (Phase 2).

### 4.3 Evaluation (required before proceeding to Phase 2)

Held-out check, not optional: verify that known-positive pairs have measurably higher cosine similarity than hard negatives, and hard negatives have measurably higher similarity than easy/random negatives. If hard negatives and positives are not well-separated, revisit the negative-mining threshold (4.1) before building the full index — do not proceed to embedding 11.7M rows on an unvalidated model.

## 5. Phase 2 — Build the embedding index

- Batch-embed every row in `subpattern_occurrences` using the Phase 1 model (same iteration/batching pattern as `backfill_subpatterns.py` — do not load all rows into memory at once).
- Storage: start with a SQLite table (`subpattern_embeddings`: `id`, `subpattern_occurrence_id`, `vector BLOB`) or a flat file (e.g. `numpy` memmap) — do not introduce FAISS/pgvector/a vector DB service unless a measured query-latency problem justifies it. This mirrors the earlier decision to prefer SQLite + index over Redis until proven insufficient.
- Incremental update path: new climbs get embedded and inserted individually (same per-climb pattern used for `subpattern_occurrences` inserts), not via full reprocessing.

## 6. Phase 3 — Query-time application layer (new module, e.g. `retrieval.py`)

Required functions:

1. `get_candidates(reference_climb_id, filters={angle, difficulty, hold_count})` — cheap metadata narrowing over `climbs`/`climb_stats`, SQL only, no model calls. Purpose: bound the search space before embedding similarity search, not to determine matches.
2. `find_similar(reference_window, candidate_pool, top_k)` — embed `reference_window` with the Phase 1 model, cosine-similarity search against `subpattern_embeddings` restricted to `candidate_pool`.
3. `find_opposite(reference_window, candidate_pool, top_k)` — negate `reference_window`'s deltas (`dx,dy → -dx,-dy`, deterministic, no model call), then call `find_similar` on the negated synthetic window. This is the negation/"opposite climb" path.
4. `assemble_context(matches)` — format `climb_name`, `climb_grade`, `angle`, and similarity score/relation into a text block for the generator prompt. Pure string formatting, no model call.

## 7. Phase 4 — Generation

- Generator: separate general-purpose instruction-tuned model (not the Phase 1 embedding model — different job, different architecture).
- Serving: LM Studio local server during development, OpenAI-compatible client (`base_url` pointed at LM Studio's endpoint). Keep the client code provider-agnostic (plain `openai` SDK usage) so switching to vLLM/TGI for production later is a `base_url` change only, not a rewrite.
- Prompt shape: system instruction constraining the model to answer only from the provided climb list + cite climb names; user message contains `assemble_context()` output plus the original question. No coordinate/delta data belongs in this prompt — the generator only ever sees already-decided, human-readable results.
- Note for deployment planning: if the embedding model (Phase 1) and the generator model need to run concurrently, plan for two served model endpoints (LM Studio serves one model per instance) — run two instances on different ports, or migrate to a multi-model server (vLLM/TGI) sooner rather than later.

## 8. Explicit non-goals for this phase

- Do not reintroduce the old causal-LM "verifier" model into the query-time hot path.
- Do not build a full-scale vector DB (Redis/FAISS/pgvector/Qdrant) until SQLite + index is measured and found insufficient.
- Do not attempt tool-calling/agentic model orchestration — the architecture is fixed as "application retrieves, then generator formats" per earlier decision.

## 9. Suggested build order (dependency-ordered)

1. Resolve schema TODOs (Section 1) → run `backfill_subpatterns.py` → verify row count
2. Build hard/easy negative pair generator (4.1)
3. Train Phase 1 bi-encoder (4.2) → evaluate (4.3) → do not proceed until separation is validated
4. Build embedding index (Phase 2)
5. Build `retrieval.py` (Phase 3)
6. Wire up generator + LM Studio client (Phase 4)
7. End-to-end smoke test: one reference climb, one similarity query, one negation query, confirm output cites real, correct climbs