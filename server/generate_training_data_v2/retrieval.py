"""
Implements plan.md Section 6 (Phase 3): the query-time application layer.

Matching is pattern-only (pattern_only_matching_plan.md Section 0): angle
and difficulty are climb metadata, not matching dimensions, so there is no
candidate-narrowing step before the vector search -- every window of the
right sequence_length is a candidate. Angle/grade are resolved only for the
final top-k matches (_enrich_angles()), never used to filter the search.

Window representation (knn_spacing_representation_plan.md): a window is NOT
a chain of consecutive-in-Y-order deltas anymore. Each hand-eligible hold is
its own anchor, and its window is the `dx,dy` deltas to its `k` nearest
neighbors by straight-line distance (`sequence_length` stores `k`, reused
as-is from the old scheme). This makes matching spatial ("what's near this
hold") rather than sequential ("what's the next move in climbing order").
canonical_key/raw_key keep the exact same string shape as before the
representation change (mirror-normalized on the nearest neighbor's dx sign,
loose/raw dx,dy pairs), so nothing below this point -- scoring, rarity
weighting, aggregation -- changed structurally; only what produced the keys
did.

Foothold matching (foothold_matching_plan.md): windows are still anchored on
hand holds only, but a window's `k` neighbors may now include Foot-Only
holds. Each pair in canonical_key/raw_key ends in `,c=H` (hand) or `,c=F`
(foot), so hand/foot-distinct windows already get distinct keys and
distinct rarity weights with no query changes.

No model calls happen in assemble_context() or _enrich_angles() -- both are
pure SQL/string formatting, per the architecture decision earlier in the
plan (app retrieves, generator only formats). The trained bi-encoder is
used only in find_similar()/find_opposite(), and only against the
precomputed vector index from Phase 2 -- there is no per-query re-embedding
of the whole database.

Score semantics (pattern_rarity_weighting_plan.md Section 3): in "similar"
mode, `score` is NOT literal cosine similarity -- it's cosine similarity
discounted by how common the matched window's canonical_key is across the
corpus (self._pattern_weight, built in __init__ from
subpattern_occurrences). This exists because window-level matches were
frequently dominated by a handful of climbs sharing one common, generic
short pattern with the reference climb (e.g. a straight ladder move) even
when the rest of those climbs were unrelated -- an exact match on a pattern
shared by 73 climbs is weak evidence of similarity, unlike an exact match on
a pattern unique to 1 climb. A climb that only matched via one generic
3-move pattern now scores well below 1.0 instead of a misleading 1.000.
"Opposite" mode scores are unaffected -- still raw cosine similarity --
since common patterns score *high*, not low, so they aren't a source of
spurious "opposite" matches.

Core functions (plan.md 6.1-6.4):
  find_similar()     -- embed a query, dot-product search against the index (nearest)
  find_opposite()    -- same search, same query vector, ranked FARTHEST instead
                         of nearest. "Opposite" is defined as: the normalized
                         pattern doesn't match at all -- NOT a geometric delta
                         negation. Mirrored windows are still considered matches
                         (they already collapse to the same canonical_key via
                         extract_canonical_windows' own mirror normalization),
                         and genuinely unrelated windows are "opposite" simply by
                         being far from the anchor in the space the model was
                         already trained to organize that way. No synthetic
                         negation string, no extra training objective needed --
                         this is exactly what the existing positive/hard-negative/
                         easy-negative contrastive training already optimizes for.
                         (An earlier version of this function negated deltas and
                         re-embedded a synthetic string; that conflated "opposite"
                         with a transformation the model was never taught to treat
                         as dissimilar -- see conversation history for the
                         diagnosis. Reverted in favor of this simpler approach.)
  assemble_context() -- format matches into text for the generator prompt

Plus recommend_for_climb(), an orchestration layer on top of these four that
aggregates window-level matches up to climb-level recommendations, since a
climb has multiple windows (sizes 2-5) and the useful output is "which
climbs", not "which windows".
"""

import sqlite3
import json
import re
import argparse
import math
import numpy as np
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

FUZZY_SCORE_CUTOFF = 75

_PAIR_RE = re.compile(r"dx=(-?\d+),dy=(-?\d+),c=([HF])")


def parse_key(key):
    return [(int(dx), int(dy), c) for dx, dy, c in _PAIR_RE.findall(key)]


def serialize_key(pairs):
    return " | ".join(f"dx={dx},dy={dy},c={c}" for dx, dy, c in pairs)


class RetrievalEngine:
    """
    Holds the model, the memmapped vector index, and DB connections, so they
    load once (e.g. at app startup) rather than per request.
    """

    def __init__(self, db_path, index_prefix, model_dir=None):
        with open(f"{index_prefix}.meta.json") as f:
            meta = json.load(f)

        self.dim = meta["dim"]
        self.count = meta["count"]
        dtype = np.float16 if meta["dtype"] == "float16" else np.float32

        self.model = SentenceTransformer(model_dir or meta["model_dir"])
        self.vectors = np.memmap(f"{index_prefix}.dat", dtype=dtype, mode="r",
                                  shape=(self.count, self.dim))

        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.map_db = sqlite3.connect(f"{index_prefix}.sqlite", check_same_thread=False)

        # Distinct names only, loaded once -- used as the candidate pool for
        # fuzzy fallback in find_climbs_by_name(). climbs.name isn't unique,
        # so matching a name string back to full rows still goes through the
        # exact-match query below.
        self._climb_names = [
            r[0] for r in self.db.execute(
                "SELECT DISTINCT name FROM climbs WHERE name IS NOT NULL"
            ).fetchall()
        ]

        # Pattern rarity weighting (pattern_rarity_weighting_plan.md Section
        # 2.1): a window's contribution to a "similar" match is discounted by
        # how common its canonical_key is across the corpus, so that one
        # exact match on a generic short pattern (shared by dozens of
        # unrelated climbs) can't single-handedly drag an otherwise
        # dissimilar climb to the top of results. df = distinct climbs
        # containing that (sequence_length, canonical_key) pattern; df >= 1
        # always, so no divide-by-zero/log(0) risk.
        self._pattern_weight = {}
        for seq_len, key, df in self.db.execute("""
            SELECT sequence_length, canonical_key, COUNT(DISTINCT climb_id)
            FROM subpattern_occurrences
            GROUP BY sequence_length, canonical_key
        """):
            self._pattern_weight[(seq_len, key)] = 1.0 / (1.0 + math.log(df))

    def close(self):
        self.db.close()
        self.map_db.close()

    # -- name -> id lookup (climbs.name isn't unique -- setters reuse names,
    #    so this can return more than one row; caller disambiguates) ---------

    def find_climbs_by_name(self, name, limit=20, fuzzy=True, fuzzy_limit=5):
        """
        Exact (case-insensitive) match first. If that finds nothing and
        `fuzzy` is set, falls back to typo-tolerant matching over the
        distinct-name pool (rapidfuzz) and resolves each matched name back
        to full rows via the same exact query -- so results are always
        real rows, fuzziness only affects which name(s) get looked up.
        """
        rows = self.db.execute(
            """
            SELECT uuid, name, setter_username, created_at
            FROM climbs
            WHERE name = ? COLLATE NOCASE
            ORDER BY created_at
            LIMIT ?
            """,
            (name, limit),
        ).fetchall()

        if not rows and fuzzy:
            fuzzy_names = self._fuzzy_match_names(name, limit=fuzzy_limit)
            for matched_name in fuzzy_names:
                rows.extend(self.db.execute(
                    """
                    SELECT uuid, name, setter_username, created_at
                    FROM climbs
                    WHERE name = ? COLLATE NOCASE
                    ORDER BY created_at
                    LIMIT ?
                    """,
                    (matched_name, limit),
                ).fetchall())

        return [
            {
                "climb_id": r[0],
                "climb_name": r[1],
                "setter_username": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    def _fuzzy_match_names(self, name, limit=5, score_cutoff=FUZZY_SCORE_CUTOFF):
        results = process.extract(
            name, self._climb_names, scorer=fuzz.WRatio,
            limit=limit, score_cutoff=score_cutoff,
        )
        return [matched_name for matched_name, _score, _index in results]

    # -- shared search core ---------------------------------------------------
    #
    # No candidate narrowing step: matching is pattern-only (governing
    # principle, plan.md Section 0) -- angle/difficulty are metadata, not
    # matching dimensions, so there is no cheap pre-filter left to apply.
    # Candidate selection collapses to "every window of the right
    # sequence_length, excluding the reference climb's own windows",
    # handled directly in _search()'s WHERE clause.

    # Chunked to stay under SQLite's default ~999-variable-per-statement
    # limit -- now that candidate narrowing is gone (Section 3), callers
    # can pass tens of thousands of occurrence_ids at once (every window of
    # a given sequence_length).
    _ROW_LOOKUP_CHUNK = 500

    def _row_indices_for(self, occurrence_ids):
        if not occurrence_ids:
            return {}
        result = {}
        for i in range(0, len(occurrence_ids), self._ROW_LOOKUP_CHUNK):
            chunk = occurrence_ids[i:i + self._ROW_LOOKUP_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self.map_db.execute(
                f"SELECT occurrence_id, row_index FROM embedding_index "
                f"WHERE occurrence_id IN ({placeholders})",
                chunk,
            ).fetchall()
            result.update(rows)
        return result

    def _load_candidate_pool(self, sequence_length, exclude_climb_id):
        """
        Fetches and vectorizes every candidate window of a given
        sequence_length, excluding one climb's own windows -- the expensive
        part of a search (SQL scan + chunked row-index lookup + vector
        gather), independent of any particular query vector.

        Exists so recommend_for_climb() can compute this ONCE per
        (sequence_length, reference_climb_id) and reuse it across every
        window of the reference climb that shares that sequence_length,
        instead of redoing it per window. Before this, a climb with N
        windows across sizes {3,4,5} ran N full re-fetches of a
        ~270K-row-per-size candidate pool -- see the perf investigation this
        method resolves (search time scaled with both corpus size and
        reference-climb window count after knn_spacing_representation_plan.md,
        since every hold now produces its own window).

        Returns None if there are no candidates (caller must handle this the
        same way _search() used to return [] for an empty candidate set).
        """
        candidates = self.db.execute(
            """
            SELECT id, climb_id, climb_name, is_mirrored, canonical_key
            FROM subpattern_occurrences
            WHERE sequence_length = ?
              AND climb_id != ?
            """,
            (sequence_length, exclude_climb_id),
        ).fetchall()
        if not candidates:
            return None

        occurrence_ids = [row[0] for row in candidates]
        row_index_map = self._row_indices_for(occurrence_ids)

        valid = [c for c in candidates if c[0] in row_index_map]
        if not valid:
            return None

        row_indices = [row_index_map[c[0]] for c in valid]
        candidate_vectors = np.asarray(self.vectors[row_indices]).astype(np.float32)

        # Precomputed regardless of mode -- cheap relative to the fetch/gather
        # above, and _search() only needs to decide whether to apply it.
        weights = np.array([
            self._pattern_weight.get((sequence_length, c[4]), 1.0) for c in valid
        ], dtype=np.float32)

        return {"rows": valid, "vectors": candidate_vectors, "weights": weights}

    def _search(self, query_vector, sequence_length, exclude_climb_id, top_k,
                farthest=False, candidate_pool=None):
        """
        query_vector: 1D normalized np array, shape (dim,)
        farthest: if True, return the LOWEST-scoring candidates instead of the
                  highest -- this is the entire mechanism behind find_opposite().
        candidate_pool: optional pre-fetched result of _load_candidate_pool()
                        for this exact (sequence_length, exclude_climb_id) --
                        pass this to skip the SQL fetch/row-index lookup/vector
                        gather when the caller already has it (recommend_for_climb()).
                        Fetched fresh via _load_candidate_pool() if omitted, so
                        standalone find_similar()/find_opposite() calls are
                        unaffected.
        Returns list of dicts: occurrence_id, climb_id, climb_name,
        is_mirrored, score. In "nearest" mode (farthest=False, i.e. "similar"),
        score is cosine similarity discounted by pattern rarity -- see
        self._pattern_weight and pattern_rarity_weighting_plan.md Section 3 --
        NOT literal cosine similarity anymore. In "farthest" mode (opposite),
        score is unweighted raw cosine similarity, unchanged from before
        (dot product, since index vectors are pre-normalized -- see
        build_embedding_index.py); rarity weighting is not applied there
        because a common pattern scores *high*, not low, so it isn't a source
        of spurious "opposite" matches.
        """
        pool = candidate_pool if candidate_pool is not None else \
            self._load_candidate_pool(sequence_length, exclude_climb_id)
        if pool is None:
            return []

        raw_scores = pool["vectors"] @ query_vector.astype(np.float32)

        if farthest:
            scores = raw_scores  # opposite mode: rarity is not the failure mode here
        else:
            scores = raw_scores * pool["weights"]

        ranked = sorted(zip(pool["rows"], scores), key=lambda x: x[1], reverse=not farthest)[:top_k]

        return [
            {
                "occurrence_id": row[0],
                "climb_id": row[1],
                "climb_name": row[2],
                "is_mirrored": bool(row[3]),
                "score": float(score),
            }
            for row, score in ranked
        ]

    # -- 6.2: similarity search ------------------------------------------------

    def find_similar(self, reference_occurrence_id, top_k=10, candidate_pool=None):
        ref = self.db.execute(
            "SELECT climb_id, sequence_length FROM subpattern_occurrences WHERE id = ?",
            (reference_occurrence_id,),
        ).fetchone()
        if ref is None:
            raise ValueError(f"No occurrence with id={reference_occurrence_id}")
        ref_climb_id, sequence_length = ref

        row_index_map = self._row_indices_for([reference_occurrence_id])
        if reference_occurrence_id not in row_index_map:
            raise ValueError(f"No embedding found for occurrence_id={reference_occurrence_id}")
        query_vector = np.asarray(self.vectors[row_index_map[reference_occurrence_id]])

        return self._search(query_vector, sequence_length, ref_climb_id, top_k,
                             candidate_pool=candidate_pool)

    # -- 6.3: "opposite" = farthest in the same trained similarity space --------

    def find_opposite(self, reference_occurrence_id, top_k=10, candidate_pool=None):
        """
        Same query vector as find_similar() -- the reference window's own
        embedding, straight from the index, no transformation. The only
        difference is farthest=True in _search(). See module docstring for
        why this replaced an earlier delta-negation approach.
        """
        ref = self.db.execute(
            "SELECT climb_id, sequence_length FROM subpattern_occurrences WHERE id = ?",
            (reference_occurrence_id,),
        ).fetchone()
        if ref is None:
            raise ValueError(f"No occurrence with id={reference_occurrence_id}")
        ref_climb_id, sequence_length = ref

        row_index_map = self._row_indices_for([reference_occurrence_id])
        if reference_occurrence_id not in row_index_map:
            raise ValueError(f"No embedding found for occurrence_id={reference_occurrence_id}")
        query_vector = np.asarray(self.vectors[row_index_map[reference_occurrence_id]])

        return self._search(query_vector, sequence_length, ref_climb_id, top_k, farthest=True,
                             candidate_pool=candidate_pool)

    # -- angle/grade enrichment: resolved only for final top-k matches ----------

    def _enrich_angles(self, climb_id):
        """
        Grade is angle-scoped (climb_stats.difficulty_average + difficulty_grades
        join), so it can't live on a deduped (climb, window) occurrence row --
        resolved here instead, only for climbs that made the final result
        (plan.md Section 4), never against the full candidate pool.
        """
        rows = self.db.execute(
            """
            SELECT cs.angle, cs.difficulty_average, dg.boulder_name
            FROM climb_stats cs
            LEFT JOIN difficulty_grades dg
              ON CAST(ROUND(cs.difficulty_average) AS INTEGER) = dg.difficulty
            WHERE cs.climb_uuid = ?
            ORDER BY cs.angle
            """,
            (climb_id,),
        ).fetchall()
        return [{"angle": r[0], "grade": r[2]} for r in rows]

    # -- 6.4: context assembly for the generator, no model call -----------------

    def assemble_context(self, matches):
        if not matches:
            return "No matching climbs found."
        lines = []
        for m in matches:
            relation = "mirrored" if m["is_mirrored"] else "direct"
            angles = ", ".join(
                f"{a['angle']}° ({a['grade'] or 'ungraded'})" for a in m["angles"]
            ) or "no logged angles"
            lines.append(
                f"- {m['climb_name']} [{angles}]: "
                f"similarity {m['score']:.3f} ({relation} movement match)"
            )
        return "\n".join(lines)

    # -- orchestration: window-level matches -> climb-level recommendations -----

    def recommend_for_climb(self, reference_climb_id, mode="similar", top_k=5):
        """
        Aggregates across all of the reference climb's windows (sizes 2-5) to
        produce climb-level recommendations, since raw window matches aren't
        directly useful to a user -- they want "which climbs", not "which
        windows". Aggregate score per candidate climb = max window-level
        score seen for that climb across all reference windows; also tracks
        how many separate windows matched, as a secondary signal of how much
        of the climb overlaps in movement, not just one lucky window.
        """
        if mode not in ("similar", "opposite"):
            raise ValueError("mode must be 'similar' or 'opposite'")

        reference_windows = self.db.execute(
            "SELECT id, sequence_length FROM subpattern_occurrences WHERE climb_id = ?",
            (reference_climb_id,),
        ).fetchall()
        if not reference_windows:
            raise ValueError(f"No subpattern windows found for climb_id={reference_climb_id}")

        search_fn = self.find_similar if mode == "similar" else self.find_opposite
        # For "similar", the best-representing window per climb is the highest
        # score seen; for "opposite" it's the lowest (most dissimilar) --
        # otherwise a climb's one accidentally-close window would outrank a
        # climb that's consistently far across all of them.
        better = (lambda new, old: new > old) if mode == "similar" else (lambda new, old: new < old)

        # The candidate pool for a given sequence_length depends only on
        # (sequence_length, reference_climb_id) -- both fixed across this
        # whole call -- not on which of the reference climb's windows is
        # being queried. Load each size's pool once and reuse it across
        # every window of that size, instead of re-fetching/re-gathering it
        # per window (see _load_candidate_pool()'s docstring for the perf
        # issue this fixes).
        candidate_pools = {}

        best_by_climb = {}
        window_counts = {}
        for occurrence_id, sequence_length in reference_windows:
            if sequence_length not in candidate_pools:
                candidate_pools[sequence_length] = self._load_candidate_pool(
                    sequence_length, reference_climb_id
                )
            matches = search_fn(occurrence_id, top_k=top_k,
                                 candidate_pool=candidate_pools[sequence_length])
            for m in matches:
                cid = m["climb_id"]
                if cid not in best_by_climb or better(m["score"], best_by_climb[cid]["score"]):
                    best_by_climb[cid] = m
                window_counts[cid] = window_counts.get(cid, 0) + 1

        for cid, m in best_by_climb.items():
            m["matched_window_count"] = window_counts[cid]
            m["angles"] = self._enrich_angles(cid)

        ranked = sorted(best_by_climb.values(), key=lambda m: m["score"],
                         reverse=(mode == "similar"))
        return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the retrieval layer (plan.md Phase 3)")
    parser.add_argument("--db", default="../db/db.sqlite")
    parser.add_argument("--index-prefix", default="../db/subpattern_index")
    parser.add_argument("--climb-id", required=True)
    parser.add_argument("--mode", choices=["similar", "opposite"], default="similar")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    engine = RetrievalEngine(args.db, args.index_prefix)
    try:
        matches = engine.recommend_for_climb(args.climb_id, mode=args.mode, top_k=args.top_k)
        print(f"\n{args.mode.capitalize()} climbs for {args.climb_id}:\n")
        print(engine.assemble_context(matches))
    finally:
        engine.close()


if __name__ == "__main__":
    main()