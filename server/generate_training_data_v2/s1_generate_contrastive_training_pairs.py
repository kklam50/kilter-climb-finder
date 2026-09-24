"""
Implements plan.md Section 4.1: training data construction for the
contrastive embedding model.

v2 change: hard negatives are now generated SYNTHETICALLY rather than mined
from naturally-occurring neighboring canonical_keys. Rationale (found while
running v1): real hold coordinates sit on the board's fixed physical grid
(confirmed via sample data: raw deltas are all multiples of 8), so a neighbor
search shifted by a step that doesn't match that grid queries bins that no
real climb can ever populate -- v1 returned zero hard negatives for this
reason, not because near-misses are rare. Separately, this also means the
+/-1 "tolerance" in extract_canonical_windows's round(dx/2)*2 is a no-op on
real data (multiples of 8 are already even), so canonical_key and raw_key
are presently identical in practice.

Rather than depending on the database happening to contain a naturally
occurring near-miss (a fairly specific coincidence, especially for longer
windows), a hard negative is constructed directly: take the anchor's own
deltas and shift ONE component (one dx or dy, at one position) by exactly
one grid step. This guarantees every anchor gets a hard negative and defines
"near miss" precisely, at the true resolution of the data.

Grid step is auto-detected from a sample of real deltas (GCD of observed
nonzero |dx|,|dy| values) rather than assumed, since this may differ by
board/product.

Output: JSONL, one record per anchor, fields:
  {
    "sequence_length": int,
    "anchor_raw_key": str,        "anchor_climb_id": str,
    "positive_raw_key": str,      "positive_climb_id": str,
    "hard_negative_raw_key": str,      "hard_negative_climb_id": null,  # synthetic, not a real climb
    "type_flip_negative_raw_key": str, # same geometry, ONE neighbor's c=H<->c=F flipped
    "easy_negative_raw_key": str | null, "easy_negative_climb_id": str | null,
  }
"""

import sqlite3
import re
import json
import random
import argparse
import time
import math

DB_PATH_DEFAULT = "../db/db.sqlite"
OUTPUT_DEFAULT = "training_pairs.jsonl"

_PAIR_RE = re.compile(r"dx=(-?\d+),dy=(-?\d+),c=([HF])")


def parse_key(key):
    return [(int(dx), int(dy), c) for dx, dy, c in _PAIR_RE.findall(key)]


def serialize_key(pairs):
    return " | ".join(f"dx={dx},dy={dy},c={c}" for dx, dy, c in pairs)


# ---------------------------------------------------------------------------
# Grid step detection
# ---------------------------------------------------------------------------

def infer_grid_step(conn, sample_size=3000, fallback=2):
    """
    Sample raw_key deltas and compute the GCD of observed nonzero |dx|,|dy|
    values. This is the smallest real distance between two adjacent hold
    positions on the board -- the correct unit for a "one step over" hard
    negative, as opposed to an arbitrary fixed value.
    """
    max_id = conn.execute("SELECT MAX(id) FROM subpattern_occurrences").fetchone()[0]
    if not max_id:
        print(f"  No data to sample -- using fallback grid step {fallback}.")
        return fallback

    values = set()
    attempts = 0
    while len(values) < 20 and attempts < sample_size:
        attempts += 1
        rand_id = random.randint(1, max_id)
        row = conn.execute(
            "SELECT raw_key FROM subpattern_occurrences WHERE id = ?", (rand_id,)
        ).fetchone()
        if row is None:
            continue
        for dx, dy, _c in parse_key(row[0]):
            if dx != 0:
                values.add(abs(dx))
            if dy != 0:
                values.add(abs(dy))

    if not values:
        print(f"  Could not sample nonzero deltas -- using fallback grid step {fallback}.")
        return fallback

    step = 0
    for v in values:
        step = math.gcd(step, v)

    if step < 1:
        print(f"  Computed grid step was invalid ({step}) -- using fallback {fallback}.")
        return fallback

    print(f"  Inferred grid step: {step} (from {len(values)} sampled distinct delta magnitudes)")
    return step


# ---------------------------------------------------------------------------
# Synthetic hard negative generation
# ---------------------------------------------------------------------------

def make_synthetic_hard_negative(raw_key, grid_step, rng, hops=1):
    """
    Shift exactly one (dx or dy) component of one position by
    +/-(hops * grid_step). Guaranteed to differ from the original,
    guaranteed available for every anchor.
    """
    pairs = parse_key(raw_key)
    if not pairs:
        return None
    idx = rng.randrange(len(pairs))
    dx, dy, c = pairs[idx]
    shift = hops * grid_step
    if rng.random() < 0.5:
        dx += shift if rng.random() < 0.5 else -shift
    else:
        dy += shift if rng.random() < 0.5 else -shift
    pairs[idx] = (dx, dy, c)  # type unchanged: same neighbor, shifted
    return serialize_key(pairs)


def make_type_flip_negative(raw_key, rng):
    """
    Flip the hand/foot type (c=H <-> c=F) of exactly one neighbor, leaving
    every dx,dy untouched. Without this, no training pair ever contrasts
    hand vs. foot -- positives are identical text and make_synthetic_hard_negative
    deliberately preserves `c` -- so the embedding model has no reason to
    treat the c= token as meaningful (found in the foothold_matching_plan.md
    smoke test: windows differing only in one c= scored ~1.0).
    """
    pairs = parse_key(raw_key)
    if not pairs:
        return None
    idx = rng.randrange(len(pairs))
    dx, dy, c = pairs[idx]
    pairs[idx] = (dx, dy, "F" if c == "H" else "H")
    return serialize_key(pairs)


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def ensure_indexes(conn):
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subpattern_key "
        "ON subpattern_occurrences(canonical_key)"
    )
    conn.commit()


def fetch_occurrences_for_key(conn, canonical_key, limit=5):
    cur = conn.execute(
        "SELECT id, climb_id, raw_key, sequence_length "
        "FROM subpattern_occurrences WHERE canonical_key = ? LIMIT ?",
        (canonical_key, limit),
    )
    return cur.fetchall()


def pick_positive_pair(occurrences):
    for i in range(len(occurrences)):
        for j in range(i + 1, len(occurrences)):
            if occurrences[i][1] != occurrences[j][1]:
                return occurrences[i], occurrences[j]
    if len(occurrences) >= 2:
        return occurrences[0], occurrences[1]
    return None, None


def find_easy_negative(conn, exclude_canonical_key, sequence_length, max_id, max_attempts=20):
    for _ in range(max_attempts):
        rand_id = random.randint(1, max_id)
        row = conn.execute(
            "SELECT canonical_key, climb_id, raw_key, sequence_length "
            "FROM subpattern_occurrences WHERE id = ?",
            (rand_id,),
        ).fetchone()
        if row is None:
            continue
        canonical_key, climb_id, raw_key, seq_len = row
        if seq_len != sequence_length or canonical_key == exclude_canonical_key:
            continue
        return {"raw_key": raw_key, "climb_id": climb_id}
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_training_pairs(db_path, output_path, max_anchors, hops, seed, grid_step_override):
    rng = random.Random(seed)
    # Separate stream so adding the type-flip negative doesn't change which
    # hard negatives the geometry-shift rng produces for a given seed.
    flip_rng = random.Random(seed + 1)
    random.seed(seed)  # find_easy_negative uses module-level random
    conn = sqlite3.connect(db_path)
    ensure_indexes(conn)

    max_id = conn.execute("SELECT MAX(id) FROM subpattern_occurrences").fetchone()[0]
    if not max_id:
        print("subpattern_occurrences is empty -- run backfill_subpatterns.py first.")
        return

    print("Detecting grid step...")
    grid_step = grid_step_override or infer_grid_step(conn)

    print("Finding canonical_keys with >=2 occurrences (positive-pair candidates)...")
    cur = conn.execute(
        "SELECT canonical_key, COUNT(*) as c FROM subpattern_occurrences "
        "GROUP BY canonical_key HAVING c >= 2"
    )

    written = 0
    start = time.time()

    with open(output_path, "w") as out:
        for canonical_key, _count in cur:
            if max_anchors and written >= max_anchors:
                break

            occurrences = fetch_occurrences_for_key(conn, canonical_key)
            anchor, positive = pick_positive_pair(occurrences)
            if anchor is None:
                continue

            sequence_length = anchor[3]

            hard_negative_raw_key = make_synthetic_hard_negative(
                anchor[2], grid_step, rng, hops=hops
            )

            type_flip_negative_raw_key = make_type_flip_negative(anchor[2], flip_rng)

            easy_negative = find_easy_negative(conn, canonical_key, sequence_length, max_id)

            record = {
                "sequence_length": sequence_length,
                "anchor_raw_key": anchor[2],
                "anchor_climb_id": anchor[1],
                "positive_raw_key": positive[2],
                "positive_climb_id": positive[1],
                "hard_negative_raw_key": hard_negative_raw_key,
                "hard_negative_climb_id": None,  # synthetic, not a real climb
                "type_flip_negative_raw_key": type_flip_negative_raw_key,
                "easy_negative_raw_key": easy_negative["raw_key"] if easy_negative else None,
                "easy_negative_climb_id": easy_negative["climb_id"] if easy_negative else None,
            }
            out.write(json.dumps(record) + "\n")
            written += 1

            if written % 5000 == 0:
                elapsed = time.time() - start
                print(f"  {written} records written ({elapsed:.1f}s elapsed)")

    conn.close()
    print(f"Done. {written} records written to {output_path}.")
    print(f"  Grid step used for hard-negative perturbation: {grid_step}")


def main():
    parser = argparse.ArgumentParser(description="Build positive/hard-negative/easy-negative training pairs")
    parser.add_argument("--db", default=DB_PATH_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--max-anchors", type=int, default=0, help="0 = no cap")
    parser.add_argument("--hops", type=int, default=1,
                         help="Hard negative shift, in grid-step units (1 = smallest real difference).")
    parser.add_argument("--grid-step", type=int, default=0,
                         help="Override auto-detected grid step (0 = auto-detect).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_training_pairs(args.db, args.output, args.max_anchors, args.hops, args.seed, args.grid_step)


if __name__ == "__main__":
    main()