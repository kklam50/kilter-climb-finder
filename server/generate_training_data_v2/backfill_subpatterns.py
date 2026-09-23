"""
Backfill script: computes canonical movement subpatterns for every climb,
across every board layout, and stores them in a queryable SQLite table
(subpattern_occurrences).

Implements knn_spacing_representation_plan.md: a window is a hold's `k`
nearest neighbors by straight-line distance (not the old sorted-by-Y
consecutive-delta chain). WINDOW_SIZES values are now `k` -- reused as-is,
still stored in the `sequence_length` column, per the plan's Section 1.2.
"""

import sqlite3
import re
import time

DB_PATH = "../db/db.sqlite"
WINDOW_SIZES = [3, 4, 5]        # k: number of nearest neighbors per window
BATCH_CLIMBS = 500


# ---------------------------------------------------------------------------
# Reused verbatim from your existing scripts
# ---------------------------------------------------------------------------

def parse_frames_to_piece_map(frames_str, holes_table, placement_roles_table):
    """Parse frames string to extract hold coordinates."""
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
            continue  # skip invalid hole references

        hole_data = holes_table[hole_id]
        x, y = int(hole_data["x"]), int(hole_data["y"])
        product_id = hole_data["product_id"]

        role_key = (product_id, piece_type_id)
        symbol = "."
        if role_key in placement_roles_table:
            symbol = placement_roles_table[role_key].get("ascii_symbol", ".")

        piece_map.append({
            "x": x,
            "y": y,
            "type_id": piece_type_id,
            "symbol": symbol,
        })

    return piece_map


def extract_canonical_windows(piece_map, k=3):
    # Filters out symbol "X" holds -- keeps everything else. See note above
    # about this not being restricted to "M" (hand) holds specifically.
    m_holds = [hold for hold in piece_map if hold.get("symbol") != "X"]

    if len(m_holds) < k + 1:
        return []

    windows = []
    for anchor in m_holds:
        # Distance to every other hold; deterministic tiebreak
        # (dist_sq, dy, dx) so exact ties on the shared grid always resolve
        # the same way across backfill runs.
        others = []
        for h in m_holds:
            if h is anchor:
                continue
            dx = h['x'] - anchor['x']
            dy = h['y'] - anchor['y']
            others.append((dx * dx + dy * dy, dy, dx, h))
        others.sort(key=lambda o: (o[0], o[1], o[2]))
        nearest = others[:k]

        deltas = [(dx, dy) for _dist_sq, dy, dx, _h in nearest]

        # Invert dx if the nearest neighbor is to the left, for canonical
        # (mirror) normalization -- generalized from "first delta" (old,
        # sorted-by-Y scheme) to "nearest neighbor" (this scheme).
        if deltas[0][0] < 0:
            norm_win = [(-dx, dy) for dx, dy in deltas]
            is_mirrored = True
        else:
            norm_win = deltas
            is_mirrored = False

        loose_parts, raw_parts = [], []
        for dx, dy in norm_win:
            # +/-1 tolerance lives here: round to nearest even number
            normalized_dx = round(dx / 2) * 2
            normalized_dy = round(dy / 2) * 2
            loose_parts.append(f"dx={normalized_dx},dy={normalized_dy}")
            raw_parts.append(f"dx={dx},dy={dy}")

        windows.append({
            "canonical_key": " | ".join(loose_parts),
            "raw_key": " | ".join(raw_parts),
            "is_mirrored": is_mirrored,
            "start_hold": anchor,             # the neighborhood's center
            "end_hold": nearest[-1][3],        # farthest of the k neighbors
        })

    return windows


# ---------------------------------------------------------------------------
# Reference-table loading
# ---------------------------------------------------------------------------

def load_holes_table(conn, product_id):
    cur = conn.execute(
        "SELECT id, product_id, x, y FROM holes WHERE product_id = ?",
        (product_id,),
    )
    return {
        row[0]: {"id": row[0], "product_id": row[1], "x": row[2], "y": row[3]}
        for row in cur
    }


def load_placement_roles_table(conn, product_id):
    cur = conn.execute(
        """
        SELECT product_id, id, CASE
            WHEN full_name = 'Start' THEN 'S'
            WHEN full_name = 'Middle' THEN 'M'
            WHEN full_name = 'Finish' THEN 'F'
            WHEN full_name = 'Foot Only' THEN 'X'
            ELSE '.'
        END as ascii_symbol, full_name 
        FROM placement_roles WHERE product_id = ?
        """,
        (product_id,),
    )
    return {
        (row[0], row[1]): {"ascii_symbol": row[2], "full_name": row[3]}
        for row in cur
    }


def get_product_id_for_layout(conn, layout_id):
    row = conn.execute(
        "SELECT product_id FROM layouts WHERE id = ?",
        (layout_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No product_id found for layout_id={layout_id}")
    return row[0]


# ---------------------------------------------------------------------------
# Your existing "high quality" climbs query, unmodified
# ---------------------------------------------------------------------------

CLIMBS_QUERY = """
SELECT c.uuid, c.layout_id, c.name, c.description, c.frames, c.created_at
FROM climbs c
WHERE c.layout_id = ?
    AND EXISTS (
        SELECT 1 FROM climb_stats cs
        WHERE cs.climb_uuid = c.uuid
          AND cs.ascensionist_count > 5
          AND cs.angle IS NOT NULL
    )
"""


# ---------------------------------------------------------------------------
# Destination table
# ---------------------------------------------------------------------------

def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subpattern_occurrences (
            id INTEGER PRIMARY KEY,
            canonical_key TEXT NOT NULL,
            raw_key TEXT NOT NULL,
            climb_id TEXT NOT NULL,
            climb_name TEXT,
            start_coords TEXT,
            end_coords TEXT,
            sequence_length INTEGER,
            is_mirrored INTEGER
        )
    """)


def create_index(conn):
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subpattern_key "
        "ON subpattern_occurrences(canonical_key)"
    )


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def backfill():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    ensure_table(conn)

    layout_ids = [row[0] for row in conn.execute("SELECT DISTINCT layout_id FROM climbs")]
    print(f"Found {len(layout_ids)} layouts to process")

    total_climbs = 0
    total_windows = 0
    start = time.time()

    for layout_id in layout_ids:
        product_id = 1
        holes_table = load_holes_table(conn, product_id)
        placement_roles_table = load_placement_roles_table(conn, product_id)

        batch_rows = []
        climbs_in_batch = 0

        for row in conn.execute(CLIMBS_QUERY, (layout_id,)):
            uuid, _layout_id, name, _desc, frames, _created_at = row

            piece_map = parse_frames_to_piece_map(frames, holes_table, placement_roles_table)
            if not piece_map:
                continue

            for window_size in WINDOW_SIZES:
                for w in extract_canonical_windows(piece_map, window_size):
                    batch_rows.append((
                        w["canonical_key"],
                        w["raw_key"],
                        uuid,
                        name,
                        f"{w['start_hold']['x']},{w['start_hold']['y']}",
                        f"{w['end_hold']['x']},{w['end_hold']['y']}",
                        window_size,
                        int(w["is_mirrored"]),
                    ))

            total_climbs += 1
            climbs_in_batch += 1

            if climbs_in_batch >= BATCH_CLIMBS:
                _insert_batch(conn, batch_rows)
                conn.commit()
                total_windows += len(batch_rows)
                batch_rows = []
                climbs_in_batch = 0

        if batch_rows:
            _insert_batch(conn, batch_rows)
            conn.commit()
            total_windows += len(batch_rows)

        print(f"layout_id={layout_id}: {total_climbs} climbs processed so far")

    print("Building index on canonical_key...")
    create_index(conn)
    conn.execute("PRAGMA synchronous=FULL")
    conn.commit()
    conn.close()

    elapsed = time.time() - start
    print(f"Done. {total_climbs} climbs, {total_windows} window rows, {elapsed:.1f}s")


def _insert_batch(conn, rows):
    conn.executemany("""
        INSERT INTO subpattern_occurrences
        (canonical_key, raw_key, climb_id, climb_name,
         start_coords, end_coords, sequence_length, is_mirrored)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)


if __name__ == "__main__":
    backfill()