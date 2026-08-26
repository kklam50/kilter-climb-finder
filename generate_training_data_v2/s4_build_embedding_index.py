"""
Implements plan.md Phase 2 (Section 5): build the embedding index.

Batch-embeds every row in subpattern_occurrences using the trained model
from Phase 1 (4.2/4.3), storing vectors in a numpy memmap file (fast for
the batch matrix operations Phase 3's similarity search will need) plus a
small SQLite mapping table (occurrence id -> memmap row index, and back).

Per plan.md: starts with memmap + SQLite, NOT FAISS/pgvector/a vector DB
service -- introduce those only if a measured query-latency problem in
Phase 3 justifies it.

Vectors are L2-normalized at encode time, so downstream similarity search
in Phase 3 is a plain dot product (equivalent to cosine similarity on unit
vectors, and faster than computing norms per query).

Outputs:
  <output-prefix>.dat        -- numpy memmap, shape (N, dim), dtype as configured
  <output-prefix>.meta.json  -- {dim, dtype, count, model_dir} for safe loading later
  <output-prefix>.sqlite     -- mapping table: occurrence_id INTEGER PRIMARY KEY, row_index INTEGER
"""

import sqlite3
import json
import argparse
import time
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH_DEFAULT = "../db/db.sqlite"
OUTPUT_PREFIX_DEFAULT = "subpattern_index"


def get_row_count(conn):
    return conn.execute("SELECT COUNT(*) FROM subpattern_occurrences").fetchone()[0]


def get_embedding_dim(model):
    return model.get_sentence_embedding_dimension()


def create_mapping_table(map_conn):
    map_conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_index (
            occurrence_id INTEGER PRIMARY KEY,
            row_index INTEGER NOT NULL
        )
    """)
    map_conn.commit()


def create_row_index_lookup(map_conn):
    map_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_row_index ON embedding_index(row_index)"
    )
    map_conn.commit()


def stream_occurrences(conn, batch_size):
    """
    Yields batches of (id, raw_key) ordered by id, using keyset pagination
    (WHERE id > last_seen) rather than OFFSET, so this stays fast even at
    the end of an 11.7M-row scan.
    """
    last_id = 0
    while True:
        rows = conn.execute(
            "SELECT id, raw_key FROM subpattern_occurrences "
            "WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break
        yield rows
        last_id = rows[-1][0]


def build_index(args):
    conn = sqlite3.connect(args.db)
    total = get_row_count(conn)
    if total == 0:
        print("subpattern_occurrences is empty -- run backfill_subpatterns.py first.")
        return

    print(f"Loading model from {args.model_dir}...")
    model = SentenceTransformer(args.model_dir)
    dim = get_embedding_dim(model)

    dtype = np.float16 if args.dtype == "float16" else np.float32
    bytes_per_vec = dim * np.dtype(dtype).itemsize
    est_size_gb = (total * bytes_per_vec) / (1024 ** 3)
    print(f"Rows to embed: {total:,}")
    print(f"Embedding dim: {dim}, dtype: {args.dtype}")
    print(f"Estimated index file size: {est_size_gb:.2f} GB")

    memmap_path = f"{args.output_prefix}.dat"
    meta_path = f"{args.output_prefix}.meta.json"
    mapping_db_path = f"{args.output_prefix}.sqlite"

    vectors = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=(total, dim))

    map_conn = sqlite3.connect(mapping_db_path)
    map_conn.execute("PRAGMA journal_mode=WAL")
    map_conn.execute("PRAGMA synchronous=OFF")
    create_mapping_table(map_conn)

    row_index = 0
    mapping_batch = []
    start = time.time()

    for batch in stream_occurrences(conn, args.batch_size):
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        embeddings = model.encode(
            texts,
            batch_size=args.encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors -> dot product == cosine similarity
            show_progress_bar=False,
        ).astype(dtype)

        n = len(ids)
        vectors[row_index:row_index + n] = embeddings
        for occurrence_id in ids:
            mapping_batch.append((occurrence_id, row_index))
            row_index += 1

        if len(mapping_batch) >= args.batch_size * 5:
            map_conn.executemany(
                "INSERT INTO embedding_index (occurrence_id, row_index) VALUES (?, ?)",
                mapping_batch,
            )
            map_conn.commit()
            mapping_batch = []

        if row_index % (args.batch_size * 20) < args.batch_size:
            elapsed = time.time() - start
            rate = row_index / elapsed if elapsed > 0 else 0
            remaining = (total - row_index) / rate if rate > 0 else float("inf")
            print(f"  {row_index:,}/{total:,} embedded "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    if mapping_batch:
        map_conn.executemany(
            "INSERT INTO embedding_index (occurrence_id, row_index) VALUES (?, ?)",
            mapping_batch,
        )
        map_conn.commit()

    print("Building index on row_index for reverse lookups...")
    create_row_index_lookup(map_conn)
    map_conn.close()

    vectors.flush()
    del vectors

    with open(meta_path, "w") as f:
        json.dump({
            "dim": dim,
            "dtype": args.dtype,
            "count": total,
            "model_dir": args.model_dir,
            "normalized": True,
        }, f, indent=2)

    conn.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"  Vectors: {memmap_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  ID mapping: {mapping_db_path}")


def main():
    parser = argparse.ArgumentParser(description="Build the embedding index (plan.md Phase 2)")
    parser.add_argument("--db", default=DB_PATH_DEFAULT)
    parser.add_argument("--model-dir", default="movement_encoder_model/stage2")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=2000, help="DB fetch batch size")
    parser.add_argument("--encode-batch-size", type=int, default=256, help="Model encode batch size")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    build_index(args)


if __name__ == "__main__":
    main()