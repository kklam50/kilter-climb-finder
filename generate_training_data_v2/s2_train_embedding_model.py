"""
Implements plan.md Section 4.2: train a contrastive embedding model
(bi-encoder) on the pairs produced by build_training_pairs.py (4.1).

Two-stage training:
  Stage 1 - MultipleNegativesRankingLoss on (anchor, positive) pairs only.
            Needs no explicit negatives -- in-batch positives from other
            examples serve as implicit negatives. Always runs; this is the
            simplest correct starting point per plan.md 4.2.
  Stage 2 - TripletLoss on (anchor, positive, hard_negative), restricted to
            records where 4.1 found a hard negative. Fine-tunes the Stage 1
            model to specifically separate near-miss windows, which is the
            behavior an exact-match registry cannot provide and the whole
            reason this model exists. Skipped automatically if too few
            hard-negative triplets are available (see --min-triplets).

Input representation: raw_key strings exactly as produced by
extract_canonical_windows / build_training_pairs.py, e.g.
"dx=8,dy=40 | dx=48,dy=24 | dx=-24,dy=0" -- tokenized as plain text.
No coordinate-specific preprocessing; the model must learn the tolerance
structure from data, not have it pre-encoded.

Train/val split is done by climb_id (not by individual pair), so no climb's
windows appear in both sets -- prevents leakage from near-duplicate windows
within the same climb.

Base model default: sentence-transformers/all-MiniLM-L6-v2. This is a
general natural-language checkpoint being repurposed for short symbolic
strings -- a reasonable, well-supported starting point, but not verified
against this specific input distribution. If Stage 2 evaluation (plan.md
4.3) shows poor separation, revisit whether a from-scratch tokenizer/small
transformer suited to symbolic input performs better before assuming the
training procedure itself is the problem.

Requires: pip install sentence-transformers --break-system-packages
"""

import json
import random
import argparse
from collections import defaultdict

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def load_records(pairs_path):
    records = []
    with open(pairs_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_by_climb(records, val_fraction, seed):
    """
    Split by anchor_climb_id so no climb's windows leak across train/val.
    """
    climb_ids = sorted({r["anchor_climb_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(climb_ids)

    n_val = max(1, int(len(climb_ids) * val_fraction))
    val_climb_ids = set(climb_ids[:n_val])

    train, val = [], []
    for r in records:
        (val if r["anchor_climb_id"] in val_climb_ids else train).append(r)
    return train, val


def build_mnrl_examples(records):
    """(anchor, positive) pairs -- always available."""
    return [
        InputExample(texts=[r["anchor_raw_key"], r["positive_raw_key"]])
        for r in records
    ]


def build_triplet_examples(records):
    """(anchor, positive, hard_negative) -- only where 4.1 found a hard negative."""
    return [
        InputExample(texts=[r["anchor_raw_key"], r["positive_raw_key"], r["hard_negative_raw_key"]])
        for r in records
        if r.get("hard_negative_raw_key")
    ]


def build_triplet_eval_lists(records):
    """
    TripletEvaluator wants three parallel lists (anchors, positives, negatives).
    Falls back to easy_negative if no hard_negative exists for a record, so
    validation coverage isn't limited to only the subset with hard negatives.
    """
    anchors, positives, negatives = [], [], []
    for r in records:
        neg = r.get("hard_negative_raw_key") or r.get("easy_negative_raw_key")
        if neg is None:
            continue
        anchors.append(r["anchor_raw_key"])
        positives.append(r["positive_raw_key"])
        negatives.append(neg)
    return anchors, positives, negatives


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"Loading records from {args.pairs_file}...")
    records = load_records(args.pairs_file)
    print(f"  {len(records)} total records")

    train_records, val_records = split_by_climb(records, args.val_fraction, args.seed)
    print(f"  {len(train_records)} train / {len(val_records)} val (split by climb_id)")

    n_triplet_train = sum(1 for r in train_records if r.get("hard_negative_raw_key"))
    print(f"  {n_triplet_train} train records have a hard negative available")

    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    val_anchors, val_positives, val_negatives = build_triplet_eval_lists(val_records)
    evaluator = None
    if val_anchors:
        evaluator = TripletEvaluator(
            anchors=val_anchors,
            positives=val_positives,
            negatives=val_negatives,
            name="val",
        )
    else:
        print("  WARNING: no usable validation triplets -- proceeding without an evaluator.")

    # --- Stage 1: MultipleNegativesRankingLoss on (anchor, positive) pairs ---
    print("\n=== Stage 1: MultipleNegativesRankingLoss ===")
    mnrl_examples = build_mnrl_examples(train_records)
    print(f"  {len(mnrl_examples)} training pairs")
    mnrl_loader = DataLoader(mnrl_examples, shuffle=True, batch_size=args.batch_size)
    mnrl_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(mnrl_loader, mnrl_loss)],
        evaluator=evaluator,
        epochs=args.stage1_epochs,
        warmup_steps=int(0.1 * len(mnrl_loader) * args.stage1_epochs),
        output_path=f"{args.output_dir}/stage1",
        save_best_model=bool(evaluator),
        show_progress_bar=True,
    )

    if evaluator:
        score = evaluator(model)
        try: 
            print(score)
            print(f"  Stage 1 val triplet accuracy: {score[evaluator.primary_metric]:.4f}")
        except:
            print("couldn't print score for stage 1") 

    # --- Stage 2: TripletLoss on (anchor, positive, hard_negative) ---
    triplet_examples = build_triplet_examples(train_records)
    if len(triplet_examples) < args.min_triplets:
        print(
            f"\n=== Stage 2 skipped: only {len(triplet_examples)} hard-negative "
            f"triplets available (< --min-triplets {args.min_triplets}) ==="
        )
        print(
            "This likely means 4.1's neighbor-bin search is finding few near-miss "
            "windows -- check --neighbor-hops in build_training_pairs.py, or the "
            "dataset may simply be too sparse in movement-space for many climbs "
            "to have close-but-different neighbors."
        )
        final_path = f"{args.output_dir}/stage1"
    else:
        print(f"\n=== Stage 2: TripletLoss ===")
        print(f"  {len(triplet_examples)} training triplets")
        triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=args.batch_size)
        triplet_loss = losses.TripletLoss(model, triplet_margin=args.triplet_margin)

        model.fit(
            train_objectives=[(triplet_loader, triplet_loss)],
            evaluator=evaluator,
            epochs=args.stage2_epochs,
            warmup_steps=int(0.1 * len(triplet_loader) * args.stage2_epochs),
            output_path=f"{args.output_dir}/stage2",
            save_best_model=bool(evaluator),
            show_progress_bar=True,
        )
        if evaluator:
            score = evaluator(model)
            try: 
                print(score)
                print(f"  Stage 2 val triplet accuracy: {score[evaluator.primary_metric]:.4f}")
            except:
                print("couldn't print score for stage 2") 
        final_path = f"{args.output_dir}/stage2"

    print(f"\nDone. Final model at: {final_path}")
    print("Next: plan.md Section 4.3 -- run the held-out separation check before "
          "proceeding to Phase 2 (full embedding index).")


def main():
    parser = argparse.ArgumentParser(description="Train the movement-window bi-encoder (plan.md 4.2)")
    parser.add_argument("--pairs-file", default="training_pairs.jsonl", help="Output of build_training_pairs.py")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="movement_encoder_model")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=1) # ~1hr each epoch
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--triplet-margin", type=float, default=0.5,
                         help="Sentence-transformers TripletLoss default margin; tune based on 4.3 results.")
    parser.add_argument("--min-triplets", type=int, default=1000,
                         help="Minimum hard-negative triplets required to run Stage 2 at all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()