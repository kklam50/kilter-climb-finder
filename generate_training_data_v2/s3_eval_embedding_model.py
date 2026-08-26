"""
Implements plan.md Section 4.3: evaluate the trained model before proceeding
to Phase 2 (full embedding index over ~11.7M rows).

Checks, on a held-out validation split, whether:
  mean cosine_similarity(anchor, positive)      >  mean cosine_similarity(anchor, hard_negative)
  mean cosine_similarity(anchor, hard_negative) >  mean cosine_similarity(anchor, easy_negative)

This is the required gate from the plan: if positives and hard negatives
aren't measurably separated, the model has not learned the near-miss
boundary that is the entire reason it exists over the exact-match registry,
and Phase 2 should not proceed until Section 4.1 (hard-negative construction,
e.g. --hops) or Section 4.2 (training config, e.g. base model, triplet
margin, epochs) is revisited.

Uses the SAME train/val split logic as train_embedding_model.py so the
validation set here matches what training held out. Pass the same
--val-fraction and --seed used at training time.
"""

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util

from generate_training_data_v2.s2_train_embedding_model import load_records, split_by_climb


def collect_eval_texts(val_records):
    """
    Returns (anchors, positives, hard_negs, easy_negs) as parallel lists,
    restricted to records where both a hard and easy negative are present
    so all three comparisons can be made on the same set of anchors.
    """
    anchors, positives, hard_negs, easy_negs = [], [], [], []
    for r in val_records:
        if not r.get("hard_negative_raw_key") or not r.get("easy_negative_raw_key"):
            continue
        anchors.append(r["anchor_raw_key"])
        positives.append(r["positive_raw_key"])
        hard_negs.append(r["hard_negative_raw_key"])
        easy_negs.append(r["easy_negative_raw_key"])
    return anchors, positives, hard_negs, easy_negs


def batch_encode_unique(model, *text_lists, batch_size=256):
    """Encode the union of all texts once, return a lookup dict text -> vector."""
    all_texts = sorted(set(t for lst in text_lists for t in lst))
    embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return dict(zip(all_texts, embeddings))


def paired_cosine(lookup, list_a, list_b):
    vecs_a = np.array([lookup[t] for t in list_a])
    vecs_b = np.array([lookup[t] for t in list_b])
    sims = util.cos_sim(vecs_a, vecs_b)
    return np.diag(sims.numpy())


def summarize(name, sims):
    print(f"  {name:20s} mean={sims.mean():.4f}  median={np.median(sims):.4f}  "
          f"std={sims.std():.4f}  min={sims.min():.4f}  max={sims.max():.4f}")


def evaluate(args):
    print(f"Loading model from {args.model_dir}...")
    model = SentenceTransformer(args.model_dir)

    print(f"Loading pairs from {args.pairs_file}...")
    records = load_records(args.pairs_file)
    _, val_records = split_by_climb(records, args.val_fraction, args.seed)
    print(f"  {len(val_records)} val records (split by climb_id, matching training)")

    anchors, positives, hard_negs, easy_negs = collect_eval_texts(val_records)
    print(f"  {len(anchors)} usable records (have both hard_negative and easy_negative)")
    if len(anchors) < 50:
        print("  WARNING: very small eval set -- results below may not be reliable.")

    print("\nEncoding...")
    lookup = batch_encode_unique(model, anchors, positives, hard_negs, easy_negs)

    sim_pos = paired_cosine(lookup, anchors, positives)
    sim_hard = paired_cosine(lookup, anchors, hard_negs)
    sim_easy = paired_cosine(lookup, anchors, easy_negs)

    print("\n=== Similarity distributions (anchor vs. each) ===")
    summarize("positive", sim_pos)
    summarize("hard_negative", sim_hard)
    summarize("easy_negative", sim_easy)

    pos_gt_hard = float(np.mean(sim_pos > sim_hard))
    hard_gt_easy = float(np.mean(sim_hard > sim_easy))

    print("\n=== Pairwise ordering checks ===")
    print(f"  sim(pos) > sim(hard_negative) in {pos_gt_hard*100:.1f}% of records "
          f"(this is the primary signal -- should be well above 50%, ideally >85-90%)")
    print(f"  sim(hard_negative) > sim(easy_negative) in {hard_gt_easy*100:.1f}% of records "
          f"(confirms hard negatives are genuinely 'harder' than random negatives)")

    mean_gap_pos_hard = sim_pos.mean() - sim_hard.mean()
    mean_gap_hard_easy = sim_hard.mean() - sim_easy.mean()

    print("\n=== Verdict ===")
    passed = True
    if mean_gap_pos_hard <= 0 or pos_gt_hard < args.pass_threshold:
        print(f"  FAIL: positives are not clearly separated from hard negatives "
              f"(mean gap={mean_gap_pos_hard:.4f}, pairwise win rate={pos_gt_hard*100:.1f}%).")
        print("  Do not proceed to Phase 2 yet. Revisit:")
        print("    - 4.1: is --hops too small/large relative to the grid step?")
        print("    - 4.2: try more Stage 2 epochs, a different --triplet-margin, "
              "or a different --base-model (all-MiniLM-L6-v2 is a natural-language "
              "checkpoint repurposed for symbolic dx,dy strings -- may not be ideal).")
        passed = False
    else:
        print(f"  PASS: positives separated from hard negatives "
              f"(mean gap={mean_gap_pos_hard:.4f}, pairwise win rate={pos_gt_hard*100:.1f}%).")

    if mean_gap_hard_easy <= 0:
        print(f"  NOTE: hard negatives are not measurably harder than easy negatives "
              f"(mean gap={mean_gap_hard_easy:.4f}). Not a hard blocker on its own, but "
              f"worth understanding -- it may mean the synthetic hard-negative shift "
              f"({'--hops in build_training_pairs.py'}) is too large to actually be 'hard'.")

    print(f"\n{'Safe to proceed to Phase 2 (full embedding index).' if passed else 'Do NOT proceed to Phase 2 yet.'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained bi-encoder before Phase 2 (plan.md 4.3)")
    parser.add_argument("--model-dir", default="movement_encoder_model/stage2")
    parser.add_argument("--pairs-file", default="training_pairs.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                         help="Must match the value used in train_embedding_model.py")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the value used in train_embedding_model.py")
    parser.add_argument("--pass-threshold", type=float, default=0.85,
                         help="Minimum fraction of records where sim(pos) > sim(hard_negative) to pass")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()