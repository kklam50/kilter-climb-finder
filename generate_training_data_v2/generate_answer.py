"""
Implements plan.md Section 7 (Phase 4): generation.

Wires retrieval.py's output into a general-purpose instruction-tuned model
served by LM Studio's OpenAI-compatible local server. This is the ONLY
place in the whole pipeline that calls a generator model -- everything
upstream (candidate filtering, similarity/opposite search, context
assembly) is retrieval.py doing plain SQL and math, per the architecture
decision made earlier: app retrieves, generator only formats.

Note on concurrent model serving (raised as an open question in plan.md
Section 7): it doesn't actually apply here. The embedding/bi-encoder model
runs in-process via sentence-transformers (see retrieval.py), never through
LM Studio. LM Studio only needs to serve this one generator model.

No coordinate or delta data ever reaches this file -- assemble_context()
in retrieval.py has already reduced matches down to climb name, grade,
angle, similarity score, and mirror/direct relation before it gets here.

Requires: pip install openai --break-system-packages
(only the OpenAI *client* library is needed -- no API key, no OpenAI account;
it's just pointed at LM Studio's local server.)
"""

import argparse
from openai import OpenAI

from retrieval import RetrievalEngine

SYSTEM_PROMPT = (
    "You are a climbing route recommendation assistant. Answer the user's "
    "question using ONLY the climbs listed below -- do not invent or assume "
    "climbs that aren't in the list. Cite climb names exactly as given, "
    "including their grade. If the list is empty or doesn't actually address "
    "the question, say so plainly rather than making something up."
)


def generate_answer(client, model, question, context):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Climbs:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


def answer_question(engine, client, generator_model, climb_id, question,
                     mode="similar", top_k=5):
    """
    Full pipeline: retrieval (Phase 3) -> context assembly -> generation (Phase 4).
    """
    matches = engine.recommend_for_climb(climb_id, mode=mode, top_k=top_k)
    context = engine.assemble_context(matches)
    return generate_answer(client, generator_model, question, context), matches


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test (plan.md Phase 4 / Section 9 step 7)")
    parser.add_argument("--db", default="../db/db.sqlite")
    parser.add_argument("--index-prefix", default="../db/subpattern_index")
    parser.add_argument("--climb-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["similar", "opposite"], default="similar")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-url", default="http://localhost:1234/v1",
                         help="LM Studio's local server endpoint")
    parser.add_argument("--generator-model", default="local-model",
                         help="LM Studio ignores/uses whatever's currently loaded for most setups; "
                              "set explicitly if you're serving multiple models")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    engine = RetrievalEngine(args.db, args.index_prefix)

    try:
        try:
            answer, matches = answer_question(
                engine, client, args.generator_model,
                args.climb_id, args.question, args.mode, args.top_k,
            )
        except Exception as e:
            print(f"Could not reach the generator at {args.base_url} -- "
                  f"is LM Studio's local server running with a model loaded?")
            print(f"  ({e})")
            return

        print(f"\n--- Retrieved context ({len(matches)} climbs, mode={args.mode}) ---")
        print(engine.assemble_context(matches))
        print(f"\n--- Generated answer ---")
        print(answer)
    finally:
        engine.close()


if __name__ == "__main__":
    main()