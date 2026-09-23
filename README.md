---
title: Kilter Climb Finder
emoji: 🧗
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Kilter Climb Finder

Given a Kilter Board climb, finds other climbs with similar (or opposite)
movement patterns using a trained embedding model over per-hold subpatterns,
rather than plain grade/name search.

## How it works

- `server/generate_training_data_v2/` builds contrastive training pairs from
  climb move sequences, trains a sentence-transformers embedding model over
  "subpattern" windows, and builds a nearest-neighbor index over every climb
  in the dataset.
- `server/app.py` is a FastAPI service that loads that model + index once at
  startup and serves `/climbs/{climb_id}/recommendations`.
- `client/` is a small static frontend that looks up a climb by name and
  displays its nearest matches.

## Running locally

```
pip install -r requirements.txt
uvicorn server.app:app --reload
```

Then open http://localhost:8000/.
