# PH Multi-Hazard Regional Watch v3

This version focuses on actual events.

## What changed
- The dynamic map now shows actual events only.
- The timeline now shows actual events only.
- The volcano watchlist is preserved as a separate reference table and is no longer injected into the map or timeline.
- The timeline supports a progressive window ending today, yesterday, two days ago, and so on, up to a 14-day past sequence.
- Added a lead-up table to inspect what happened before the largest event in the selected window.

## Included
- `app.py`
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
