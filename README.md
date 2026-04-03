# PH Multi-Hazard Regional Watch

A Streamlit dashboard concept for monitoring regional hazards that may influence the Philippines:
- Typhoon / tropical cyclone status
- Earthquakes with configurable minimum magnitude
- Volcano watchlist across the broader regional neighborhood
- 14-day sequence timeline with horizontal scroll and day-by-day event ledger

## Included
- `app.py`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Timeline feature
- Horizontal scroll timeline cards to compare event order
- Daily sequence ledger for the last 3 to 14 days
- Day inspector slider to review detailed order of events on a selected day

## Notes
- Earthquakes are live via USGS.
- Typhoon status reads the PAGASA tropical cyclone bulletin page.
- Volcanoes are still a curated regional watchlist in this version.
- Volcano timeline timestamps are placeholders for now until a full eruption/advisory parser is added.
