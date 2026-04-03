
import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="PH Multi-Hazard Watch v6", layout="wide")

DEFAULT_BOUNDS = {
    "min_lat": -50.0,
    "max_lat": 45.0,
    "min_lon": 90.0,
    "max_lon": -120.0,
}
PH_CENTER = {"lat": 12.8797, "lon": 121.7740}

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
PAGASA_BULLETIN_URL = "https://www.pagasa.dost.gov.ph/tropical-cyclone/severe-weather-bulletin"
PHIVOLCS_ALERT_URL = "https://hazardhunter.georisk.gov.ph/monitoring/volcano"
GVP_CURRENT_URL = "https://volcano.si.edu/gvp_currenteruptions.cfm"

VOLCANO_META = {
    "Mayon": {"lat": 13.257, "lon": 123.685, "country": "Philippines", "aliases": ["Mayon"]},
    "Kanlaon": {"lat": 10.412, "lon": 123.132, "country": "Philippines", "aliases": ["Kanlaon", "Canlaon"]},
    "Taal": {"lat": 14.002, "lon": 120.993, "country": "Philippines", "aliases": ["Taal"]},
    "Bulusan": {"lat": 12.770, "lon": 124.050, "country": "Philippines", "aliases": ["Bulusan"]},
    "Pinatubo": {"lat": 15.130, "lon": 120.350, "country": "Philippines", "aliases": ["Pinatubo"]},
    "Lewotobi": {"lat": -8.530, "lon": 122.775, "country": "Indonesia", "aliases": ["Lewotobi", "Lewotobi Laki-Laki"]},
    "Merapi": {"lat": -7.540, "lon": 110.446, "country": "Indonesia", "aliases": ["Merapi"]},
    "Semeru": {"lat": -8.108, "lon": 112.922, "country": "Indonesia", "aliases": ["Semeru"]},
    "Anak Krakatau": {"lat": -6.102, "lon": 105.423, "country": "Indonesia", "aliases": ["Anak Krakatau", "Krakatau"]},
    "Aso": {"lat": 32.884, "lon": 131.104, "country": "Japan", "aliases": ["Aso"]},
    "Sakurajima": {"lat": 31.585, "lon": 130.657, "country": "Japan", "aliases": ["Sakurajima"]},
    "Whakaari/White Island": {"lat": -37.521, "lon": 177.183, "country": "New Zealand", "aliases": ["Whakaari", "White Island", "Whakaari/White Island"]},
    "Ruapehu": {"lat": -39.281, "lon": 175.570, "country": "New Zealand", "aliases": ["Ruapehu"]},
    "Changbaishan": {"lat": 41.980, "lon": 128.080, "country": "China / DPRK", "aliases": ["Changbaishan", "Paektu"]},
}

def safe_get(url, params=None, timeout=25, headers=None):
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; Streamlit MultiHazard/1.0)"}
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers)
        r.raise_for_status()
        return r
    except Exception as e:
        return e

def to_360(lon):
    return lon if lon >= 0 else lon + 360

def in_extent(lat, lon, min_lat, max_lat, min_lon, max_lon):
    if not (min_lat <= lat <= max_lat):
        return False
    lo = to_360(lon)
    a = to_360(min_lon)
    b = to_360(max_lon)
    if a <= b:
        return a <= lo <= b
    return lo >= a or lo <= b

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def influence_band(distance_km):
    if distance_km <= 500:
        return "Very near"
    if distance_km <= 1500:
        return "Near"
    if distance_km <= 3000:
        return "Regional"
    if distance_km <= 6000:
        return "Far but relevant"
    return "Very far"

def score_eq(mag, depth_km, distance_km):
    mag_term = max(0.0, (mag - 4.5) * 18.0)
    depth_bonus = 10 if depth_km <= 70 else 4 if depth_km <= 300 else 0
    distance_factor = max(0.05, 1.0 - min(distance_km, 6500) / 7000.0)
    return round((mag_term + depth_bonus) * distance_factor, 1)

def score_typhoon(text, distance_km):
    txt = (text or "").lower()
    if "super typhoon" in txt:
        base = 90
    elif "typhoon" in txt:
        base = 72
    elif "tropical storm" in txt:
        base = 48
    elif "depression" in txt:
        base = 30
    else:
        base = 12
    distance_factor = max(0.2, 1.0 - min(distance_km, 3500) / 4000.0)
    return round(base * distance_factor, 1)

def score_volcano(alert_level, distance_km, source="PHIVOLCS"):
    try:
        level_num = int(str(alert_level).strip())
    except Exception:
        level_num = 1
    base = 20 + level_num * 18 if source == "PHIVOLCS" else 42
    distance_factor = max(0.1, 1.0 - min(distance_km, 6500) / 7000.0)
    return round(base * distance_factor, 1)

def classify(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    if score > 0:
        return "Low"
    return "Info"

def event_color(level):
    if level == "High":
        return "#ef4444"
    if level == "Moderate":
        return "#f59e0b"
    if level == "Low":
        return "#10b981"
    return "#60a5fa"

def indicator(text, level):
    emoji = {"High": "🔴", "Moderate": "🟠", "Low": "🟢", "Info": "🔵"}.get(level, "🔵")
    return f"{emoji} {text}"

def best_meta_match(name):
    n = str(name).lower()
    for key, meta in VOLCANO_META.items():
        for alias in meta["aliases"]:
            if alias.lower() in n:
                return key, meta
    return None, None

def parse_since_date(text):
    patterns = [
        r"Since\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"since\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Start Date\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Started\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return pd.to_datetime(m.group(1), utc=True)
            except Exception:
                pass
    return pd.Timestamp.now(tz="UTC")

@st.cache_data(ttl=900, show_spinner=False)
def load_earthquakes(min_mag=5.0, days_back=14, bounds=None):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
        "orderby": "time-asc",
        "limit": 1000,
    }
    resp = safe_get(USGS_QUERY_URL, params=params)
    if isinstance(resp, Exception):
        return pd.DataFrame(), str(resp)
    rows = []
    bounds = bounds or DEFAULT_BOUNDS
    for f in resp.json().get("features", []):
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        if len(coords) < 3:
            continue
        lon, lat, depth = coords[0], coords[1], coords[2]
        if lat is None or lon is None:
            continue
        if not in_extent(lat, lon, bounds["min_lat"], bounds["max_lat"], bounds["min_lon"], bounds["max_lon"]):
            continue
        props = f.get("properties", {})
        mag = props.get("mag") or 0.0
        dist = haversine_km(lat, lon, PH_CENTER["lat"], PH_CENTER["lon"])
        score = score_eq(mag, depth or 0, dist)
        t = datetime.fromtimestamp(props.get("time", 0) / 1000, tz=timezone.utc)
        rows.append({
            "hazard_type": "Earthquake",
            "name": props.get("title", "Earthquake"),
            "source": "USGS",
            "time_utc": t,
            "lat": lat,
            "lon": lon,
            "magnitude": mag,
            "depth_km": depth,
            "distance_to_ph_km": round(dist, 0),
            "influence_band": influence_band(dist),
            "risk_score": score,
            "risk_level": classify(score),
            "details": props.get("place", ""),
            "url": props.get("url", ""),
        })
    return pd.DataFrame(rows), None

@st.cache_data(ttl=1800, show_spinner=False)
def load_typhoon_actual():
    resp = safe_get(PAGASA_BULLETIN_URL)
    if isinstance(resp, Exception):
        return pd.DataFrame(), str(resp)
    html = resp.text
    if "No Active Tropical Cyclone within the Philippine Area of Responsibility" in html:
        return pd.DataFrame(), None
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Active PAGASA tropical cyclone bulletin"
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    name_match = re.search(r"(Tropical Depression|Tropical Storm|Typhoon|Super Typhoon)\s+([A-Z][A-Z0-9\-]+)", text, re.IGNORECASE)
    label = f"{name_match.group(1).title()} {name_match.group(2).upper()}" if name_match else title
    score = score_typhoon(label, 0.0)
    return pd.DataFrame([{
        "hazard_type": "Typhoon",
        "name": label,
        "source": "PAGASA",
        "time_utc": datetime.now(timezone.utc),
        "lat": PH_CENTER["lat"],
        "lon": PH_CENTER["lon"],
        "magnitude": None,
        "depth_km": None,
        "distance_to_ph_km": 0.0,
        "influence_band": "Within PH context",
        "risk_score": score,
        "risk_level": classify(score),
        "details": "Active PAGASA tropical cyclone bulletin detected.",
        "url": PAGASA_BULLETIN_URL,
    }]), None

@st.cache_data(ttl=3600, show_spinner=False)
def load_phivolcs_actual_volcanoes():
    resp = safe_get(PHIVOLCS_ALERT_URL)
    if isinstance(resp, Exception):
        return pd.DataFrame(), str(resp)
    html = resp.text
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    rows = []
    for key, meta in VOLCANO_META.items():
        if meta["country"] != "Philippines":
            continue
        pat = rf"{re.escape(key)}.*?Alert Level\s*([0-5]).*?(Since\s+[A-Za-z]+\s+\d{{1,2}},\s+\d{{4}})"
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        alert_level = m.group(1)
        since_text = m.group(2)
        t = parse_since_date(since_text)
        dist = haversine_km(meta["lat"], meta["lon"], PH_CENTER["lat"], PH_CENTER["lon"])
        score = score_volcano(alert_level, dist, source="PHIVOLCS")
        rows.append({
            "hazard_type": "Volcano",
            "name": key,
            "source": "PHIVOLCS / HazardHunter",
            "time_utc": t,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "magnitude": None,
            "depth_km": None,
            "distance_to_ph_km": round(dist, 0),
            "influence_band": influence_band(dist),
            "risk_score": score,
            "risk_level": classify(score),
            "details": f"Alert Level {alert_level}. {since_text}.",
            "url": PHIVOLCS_ALERT_URL,
        })
    return pd.DataFrame(rows), None

@st.cache_data(ttl=3600, show_spinner=False)
def load_gvp_actual_volcanoes():
    try:
        tables = pd.read_html(GVP_CURRENT_URL)
    except Exception as e:
        return pd.DataFrame(), str(e)
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    seen = set()
    for tbl in tables:
        tbl.columns = [str(c).strip() for c in tbl.columns]
        if "volcano" not in " | ".join(tbl.columns).lower():
            continue
        for _, r in tbl.iterrows():
            raw_name = " ".join([str(v) for v in r.tolist() if pd.notna(v)])
            key, meta = best_meta_match(raw_name)
            if not meta or key in seen:
                continue
            seen.add(key)
            dist = haversine_km(meta["lat"], meta["lon"], PH_CENTER["lat"], PH_CENTER["lon"])
            time_guess = now
            for cell in r.tolist():
                dt = parse_since_date(str(cell))
                if abs((dt - now).days) > 0:
                    time_guess = dt
                    break
            score = score_volcano(2, dist, source="GVP")
            rows.append({
                "hazard_type": "Volcano",
                "name": key,
                "source": "Smithsonian GVP",
                "time_utc": time_guess,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "magnitude": None,
                "depth_km": None,
                "distance_to_ph_km": round(dist, 0),
                "influence_band": influence_band(dist),
                "risk_score": score,
                "risk_level": classify(score),
                "details": "Current eruption / active volcanic activity detected from Smithsonian GVP current eruptions page.",
                "url": GVP_CURRENT_URL,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["distance_to_ph_km", "name"]).drop_duplicates(subset=["name"], keep="first")
    return df, None

def combine_actual_events(*dfs):
    frames = [df.copy() for df in dfs if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True)
    return out.sort_values(["time_utc", "risk_score"], ascending=[True, False])

def build_day_ledger(events_df, sequence_days):
    today = datetime.now(timezone.utc).date()
    rows = []
    work = events_df.copy()
    if not work.empty:
        work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date
    for n in range(sequence_days, 0, -1):
        target_day = today - timedelta(days=n)
        day_df = work[work["event_date"] == target_day].sort_values(["time_utc", "risk_score"], ascending=[True, False]) if not work.empty else pd.DataFrame()
        seq_parts = []
        for _, r in day_df.iterrows():
            tag = f"{r['hazard_type']}: {r['name']}"
            if pd.notna(r.get("magnitude")):
                tag += f", M{r['magnitude']}"
            seq_parts.append(tag)
        rows.append({
            "days_ago": n,
            "date": target_day.isoformat(),
            "event_count": len(day_df),
            "sequence": "  |  ".join(seq_parts) if seq_parts else "No actual events"
        })
    return pd.DataFrame(rows)

def get_exact_day(events_df, days_ago):
    if events_df.empty:
        return events_df.copy()
    target_day = datetime.now(timezone.utc).date() - timedelta(days=days_ago)
    work = events_df.copy()
    work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date
    return work[work["event_date"] == target_day].sort_values(["time_utc", "risk_score"], ascending=[True, False])

def get_leadup(events_df, days_ago):
    if events_df.empty:
        return events_df.copy()
    today = datetime.now(timezone.utc).date()
    start_day = today - timedelta(days=days_ago)
    end_day = today - timedelta(days=1)
    work = events_df.copy()
    work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date
    return work[(work["event_date"] >= start_day) & (work["event_date"] <= end_day)].sort_values(["time_utc", "risk_score"], ascending=[True, False])

def filter_map_df(events_df, map_mode, focus_days_ago):
    if events_df.empty:
        return events_df.copy()
    work = events_df.copy()
    work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date
    today = datetime.now(timezone.utc).date()
    if map_mode == "All actual events in source window":
        out = work
    elif map_mode == "Exact selected past day only":
        target = today - timedelta(days=focus_days_ago)
        out = work[work["event_date"] == target]
    else:  # Lead-up from selected day to yesterday
        start_day = today - timedelta(days=focus_days_ago)
        end_day = today - timedelta(days=1)
        out = work[(work["event_date"] >= start_day) & (work["event_date"] <= end_day)]
    out = out.copy()
    if out.empty:
        return out
    # dynamic radius scaling based on current filtered view
    max_score = max(float(out["risk_score"].max()), 1.0)
    out["radius"] = out["risk_score"].apply(lambda s: 150000 + 850000 * (float(s) / max_score))
    return out

def build_timeline_html(df):
    if df.empty:
        return "<div style='padding:8px;'>No actual events in this selection.</div>"
    work = df.copy()
    work["time_str"] = pd.to_datetime(work["time_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    cards = []
    for _, r in work.iterrows():
        color = event_color(r["risk_level"])
        mag_txt = f" | M {r['magnitude']}" if pd.notna(r.get("magnitude")) else ""
        cards.append(f"""
        <div style="min-width:340px; max-width:340px; background:white; border-left:8px solid {color};
                    border-radius:12px; padding:12px; margin-right:12px; box-shadow:0 1px 4px rgba(0,0,0,0.12);">
            <div style="font-size:12px; color:#555;">{r['time_str']}</div>
            <div style="font-weight:700; font-size:18px; margin-top:4px;">{r['hazard_type']}</div>
            <div style="font-weight:600; margin-top:4px;">{r['name']}</div>
            <div style="font-size:13px; color:#333; margin-top:6px;">
                Score: {r['risk_score']} | {r['risk_level']}{mag_txt}
            </div>
            <div style="font-size:13px; color:#444; margin-top:6px;">
                Distance to PH: {r['distance_to_ph_km']} km | {r['influence_band']}
            </div>
            <div style="font-size:12px; color:#666; margin-top:8px;">
                {str(r['details'])[:160]}
            </div>
        </div>
        """)
    return f"""
    <div style="overflow-x:auto; white-space:nowrap; padding:6px 0 10px 0; border:1px solid #e5e7eb; border-radius:12px; background:#f8fafc;">
        <div style="display:flex; flex-direction:row; padding:10px; align-items:stretch;">
            {''.join(cards)}
        </div>
    </div>
    """

st.sidebar.title("Controls")
sequence_days = st.sidebar.slider("Sequence depth (days ago back to yesterday)", 1, 14, 14)
focus_days_ago = st.sidebar.slider("Inspect this specific past day", 1, 14, 1)
min_mag = st.sidebar.slider("Minimum earthquake magnitude", 5.0, 8.0, 5.0, 0.1)
map_mode = st.sidebar.radio(
    "Map display mode",
    ["All actual events in source window", "Exact selected past day only", "Lead-up from selected day to yesterday"],
    index=1,
)

st.sidebar.subheader("Regional extent")
min_lat = st.sidebar.number_input("Min latitude", value=DEFAULT_BOUNDS["min_lat"])
max_lat = st.sidebar.number_input("Max latitude", value=DEFAULT_BOUNDS["max_lat"])
min_lon = st.sidebar.number_input("Min longitude", value=DEFAULT_BOUNDS["min_lon"])
max_lon = st.sidebar.number_input("Max longitude", value=DEFAULT_BOUNDS["max_lon"])
bounds = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}

include_eq = st.sidebar.checkbox("Include earthquakes", value=True)
include_typhoon = st.sidebar.checkbox("Include active typhoon bulletin", value=True)
include_ph_volcano = st.sidebar.checkbox("Include PHIVOLCS volcano events", value=True)
include_gvp_volcano = st.sidebar.checkbox("Include regional GVP eruptions", value=True)

st.title("Philippines Multi-Hazard Regional Watch v6")
st.caption("Now the map changes with the left controls. You can bind it to one exact past day or to the progressive lead-up period.")

eq_df, eq_err = load_earthquakes(min_mag=min_mag, days_back=14, bounds=bounds)
ty_df, ty_err = load_typhoon_actual()
ph_df, ph_err = load_phivolcs_actual_volcanoes()
gvp_df, gvp_err = load_gvp_actual_volcanoes()

actual_df = combine_actual_events(
    eq_df if include_eq else pd.DataFrame(),
    ty_df if include_typhoon else pd.DataFrame(),
    ph_df if include_ph_volcano else pd.DataFrame(),
    gvp_df if include_gvp_volcano else pd.DataFrame(),
)

seq_df = build_day_ledger(actual_df, sequence_days)
focus_df = get_exact_day(actual_df, focus_days_ago)
leadup_df = get_leadup(actual_df, focus_days_ago)
map_df = filter_map_df(actual_df, map_mode, focus_days_ago)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Earthquake rows", len(eq_df) if include_eq else 0)
c2.metric("Typhoon rows", len(ty_df) if include_typhoon else 0)
c3.metric("PH volcano rows", len(ph_df) if include_ph_volcano else 0)
c4.metric("Regional volcano rows", len(gvp_df) if include_gvp_volcano else 0)

if not actual_df.empty:
    top = actual_df.sort_values(["risk_score", "time_utc"], ascending=[False, False]).iloc[0]
    st.info(indicator(f"Top current regional concern: {top['name']} ({top['hazard_type']})", top["risk_level"]))

left, right = st.columns([2, 1])

with left:
    st.subheader("Regional hazard map")
    st.caption(f"Current map mode: {map_mode}")
    if map_df.empty:
        st.warning("No actual events to plot for the current map mode.")
    else:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_radius="radius",
            radius_min_pixels=6,
            radius_max_pixels=55,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color=[
                "risk_level == 'High' ? 220 : (risk_level == 'Moderate' ? 245 : 60)",
                "risk_level == 'High' ? 70 : (risk_level == 'Moderate' ? 160 : 170)",
                "risk_level == 'High' ? 70 : (risk_level == 'Moderate' ? 40 : 90)",
                150,
            ],
            get_line_color=[30, 30, 30, 180],
        )
        tooltip = {
            "html": "<b>{hazard_type}</b><br/>{name}<br/>Source: {source}<br/>Time: {time_utc}<br/>Score: {risk_score}<br/>Distance to PH: {distance_to_ph_km} km<br/>{details}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        # center map around filtered data if possible
        view_lat = float(map_df["lat"].mean())
        view_lon = float(map_df["lon"].mean())
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=3.0), tooltip=tooltip),
            use_container_width=True
        )

with right:
    st.subheader("Current mapped events")
    if map_df.empty:
        st.write("No mapped events.")
    else:
        st.dataframe(
            map_df.sort_values(["time_utc", "risk_score"], ascending=[True, False])[
                ["time_utc", "hazard_type", "name", "source", "risk_level", "risk_score", "distance_to_ph_km"]
            ],
            use_container_width=True,
            hide_index=True
        )

st.subheader("Past-day sequence ledger")
st.caption("Each row is one exact past day only.")
st.dataframe(seq_df, use_container_width=True, hide_index=True)

st.subheader(f"Exact sequence for {focus_days_ago} day(s) ago")
if focus_df.empty:
    st.warning("No actual events on the selected past day.")
else:
    st.components.v1.html(build_timeline_html(focus_df), height=270, scrolling=True)
    st.dataframe(
        focus_df[["time_utc", "hazard_type", "name", "source", "magnitude", "depth_km", "distance_to_ph_km", "risk_level", "risk_score", "details"]],
        use_container_width=True,
        hide_index=True
    )

st.subheader("Progressive lead-up sequence")
if leadup_df.empty:
    st.info("No actual lead-up events in this span.")
else:
    st.components.v1.html(build_timeline_html(leadup_df), height=270, scrolling=True)
    st.dataframe(
        leadup_df[["time_utc", "hazard_type", "name", "source", "magnitude", "depth_km", "distance_to_ph_km", "risk_level", "risk_score", "details"]],
        use_container_width=True,
        hide_index=True
    )

tab1, tab2, tab3 = st.tabs(["Actual volcano events", "All actual events", "Notes"])
with tab1:
    volcano_df = combine_actual_events(pd.DataFrame(), pd.DataFrame(), ph_df if include_ph_volcano else pd.DataFrame(), gvp_df if include_gvp_volcano else pd.DataFrame())
    if volcano_df.empty:
        st.warning("No actual volcano rows parsed.")
    else:
        st.dataframe(volcano_df, use_container_width=True, hide_index=True)

with tab2:
    if actual_df.empty:
        st.warning("No actual events.")
    else:
        st.dataframe(actual_df, use_container_width=True, hide_index=True)

with tab3:
    st.write(
        """
        The map did not change before because it was plotting the full actual-event dataset regardless of the sequence controls.
        In this version, the map can now switch between:
        1. all actual events,
        2. exact selected past day only,
        3. lead-up from selected day to yesterday.

        For the behavior you described, use:
        - Map display mode = **Exact selected past day only**
        - then move **Inspect this specific past day**

        That will visibly change the map points on the right.
        """
    )
