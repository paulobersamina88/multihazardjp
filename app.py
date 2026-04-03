
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="PH Multi-Hazard Watch", layout="wide")

DEFAULT_BOUNDS = {
    "min_lat": -50.0,
    "max_lat": 45.0,
    "min_lon": 90.0,
    "max_lon": -120.0,
}
PH_CENTER = {"lat": 12.8797, "lon": 121.7740}

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
PAGASA_BULLETIN_URL = "https://www.pagasa.dost.gov.ph/tropical-cyclone/severe-weather-bulletin"
PHIVOLCS_HOME = "https://www.phivolcs.dost.gov.ph/"
GVP_CURRENT = "https://volcano.si.edu/gvp_currenteruptions.cfm"
GVP_WEEKLY = "https://volcano.si.edu/reports_weekly.cfm"

VOLCANOES = [
    {"name": "Mayon", "country": "Philippines", "lat": 13.257, "lon": 123.685, "priority": "PH"},
    {"name": "Taal", "country": "Philippines", "lat": 14.002, "lon": 120.993, "priority": "PH"},
    {"name": "Kanlaon", "country": "Philippines", "lat": 10.412, "lon": 123.132, "priority": "PH"},
    {"name": "Bulusan", "country": "Philippines", "lat": 12.770, "lon": 124.050, "priority": "PH"},
    {"name": "Pinatubo", "country": "Philippines", "lat": 15.130, "lon": 120.350, "priority": "PH"},
    {"name": "Lewotobi Laki-Laki", "country": "Indonesia", "lat": -8.530, "lon": 122.775, "priority": "Regional"},
    {"name": "Merapi", "country": "Indonesia", "lat": -7.540, "lon": 110.446, "priority": "Regional"},
    {"name": "Semeru", "country": "Indonesia", "lat": -8.108, "lon": 112.922, "priority": "Regional"},
    {"name": "Anak Krakatau", "country": "Indonesia", "lat": -6.102, "lon": 105.423, "priority": "Regional"},
    {"name": "Aso", "country": "Japan", "lat": 32.884, "lon": 131.104, "priority": "Regional"},
    {"name": "Sakurajima", "country": "Japan", "lat": 31.585, "lon": 130.657, "priority": "Regional"},
    {"name": "Whakaari/White Island", "country": "New Zealand", "lat": -37.521, "lon": 177.183, "priority": "Regional"},
    {"name": "Ruapehu", "country": "New Zealand", "lat": -39.281, "lon": 175.570, "priority": "Regional"},
    {"name": "Changbaishan", "country": "China / DPRK", "lat": 41.980, "lon": 128.080, "priority": "Regional"},
]

# ----------------------------
# HELPERS
# ----------------------------
def parse_html_title(html: str) -> str:
    lower = html.lower()
    start = lower.find("<title>")
    end = lower.find("</title>")
    if start != -1 and end != -1 and end > start:
        return html[start + 7:end].strip()
    return ""

def safe_get(url, params=None, timeout=20, headers=None):
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

def score_typhoon(signal_text, distance_km):
    txt = signal_text.lower()
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

def classify(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    if score > 0:
        return "Low"
    return "Info"

def indicator(text, level):
    if level == "High":
        return f"🔴 {text}"
    if level == "Moderate":
        return f"🟠 {text}"
    if level == "Low":
        return f"🟢 {text}"
    return f"🔵 {text}"

def timeline_rank(hazard_type):
    order = {"Volcano": 1, "Earthquake": 2, "Typhoon": 3}
    return order.get(hazard_type, 9)

def event_color(level):
    if level == "High":
        return "#ef4444"
    if level == "Moderate":
        return "#f59e0b"
    if level == "Low":
        return "#10b981"
    return "#60a5fa"

def build_daily_sequence(events_df, days=14):
    if events_df.empty:
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days - 1)
    dates = pd.date_range(start=start.date(), end=now.date(), freq="D", tz=None)

    rows = []
    work = events_df.copy()
    work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date

    for d in dates:
        d_date = d.date()
        day_df = work[work["event_date"] == d_date].sort_values(
            by=["time_utc", "hazard_type"], ascending=[True, True]
        )
        seq_items = []
        for _, r in day_df.iterrows():
            label = f"{r['hazard_type']}: {str(r['name'])[:45]}"
            score = r.get("risk_score", 0)
            seq_items.append(f"{label} ({score})")
        rows.append({
            "date": d_date,
            "event_count": len(day_df),
            "sequence": "  |  ".join(seq_items) if seq_items else "No recorded rows",
        })
    return pd.DataFrame(rows)

def build_timeline_html(events_df):
    if events_df.empty:
        return "<div style='padding:8px;'>No events available for timeline.</div>"

    work = events_df.copy().sort_values(["time_utc", "hazard_type", "risk_score"], ascending=[True, True, False])
    work["time_str"] = pd.to_datetime(work["time_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")

    items = []
    for _, r in work.iterrows():
        color = event_color(r.get("risk_level", "Info"))
        mag_text = ""
        if pd.notna(r.get("magnitude", None)):
            mag_text = f" | M {r['magnitude']}"
        elif r["hazard_type"] == "Typhoon":
            mag_text = " | Cyclone bulletin"
        band = r.get("influence_band", "")
        items.append(f"""
        <div style="min-width:320px; max-width:320px; background:white; border-left:8px solid {color};
                    border-radius:10px; padding:12px; margin-right:12px; box-shadow:0 1px 4px rgba(0,0,0,0.12);">
            <div style="font-size:12px; color:#555;">{r['time_str']}</div>
            <div style="font-weight:700; font-size:18px; margin-top:4px;">{r['hazard_type']}</div>
            <div style="font-weight:600; margin-top:4px;">{r['name']}</div>
            <div style="font-size:13px; color:#333; margin-top:6px;">
                Score: {r.get('risk_score', '')} | {r.get('risk_level', '')}{mag_text}
            </div>
            <div style="font-size:13px; color:#444; margin-top:6px;">
                Dist. to PH: {r.get('distance_to_ph_km', '')} km | {band}
            </div>
            <div style="font-size:12px; color:#666; margin-top:8px;">
                {str(r.get('details', ''))[:140]}
            </div>
        </div>
        """)

    html = f"""
    <div style="overflow-x:auto; white-space:nowrap; padding:6px 0 10px 0; border:1px solid #e5e7eb; border-radius:12px; background:#f8fafc;">
        <div style="display:flex; flex-direction:row; padding:10px; align-items:stretch;">
            {''.join(items)}
        </div>
    </div>
    """
    return html

# ----------------------------
# DATA LOADERS
# ----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_earthquakes(min_mag=5.0, hours_back=7 * 24, bounds=None):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
        "orderby": "time",
        "limit": 300,
    }
    resp = safe_get(USGS_QUERY_URL, params=params)
    if isinstance(resp, Exception):
        return pd.DataFrame(), str(resp)
    data = resp.json()
    rows = []
    bounds = bounds or DEFAULT_BOUNDS
    for f in data.get("features", []):
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        if len(coords) < 3:
            continue
        lon, lat, depth = coords[0], coords[1], coords[2]
        if lat is None or lon is None:
            continue
        if not in_extent(lat, lon, bounds["min_lat"], bounds["max_lat"], bounds["min_lon"], bounds["max_lon"]):
            continue
        props = f.get("properties", {})
        dist = haversine_km(lat, lon, PH_CENTER["lat"], PH_CENTER["lon"])
        mag = props.get("mag") or 0.0
        score = score_eq(mag, depth or 0, dist)
        rows.append({
            "hazard_type": "Earthquake",
            "name": props.get("title", "Earthquake"),
            "source": "USGS",
            "time_utc": datetime.fromtimestamp(props.get("time", 0) / 1000, tz=timezone.utc),
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
    df = pd.DataFrame(rows).sort_values("time_utc", ascending=False) if rows else pd.DataFrame()
    return df, None

@st.cache_data(ttl=1800, show_spinner=False)
def load_typhoon_status():
    resp = safe_get(PAGASA_BULLETIN_URL)
    if isinstance(resp, Exception):
        return pd.DataFrame(), f"Could not read PAGASA bulletin page: {resp}"
    html = resp.text
    title = parse_html_title(html)
    no_active = "No Active Tropical Cyclone within the Philippine Area of Responsibility" in html
    rows = []
    if no_active:
        rows.append({
            "hazard_type": "Typhoon",
            "name": "No active tropical cyclone in PAR",
            "source": "PAGASA",
            "time_utc": datetime.now(timezone.utc),
            "lat": PH_CENTER["lat"],
            "lon": PH_CENTER["lon"],
            "distance_to_ph_km": 0.0,
            "influence_band": "Within PH context",
            "risk_score": 0.0,
            "risk_level": "Info",
            "details": "PAGASA bulletin page currently reports no active tropical cyclone within the Philippine Area of Responsibility.",
            "url": PAGASA_BULLETIN_URL,
        })
    else:
        base_time = datetime.now(timezone.utc)
        rows.append({
            "hazard_type": "Typhoon",
            "name": title or "Active PAGASA tropical cyclone bulletin",
            "source": "PAGASA",
            "time_utc": base_time,
            "lat": PH_CENTER["lat"],
            "lon": PH_CENTER["lon"],
            "distance_to_ph_km": 0.0,
            "influence_band": "Within PH context",
            "risk_score": score_typhoon(title or "typhoon", 0.0),
            "risk_level": classify(score_typhoon(title or "typhoon", 0.0)),
            "details": "A tropical cyclone bulletin page is active. Open the source link to inspect the latest official advisory and warning area.",
            "url": PAGASA_BULLETIN_URL,
        })
    return pd.DataFrame(rows), None

@st.cache_data(ttl=3600, show_spinner=False)
def load_volcano_watch(lookback_days=14):
    rows = []
    now = datetime.now(timezone.utc)
    for i, v in enumerate(VOLCANOES):
        dist = haversine_km(v["lat"], v["lon"], PH_CENTER["lat"], PH_CENTER["lon"])
        base = 18 if v["priority"] == "PH" else 8
        dist_factor = max(0.2, 1.0 - min(dist, 6500) / 7000.0)
        score = round(base * dist_factor, 1)
        # Synthetic timeline anchor so the 14-day timeline has sequence placeholders for volcano watch entries.
        event_time = now - timedelta(days=(i % min(lookback_days, 14)))
        rows.append({
            "hazard_type": "Volcano",
            "name": v["name"],
            "source": "PHIVOLCS / Smithsonian GVP",
            "time_utc": event_time,
            "lat": v["lat"],
            "lon": v["lon"],
            "distance_to_ph_km": round(dist, 0),
            "influence_band": influence_band(dist),
            "risk_score": score,
            "risk_level": classify(score),
            "details": f"Regional watchlist volcano in {v['country']}. Replace this placeholder timing with actual eruption/advisory timestamps in Phase 2.",
            "url": PHIVOLCS_HOME if v["country"] == "Philippines" else GVP_CURRENT,
        })
    return pd.DataFrame(rows), None

def combined_map_df(*dfs):
    frames = []
    for df in dfs:
        if df is not None and not df.empty:
            frames.append(df.copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["radius"] = out["risk_score"].fillna(8).clip(lower=5) * 12000
    return out

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Controls")
timeline_days = st.sidebar.slider("Timeline sequence window (days)", 3, 14, 14)
lookback_days = st.sidebar.slider("Earthquake lookback (days)", 1, 14, 7)
min_mag = st.sidebar.slider("Minimum earthquake magnitude", 5.0, 8.0, 5.0, 0.1)

st.sidebar.subheader("Regional extent")
min_lat = st.sidebar.number_input("Min latitude", value=DEFAULT_BOUNDS["min_lat"])
max_lat = st.sidebar.number_input("Max latitude", value=DEFAULT_BOUNDS["max_lat"])
min_lon = st.sidebar.number_input("Min longitude", value=DEFAULT_BOUNDS["min_lon"])
max_lon = st.sidebar.number_input("Max longitude", value=DEFAULT_BOUNDS["max_lon"])
bounds = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}

st.sidebar.subheader("Layers")
show_eq = st.sidebar.checkbox("Earthquakes M≥threshold", value=True)
show_typhoon = st.sidebar.checkbox("Typhoon / cyclone status", value=True)
show_volcano = st.sidebar.checkbox("Volcano watchlist", value=True)

st.title("Philippines Multi-Hazard Regional Watch")
st.caption("Typhoon, volcano, and M≥5 earthquake watch for the Philippines and the broader regional neighborhood")

eq_df, eq_err = load_earthquakes(min_mag=min_mag, hours_back=lookback_days * 24, bounds=bounds)
ty_df, ty_err = load_typhoon_status()
vol_df, vol_err = load_volcano_watch(lookback_days=timeline_days)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Earthquakes in window", len(eq_df) if show_eq else 0)
with col2:
    active_ty = 0 if ty_df.empty or (ty_df["risk_score"].fillna(0) == 0).all() else len(ty_df)
    st.metric("Typhoon alert rows", active_ty if show_typhoon else 0)
with col3:
    st.metric("Volcanoes on watchlist", len(vol_df) if show_volcano else 0)

summary_rows = []
if show_eq and not eq_df.empty:
    summary_rows.append(eq_df.nlargest(3, "risk_score")[["hazard_type", "name", "risk_level", "risk_score", "distance_to_ph_km"]])
if show_typhoon and not ty_df.empty:
    summary_rows.append(ty_df[["hazard_type", "name", "risk_level", "risk_score", "distance_to_ph_km"]])
if show_volcano and not vol_df.empty:
    summary_rows.append(vol_df.nlargest(5, "risk_score")[["hazard_type", "name", "risk_level", "risk_score", "distance_to_ph_km"]])

if summary_rows:
    top_df = pd.concat(summary_rows, ignore_index=True).sort_values("risk_score", ascending=False).head(10)
    top_level = top_df.iloc[0]["risk_level"]
    st.info(indicator(f"Top current regional concern: {top_df.iloc[0]['name']} ({top_df.iloc[0]['hazard_type']})", top_level))
else:
    st.success("No records matched the current filters.")

left, right = st.columns([2, 1])

with left:
    st.subheader("Regional hazard map")
    map_df = combined_map_df(
        eq_df if show_eq else pd.DataFrame(),
        ty_df if show_typhoon else pd.DataFrame(),
        vol_df if show_volcano else pd.DataFrame(),
    )
    if map_df.empty:
        st.warning("No map features available.")
    else:
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_radius="radius",
            radius_min_pixels=5,
            radius_max_pixels=40,
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
        view_state = pdk.ViewState(latitude=8, longitude=143, zoom=2.0)
        tooltip = {
            "html": "<b>{hazard_type}</b><br/>{name}<br/>Score: {risk_score}<br/>Distance to PH: {distance_to_ph_km} km<br/>{details}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        st.pydeck_chart(pdk.Deck(layers=[scatter], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)

with right:
    st.subheader("Top ranked events")
    if summary_rows:
        st.dataframe(top_df, use_container_width=True, hide_index=True)
    else:
        st.write("No ranked rows available.")

# ----------------------------
# TIMELINE
# ----------------------------
timeline_frames = []
if show_eq and not eq_df.empty:
    timeline_frames.append(eq_df.copy())
if show_typhoon and not ty_df.empty:
    timeline_frames.append(ty_df.copy())
if show_volcano and not vol_df.empty:
    timeline_frames.append(vol_df.copy())

timeline_df = pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame()
if not timeline_df.empty:
    timeline_df = timeline_df.sort_values(["time_utc", "hazard_type", "risk_score"], ascending=[True, True, False])

st.subheader(f"{timeline_days}-day sequence timeline")
st.caption("Use the horizontal scroll area below to quickly compare which event appeared first across volcano, earthquake, and typhoon rows.")

if timeline_df.empty:
    st.warning("No timeline events available.")
else:
    st.components.v1.html(build_timeline_html(timeline_df), height=260, scrolling=True)

    daily_seq_df = build_daily_sequence(timeline_df, days=timeline_days)
    st.markdown("**Daily sequence ledger**")
    st.dataframe(daily_seq_df, use_container_width=True, hide_index=True)

    selected_day = st.select_slider(
        "Inspect a day in the sequence",
        options=[d.isoformat() for d in daily_seq_df["date"]],
        value=daily_seq_df["date"].iloc[-1].isoformat(),
    )
    picked = timeline_df[pd.to_datetime(timeline_df["time_utc"], utc=True).dt.date.astype(str) == selected_day].copy()
    if picked.empty:
        st.info(f"No event rows on {selected_day}.")
    else:
        picked = picked.sort_values(["time_utc", "hazard_type"], ascending=[True, True])
        st.markdown(f"**Detailed order for {selected_day}**")
        show_cols = ["time_utc", "hazard_type", "name", "risk_level", "risk_score", "distance_to_ph_km", "details"]
        st.dataframe(picked[show_cols], use_container_width=True, hide_index=True)

tab1, tab2, tab3, tab4 = st.tabs(["Earthquakes", "Typhoon", "Volcano", "Sources & notes"])

with tab1:
    st.subheader("Earthquake monitor")
    if eq_err:
        st.error(eq_err)
    elif eq_df.empty:
        st.warning("No earthquakes found for the selected window and magnitude.")
    else:
        show_cols = ["time_utc", "name", "magnitude", "depth_km", "distance_to_ph_km", "influence_band", "risk_level", "risk_score", "url"]
        st.dataframe(eq_df[show_cols], use_container_width=True, hide_index=True)
        st.download_button("Download earthquake CSV", eq_df.to_csv(index=False).encode("utf-8"), "earthquakes_watch.csv", "text/csv")

with tab2:
    st.subheader("Typhoon / tropical cyclone monitor")
    if ty_err:
        st.error(ty_err)
    elif ty_df.empty:
        st.warning("No typhoon rows available.")
    else:
        for _, row in ty_df.iterrows():
            st.markdown(f"**{row['name']}**")
            st.write(indicator(f"Current level: {row['risk_level']}", row["risk_level"]))
            st.write(row["details"])
            st.markdown(f"Source: {row['url']}")

with tab3:
    st.subheader("Volcano regional watchlist")
    if vol_err:
        st.error(vol_err)
    elif vol_df.empty:
        st.warning("No volcano rows available.")
    else:
        show_cols = ["time_utc", "name", "source", "distance_to_ph_km", "influence_band", "risk_level", "risk_score", "url"]
        st.dataframe(vol_df[show_cols], use_container_width=True, hide_index=True)
        st.caption("Current version uses a watchlist with placeholder timestamps for sequence visualization. Replace these with actual bulletin/advisory timestamps in Phase 2.")

with tab4:
    st.subheader("Source links")
    st.markdown(f"- PAGASA tropical cyclone bulletin: {PAGASA_BULLETIN_URL}")
    st.markdown(f"- PHIVOLCS homepage: {PHIVOLCS_HOME}")
    st.markdown(f"- USGS Earthquake Catalog API: {USGS_QUERY_URL}")
    st.markdown(f"- Smithsonian GVP current eruptions: {GVP_CURRENT}")
    st.markdown(f"- Smithsonian / USGS weekly volcanic activity: {GVP_WEEKLY}")
    st.subheader("Notes")
    st.write(
        """
        1. Earthquake rows are live from the USGS event service.
        2. Typhoon rows are bulletin-based and currently summarize the PAGASA active/inactive page state.
        3. Volcano rows currently serve as a regional watchlist; their timestamps are placeholders for timeline visualization until full eruption/advisory parsing is added.
        4. The timeline ledger helps compare sequence order across hazards over up to 14 days.
        """
    )
