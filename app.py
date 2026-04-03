
import math
import re
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

VOLCANO_WATCHLIST = [
    {"name": "Mayon", "country": "Philippines", "lat": 13.257, "lon": 123.685},
    {"name": "Taal", "country": "Philippines", "lat": 14.002, "lon": 120.993},
    {"name": "Kanlaon", "country": "Philippines", "lat": 10.412, "lon": 123.132},
    {"name": "Bulusan", "country": "Philippines", "lat": 12.770, "lon": 124.050},
    {"name": "Pinatubo", "country": "Philippines", "lat": 15.130, "lon": 120.350},
    {"name": "Lewotobi Laki-Laki", "country": "Indonesia", "lat": -8.530, "lon": 122.775},
    {"name": "Merapi", "country": "Indonesia", "lat": -7.540, "lon": 110.446},
    {"name": "Semeru", "country": "Indonesia", "lat": -8.108, "lon": 112.922},
    {"name": "Anak Krakatau", "country": "Indonesia", "lat": -6.102, "lon": 105.423},
    {"name": "Aso", "country": "Japan", "lat": 32.884, "lon": 131.104},
    {"name": "Sakurajima", "country": "Japan", "lat": 31.585, "lon": 130.657},
    {"name": "Whakaari/White Island", "country": "New Zealand", "lat": -37.521, "lon": 177.183},
    {"name": "Ruapehu", "country": "New Zealand", "lat": -39.281, "lon": 175.570},
    {"name": "Changbaishan", "country": "China / DPRK", "lat": 41.980, "lon": 128.080},
]


def safe_get(url, params=None, timeout=20, headers=None):
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; Streamlit MultiHazard/1.0)"}
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers)
        r.raise_for_status()
        return r
    except Exception as e:
        return e


def parse_html_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


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
    txt = text.lower()
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


def format_time(dt):
    return pd.to_datetime(dt, utc=True).strftime("%Y-%m-%d %H:%M UTC")


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

    df = pd.DataFrame(rows)
    return df, None


@st.cache_data(ttl=1800, show_spinner=False)
def load_typhoon_actual():
    resp = safe_get(PAGASA_BULLETIN_URL)
    if isinstance(resp, Exception):
        return pd.DataFrame(), f"Could not read PAGASA bulletin page: {resp}"

    html = resp.text
    title = parse_html_title(html)
    now = datetime.now(timezone.utc)
    rows = []

    if "No Active Tropical Cyclone within the Philippine Area of Responsibility" in html:
        return pd.DataFrame(), None

    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    name_match = re.search(r"(Tropical Depression|Tropical Storm|Typhoon|Super Typhoon)\s+([A-Z][A-Z0-9\-]+)", text, re.IGNORECASE)
    label = title or "Active PAGASA tropical cyclone bulletin"
    if name_match:
        label = f"{name_match.group(1).title()} {name_match.group(2).upper()}"

    score = score_typhoon(label, 0.0)
    rows.append({
        "hazard_type": "Typhoon",
        "name": label,
        "source": "PAGASA",
        "time_utc": now,
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
    })
    return pd.DataFrame(rows), None


def build_watchlist_table():
    rows = []
    for item in VOLCANO_WATCHLIST:
        dist = haversine_km(item["lat"], item["lon"], PH_CENTER["lat"], PH_CENTER["lon"])
        rows.append({
            "volcano": item["name"],
            "country": item["country"],
            "lat": item["lat"],
            "lon": item["lon"],
            "distance_to_ph_km": round(dist, 0),
            "influence_band": influence_band(dist),
        })
    return pd.DataFrame(rows).sort_values(["distance_to_ph_km", "volcano"])


def build_map_df(eq_df, ty_df):
    frames = []
    if not eq_df.empty:
        frames.append(eq_df.copy())
    if not ty_df.empty:
        frames.append(ty_df.copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["radius"] = out["risk_score"].fillna(8).clip(lower=5) * 12000
    return out


def build_timeline_html(events_df):
    if events_df.empty:
        return "<div style='padding:8px;'>No actual events available for the selected window.</div>"

    work = events_df.copy().sort_values(["time_utc", "risk_score"], ascending=[True, False])
    work["time_str"] = pd.to_datetime(work["time_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")

    blocks = []
    for _, r in work.iterrows():
        color = event_color(r.get("risk_level", "Info"))
        mag_txt = f" | M {r['magnitude']}" if pd.notna(r.get("magnitude")) else ""
        blocks.append(f"""
        <div style="min-width:340px; max-width:340px; background:white; border-left:8px solid {color};
                    border-radius:12px; padding:12px; margin-right:12px; box-shadow:0 1px 4px rgba(0,0,0,0.12);">
            <div style="font-size:12px; color:#555;">{r['time_str']}</div>
            <div style="font-weight:700; font-size:18px; margin-top:4px;">{r['hazard_type']}</div>
            <div style="font-weight:600; margin-top:4px;">{r['name']}</div>
            <div style="font-size:13px; color:#333; margin-top:6px;">
                Score: {r.get('risk_score', '')} | {r.get('risk_level', '')}{mag_txt}
            </div>
            <div style="font-size:13px; color:#444; margin-top:6px;">
                Distance to PH: {r.get('distance_to_ph_km', '')} km | {r.get('influence_band', '')}
            </div>
            <div style="font-size:12px; color:#666; margin-top:8px;">
                {str(r.get('details', ''))[:160]}
            </div>
        </div>
        """)

    return f"""
    <div style="overflow-x:auto; white-space:nowrap; padding:6px 0 10px 0; border:1px solid #e5e7eb; border-radius:12px; background:#f8fafc;">
        <div style="display:flex; flex-direction:row; padding:10px; align-items:stretch;">
            {''.join(blocks)}
        </div>
    </div>
    """


def build_progressive_daily_ledger(events_df, days=14):
    now = datetime.now(timezone.utc)
    dates = [now.date() - timedelta(days=i) for i in range(days - 1, -1, -1)]
    if events_df.empty:
        return pd.DataFrame([{"date": d.isoformat(), "event_count": 0, "sequence": "No actual events"} for d in dates])

    work = events_df.copy()
    work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date

    rows = []
    for d in dates:
        day_df = work[work["event_date"] == d].sort_values(["time_utc", "risk_score"], ascending=[True, False])
        seq_parts = []
        for _, r in day_df.iterrows():
            mag_txt = f", M{r['magnitude']}" if pd.notna(r.get("magnitude")) else ""
            seq_parts.append(f"{r['hazard_type']}: {r['name']}{mag_txt}")
        rows.append({
            "date": d.isoformat(),
            "event_count": len(day_df),
            "sequence": "  |  ".join(seq_parts) if seq_parts else "No actual events",
        })
    return pd.DataFrame(rows)


def filter_window(events_df, end_day_offset, span_days):
    if events_df.empty:
        return events_df.copy()

    today_utc = datetime.now(timezone.utc).date()
    window_end = today_utc - timedelta(days=end_day_offset)
    window_start = window_end - timedelta(days=span_days - 1)

    work = events_df.copy()
    work["event_date"] = pd.to_datetime(work["time_utc"], utc=True).dt.date
    out = work[(work["event_date"] >= window_start) & (work["event_date"] <= window_end)].copy()
    out = out.sort_values(["time_utc", "risk_score"], ascending=[True, False])
    return out, window_start, window_end


st.sidebar.title("Controls")
timeline_days = st.sidebar.slider("Timeline progressive window (days)", 1, 14, 14)
end_day_offset = st.sidebar.slider("Timeline ending day offset (0=today, 1=yesterday, 2=two days ago)", 0, 13, 0)
min_mag = st.sidebar.slider("Minimum earthquake magnitude", 5.0, 8.0, 5.0, 0.1)

st.sidebar.subheader("Regional extent")
min_lat = st.sidebar.number_input("Min latitude", value=DEFAULT_BOUNDS["min_lat"])
max_lat = st.sidebar.number_input("Max latitude", value=DEFAULT_BOUNDS["max_lat"])
min_lon = st.sidebar.number_input("Min longitude", value=DEFAULT_BOUNDS["min_lon"])
max_lon = st.sidebar.number_input("Max longitude", value=DEFAULT_BOUNDS["max_lon"])
bounds = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}

show_eq = st.sidebar.checkbox("Show actual earthquakes", value=True)
show_typhoon = st.sidebar.checkbox("Show actual typhoon bulletin event", value=True)

st.title("Philippines Multi-Hazard Regional Watch")
st.caption("Actual-event focused dashboard for earthquakes and active typhoon bulletins, with volcano watchlist kept separate as reference")

eq_df, eq_err = load_earthquakes(min_mag=min_mag, days_back=14, bounds=bounds)
ty_df, ty_err = load_typhoon_actual()
watchlist_df = build_watchlist_table()

actual_frames = []
if show_eq and not eq_df.empty:
    actual_frames.append(eq_df.copy())
if show_typhoon and not ty_df.empty:
    actual_frames.append(ty_df.copy())
actual_df = pd.concat(actual_frames, ignore_index=True) if actual_frames else pd.DataFrame()

timeline_df, window_start, window_end = filter_window(actual_df, end_day_offset=end_day_offset, span_days=timeline_days)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Actual EQ rows in 14-day source window", len(eq_df) if show_eq else 0)
with col2:
    st.metric("Actual active typhoon rows", len(ty_df) if show_typhoon else 0)
with col3:
    st.metric("Volcano watchlist rows", len(watchlist_df))

if not actual_df.empty:
    top = actual_df.sort_values("risk_score", ascending=False).iloc[0]
    st.info(indicator(f"Top current regional concern: {top['name']} ({top['hazard_type']})", top["risk_level"]))
else:
    st.success("No actual hazard events matched the current settings.")

left, right = st.columns([2, 1])

with left:
    st.subheader("Regional hazard map")
    st.caption("Map uses actual events only. Volcano watchlist is excluded from the dynamic map.")
    map_df = build_map_df(eq_df if show_eq else pd.DataFrame(), ty_df if show_typhoon else pd.DataFrame())

    if map_df.empty:
        st.warning("No actual events available for the map.")
    else:
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_radius="radius",
            radius_min_pixels=5,
            radius_max_pixels=45,
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
            "html": "<b>{hazard_type}</b><br/>{name}<br/>Time: {time_utc}<br/>Score: {risk_score}<br/>Distance to PH: {distance_to_ph_km} km<br/>{details}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        view_state = pdk.ViewState(latitude=8, longitude=143, zoom=2.0)
        st.pydeck_chart(pdk.Deck(layers=[scatter], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)

with right:
    st.subheader("Top actual events")
    if actual_df.empty:
        st.write("No actual events available.")
    else:
        show_top = actual_df.sort_values(["risk_score", "time_utc"], ascending=[False, False]).head(10)
        st.dataframe(show_top[["time_utc", "hazard_type", "name", "risk_level", "risk_score", "distance_to_ph_km"]], use_container_width=True, hide_index=True)

st.subheader("Progressive timeline of actual events")
st.caption(f"Showing actual events from {window_start.isoformat()} to {window_end.isoformat()}. Move the ending-day offset to inspect what led up to a major event in earlier days.")

if timeline_df.empty:
    st.warning("No actual events available in the selected progressive window.")
else:
    st.components.v1.html(build_timeline_html(timeline_df), height=265, scrolling=True)

ledger_df = build_progressive_daily_ledger(timeline_df, days=timeline_days)
st.markdown("**Daily progressive sequence ledger**")
st.dataframe(ledger_df, use_container_width=True, hide_index=True)

selected_day = st.select_slider("Inspect one day in the sequence", options=ledger_df["date"].tolist(), value=ledger_df["date"].iloc[-1])

if timeline_df.empty:
    picked = pd.DataFrame()
else:
    tmp = timeline_df.copy()
    tmp["event_date"] = pd.to_datetime(tmp["time_utc"], utc=True).dt.date.astype(str)
    picked = tmp[tmp["event_date"] == selected_day].sort_values(["time_utc", "risk_score"], ascending=[True, False])

st.markdown(f"**Detailed order for {selected_day}**")
if picked.empty:
    st.info("No actual events on this date.")
else:
    st.dataframe(
        picked[["time_utc", "hazard_type", "name", "magnitude", "depth_km", "distance_to_ph_km", "risk_level", "risk_score", "details"]],
        use_container_width=True,
        hide_index=True
    )

st.markdown("**Lead-up view to the largest event in the selected window**")
if timeline_df.empty:
    st.info("No lead-up analysis available.")
else:
    major = timeline_df.sort_values(["risk_score", "time_utc"], ascending=[False, False]).iloc[0]
    major_time = pd.to_datetime(major["time_utc"], utc=True)
    leadup = timeline_df[pd.to_datetime(timeline_df["time_utc"], utc=True) <= major_time].copy()
    leadup = leadup.sort_values(["time_utc", "risk_score"], ascending=[True, False])
    st.write(f"Major event selected: **{major['name']}** at **{format_time(major_time)}**")
    st.dataframe(
        leadup[["time_utc", "hazard_type", "name", "magnitude", "distance_to_ph_km", "risk_level", "risk_score", "details"]],
        use_container_width=True,
        hide_index=True
    )

tab1, tab2, tab3, tab4 = st.tabs(["Earthquakes", "Typhoon", "Volcano watchlist", "Sources & notes"])

with tab1:
    st.subheader("Actual earthquake records")
    if eq_err:
        st.error(eq_err)
    elif eq_df.empty:
        st.warning("No actual earthquake records for the last 14 days with current filters.")
    else:
        st.dataframe(
            eq_df.sort_values("time_utc", ascending=False)[["time_utc", "name", "magnitude", "depth_km", "distance_to_ph_km", "influence_band", "risk_level", "risk_score", "url"]],
            use_container_width=True,
            hide_index=True
        )
        st.download_button("Download actual earthquake CSV", eq_df.to_csv(index=False).encode("utf-8"), "actual_earthquakes.csv", "text/csv")

with tab2:
    st.subheader("Actual typhoon event rows")
    if ty_err:
        st.error(ty_err)
    elif ty_df.empty:
        st.info("No active PAGASA tropical cyclone bulletin detected right now.")
    else:
        st.dataframe(
            ty_df[["time_utc", "name", "risk_level", "risk_score", "details", "url"]],
            use_container_width=True,
            hide_index=True
        )

with tab3:
    st.subheader("Volcano watchlist reference table")
    st.caption("Reference only. This table is not plotted on the dynamic map and not injected into the actual timeline.")
    st.dataframe(watchlist_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Sources")
    st.markdown(f"- PAGASA tropical cyclone bulletin: {PAGASA_BULLETIN_URL}")
    st.markdown(f"- PHIVOLCS homepage: {PHIVOLCS_HOME}")
    st.markdown(f"- USGS Earthquake Catalog API: {USGS_QUERY_URL}")
    st.markdown(f"- Smithsonian GVP current eruptions: {GVP_CURRENT}")
    st.markdown(f"- Smithsonian / USGS weekly volcanic activity: {GVP_WEEKLY}")

    st.subheader("Notes")
    st.write(
        """
        1. The map and progressive timeline now use actual event rows only.
        2. The volcano list is preserved as a reference table only and is excluded from both the map and the actual-event timeline.
        3. The progressive window can end today, yesterday, two days ago, and so on, allowing you to examine what happened before a major event.
        4. Earthquake rows are fully live from the USGS event service.
        5. The typhoon row appears only when an active PAGASA bulletin is detected.
        6. This version still does not auto-parse live volcano eruption bulletins into actual event rows. That can be added in a later phase.
        """
    )
