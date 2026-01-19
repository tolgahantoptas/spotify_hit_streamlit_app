import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "final_model.joblib")
META_PATH = os.path.join(ART_DIR, "meta.json")
GENRE_MAP_PATH = os.path.join(ART_DIR, "genre_freq_map.json")
LOGO_PATH = os.path.join("assets", "spotify_icon.png")

page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🎵"
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide", page_icon=page_icon)

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

COMMON_CSS = """
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 2.5rem;}
h1 {margin-bottom: 0.2rem;}
.metric-hint {font-size: 0.95rem; margin-top: -0.25rem;}
[data-baseweb="slider"] {padding-top: 0.35rem;}
@media (max-width: 768px){
  .block-container {padding-top: 3.2rem;}
  [data-testid="stImage"] img {margin-top: 10px;}
}
</style>
"""

DARK_CSS = """
<style>
:root{
 --bg:#0e1117;
 --txt:#ffffff;
 --muted:rgba(255,255,255,0.86);
 --control-bg:#141a24;
 --control-border:rgba(255,255,255,0.30);
 --menu-hover:rgba(0,0,0,0.06);
 --btn-bg:#1b2330;
 --btn-hover:#263244;
 --exp-bg:#141a24;
}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
h1,h2,h3,h4,h5,h6,p,span,label,small{color:#ffffff!important;opacity:1!important;}
.metric-hint{color:var(--muted)!important;}

div[data-baseweb="select"]>div{
  background:var(--control-bg)!important;
  border-color:var(--control-border)!important;
}
div[data-baseweb="select"] *{
  color:#ffffff!important;
  fill:#ffffff!important;
  opacity:1!important;
  -webkit-text-fill-color:#ffffff!important;
}
div[data-baseweb="select"] input{opacity:1!important;}

div[role="listbox"], ul[role="listbox"], [data-baseweb="popover"], [data-baseweb="menu"]{
  background:#ffffff!important;
  border:1px solid rgba(0,0,0,0.12)!important;
}
div[role="option"], li[role="option"],
div[role="option"] *, li[role="option"] *,
[data-baseweb="popover"] *, [data-baseweb="menu"] *{
  color:#000000!important;
  opacity:1!important;
  background:#ffffff!important;
}
div[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"],
div[role="option"]:hover, li[role="option"]:hover{
  background:var(--menu-hover)!important;
}

[data-baseweb="accordion"] > div{
  background:var(--exp-bg)!important;
  border-radius: 14px!important;
  border: 1px solid rgba(255,255,255,0.18)!important;
}
[data-baseweb="accordion"] button,
[data-baseweb="accordion"] span{
  color:#ffffff!important;
  opacity:1!important;
  font-weight:650!important;
}

.stButton>button{
  background:var(--btn-bg)!important;
  color:#ffffff!important;
  border:1px solid rgba(255,255,255,0.18)!important;
  border-radius:12px!important;
  font-weight:700!important;
  padding: 0.62rem 1.15rem!important;
}
.stButton>button *{color:#ffffff!important; opacity:1!important;}
.stButton>button:hover{background:var(--btn-hover)!important;}
.stButton>button:disabled, .stButton>button[disabled]{
  background:rgba(27,35,48,0.65)!important;
  color:rgba(255,255,255,0.75)!important;
}
.stButton>button:disabled *,.stButton>button[disabled] *{
  color:rgba(255,255,255,0.75)!important;
  opacity:1!important;
}

[data-baseweb="toggle"] span{color:#ffffff!important; font-weight:800!important; opacity:1!important;}
[data-baseweb="toggle"] [role="switch"]{
  background:#334155!important;
  border:1px solid rgba(255,255,255,0.22)!important;
}
[data-baseweb="toggle"] [role="switch"][aria-checked="true"]{background:#1db954!important;}
[data-baseweb="toggle"] [data-baseweb="thumb"]{background:#ffffff!important;}

[data-testid="stTooltipContent"]{
  background:#111827!important;
  color:#ffffff!important;
  border:1px solid rgba(255,255,255,0.20)!important;
}
</style>
"""

LIGHT_CSS = """
<style>
:root{
 --bg:#ffffff;
 --txt:#000000;
 --muted:rgba(0,0,0,0.82);
 --control-bg:#ffffff;
 --control-border:rgba(0,0,0,0.32);
 --menu-hover:rgba(0,0,0,0.06);
 --btn-bg:#f3f4f6;
 --btn-hover:#e5e7eb;
 --exp-bg:#ffffff;
}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
h1,h2,h3,h4,h5,h6,p,span,label,small{color:#000000!important;opacity:1!important;}
.metric-hint{color:var(--muted)!important;}

div[data-baseweb="select"]>div{
  background:var(--control-bg)!important;
  border-color:var(--control-border)!important;
}
div[data-baseweb="select"] *{
  color:#000000!important;
  fill:#000000!important;
  opacity:1!important;
  -webkit-text-fill-color:#000000!important;
}
div[data-baseweb="select"] input{opacity:1!important;}

div[role="listbox"], ul[role="listbox"], [data-baseweb="popover"], [data-baseweb="menu"]{
  background:#ffffff!important;
  border:1px solid var(--control-border)!important;
}
div[role="option"], li[role="option"],
div[role="option"] *, li[role="option"] *,
[data-baseweb="popover"] *, [data-baseweb="menu"] *{
  color:#000000!important;
  opacity:1!important;
  background:#ffffff!important;
}
div[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"],
div[role="option"]:hover, li[role="option"]:hover{
  background:var(--menu-hover)!important;
}

[data-baseweb="accordion"] > div{
  background:var(--exp-bg)!important;
  border-radius: 14px!important;
  border: 1px solid rgba(0,0,0,0.14)!important;
}
[data-baseweb="accordion"] button,
[data-baseweb="accordion"] span{
  color:#000000!important;
  opacity:1!important;
  font-weight:650!important;
}

.stButton>button{
  background:var(--btn-bg)!important;
  color:#000000!important;
  border:1px solid rgba(0,0,0,0.16)!important;
  border-radius:12px!important;
  font-weight:700!important;
  padding: 0.62rem 1.15rem!important;
}
.stButton>button *{color:#000000!important; opacity:1!important;}
.stButton>button:hover{background:var(--btn-hover)!important;}
.stButton>button:disabled, .stButton>button[disabled]{
  background:rgba(243,244,246,0.75)!important;
  color:rgba(0,0,0,0.70)!important;
}
.stButton>button:disabled *,.stButton>button[disabled] *{
  color:rgba(0,0,0,0.70)!important;
  opacity:1!important;
}

[data-baseweb="toggle"] span{color:#000000!important; font-weight:850!important; opacity:1!important;}
[data-baseweb="toggle"] [role="switch"]{
  background:#e5e7eb!important;
  border:1px solid rgba(0,0,0,0.18)!important;
}
[data-baseweb="toggle"] [role="switch"][aria-checked="true"]{background:#1db954!important;}
[data-baseweb="toggle"] [data-baseweb="thumb"]{background:#111111!important;}

[data-testid="stTooltipContent"]{
  background:#ffffff!important;
  color:#000000!important;
  border:1px solid rgba(0,0,0,0.18)!important;
}
</style>
"""

st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.markdown(DARK_CSS if st.session_state["theme_mode"] == "dark" else LIGHT_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_freq = json.load(f)
    genre_freq = {str(k).strip().lower(): float(v) for k, v in genre_freq.items()}
    return model, meta, genre_freq

model, meta, GENRE_FREQ_MAP = load_artifacts()
TH = float(meta.get("threshold", 0.67))
FEATURES = meta.get("feature_columns", [])

if not FEATURES:
    st.error("feature_columns not found in artifacts/meta.json")
    st.stop()

if "track_genre_freq" not in FEATURES:
    st.error("Model does not include track_genre_freq.")
    st.stop()

if "reset_nonce" not in st.session_state:
    st.session_state["reset_nonce"] = 0

def do_reset():
    keep = {"reset_nonce", "theme_mode"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state["reset_nonce"] += 1

top_l, top_r = st.columns([0.72, 0.28], vertical_alignment="center")

with top_r:
    is_dark = st.toggle(
        "🌙 Dark mode",
        value=(st.session_state["theme_mode"] == "dark"),
        key=f"theme_toggle_{st.session_state['reset_nonce']}",
        help="Switch the interface between Dark and Light. Dark is easier on the eyes at night; Light is better in bright environments."
    )
    new_mode = "dark" if is_dark else "light"
    if new_mode != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = new_mode
        st.rerun()

with top_l:
    c1, c2 = st.columns([0.10, 0.90], vertical_alignment="center")
    with c1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=72)
    with c2:
        st.title("Spotify Hit Predictor")
        st.caption("Select a genre, adjust track/artist features, then press Predict. Hover over the ? icons to understand each input.")

st.button("🔄 Reset", on_click=do_reset, key="reset_btn", help="Clears all inputs and returns everything to the default values.")

super_map = {
    "rock": "Rock/Metal", "alt-rock": "Rock/Metal", "alternative": "Rock/Metal",
    "hard-rock": "Rock/Metal", "punk": "Rock/Metal", "punk-rock": "Rock/Metal",
    "metal": "Rock/Metal", "black-metal": "Rock/Metal", "death-metal": "Rock/Metal",
    "metalcore": "Rock/Metal", "grunge": "Rock/Metal", "industrial": "Rock/Metal",
    "rock-n-roll": "Rock/Metal", "rockabilly": "Rock/Metal", "hardcore": "Rock/Metal",
    "psych-rock": "Rock/Metal", "emo": "Rock/Metal", "garage": "Rock/Metal",
    "pop": "Pop", "indie-pop": "Pop", "synth-pop": "Pop", "pop-film": "Pop",
    "k-pop": "Pop", "j-pop": "Pop", "mandopop": "Pop", "cantopop": "Pop",
    "british": "Pop",
    "hip-hop": "Hip-Hop/R&B", "rap": "Hip-Hop/R&B", "r-n-b": "Hip-Hop/R&B",
    "soul": "Hip-Hop/R&B", "funk": "Hip-Hop/R&B",
    "edm": "Electronic/Dance", "electronic": "Electronic/Dance", "electro": "Electronic/Dance",
    "house": "Electronic/Dance", "deep-house": "Electronic/Dance",
    "techno": "Electronic/Dance", "detroit-techno": "Electronic/Dance", "chicago-house": "Electronic/Dance",
    "drum-and-bass": "Electronic/Dance", "dubstep": "Electronic/Dance",
    "dance": "Electronic/Dance", "club": "Electronic/Dance", "disco": "Electronic/Dance",
    "acoustic": "Acoustic/Folk/Country", "folk": "Acoustic/Folk/Country",
    "country": "Acoustic/Folk/Country", "bluegrass": "Acoustic/Folk/Country",
    "singer-songwriter": "Acoustic/Folk/Country", "songwriter": "Acoustic/Folk/Country",
    "latin": "Latin/Reggae", "latino": "Latin/Reggae", "reggaeton": "Latin/Reggae",
    "reggae": "Latin/Reggae", "dancehall": "Latin/Reggae", "brazil": "Latin/Reggae",
    "classical": "Classical/Jazz", "piano": "Classical/Jazz", "jazz": "Classical/Jazz",
    "ambient": "Classical/Jazz", "new-age": "Classical/Jazz",
    "anime": "Other", "disney": "Other", "children": "Other", "comedy": "Other",
}

genres = sorted(GENRE_FREQ_MAP.keys())
genre_groups = sorted(set(super_map.get(g, "Other") for g in genres))

st.header("🎛️ Genre Selection")
g1, g2 = st.columns(2)

with g1:
    default_group = "Pop" if "Pop" in genre_groups else genre_groups[0]
    chosen_super = st.selectbox(
        "🎧 Genre",
        genre_groups,
        index=genre_groups.index(default_group),
        key=f"genre_group_{st.session_state['reset_nonce']}",
        help="Step 1/2: Choose a broad genre family. This is only a filter to help you find the right style faster."
    )

with g2:
    sub_list = [g for g in genres if super_map.get(g, "Other") == chosen_super]
    sub_list = sub_list if sub_list else genres
    chosen_genre = st.selectbox(
        "🏷️ Alt genre",
        sub_list,
        index=0,
        key=f"alt_genre_{st.session_state['reset_nonce']}",
        help="Step 2/2: Choose the specific style inside the selected family."
    )

track_genre_freq = float(GENRE_FREQ_MAP.get(chosen_genre, 0.0))

st.header("📌 Basic Features")
c1, c2 = st.columns(2)

with c1:
    duration_sec = st.slider(
        "⏱️ duration (sec)", 30, 900, 180,
        key=f"duration_{st.session_state['reset_nonce']}",
        help="Total track length in seconds. Example: 180 sec equals 3 minutes."
    )
    st.markdown(f"<div class='metric-hint'>= {duration_sec//60} min {duration_sec%60:02d} sec</div>", unsafe_allow_html=True)

    artist_followers_k = st.slider(
        "👥 artist_followers (K)", 0, 150_000, 100, step=100,
        key=f"followers_{st.session_state['reset_nonce']}",
        help="Artist follower count in thousands (K). Example: 250K equals 250,000 followers."
    )
    st.markdown(f"<div class='metric-hint'>= {artist_followers_k*1000:,} followers</div>", unsafe_allow_html=True)

    danceability = st.slider(
        "💃 danceability", 0.0, 1.0, 0.50,
        key=f"dance_{st.session_state['reset_nonce']}",
        help="Dance-friendliness (0–1). Higher means a clearer rhythm and easier groove to dance to."
    )

    energy = st.slider(
        "🔋 energy", 0.0, 1.0, 0.50,
        key=f"energy_{st.session_state['reset_nonce']}",
        help="Intensity (0–1). Higher feels more powerful, loud, and fast."
    )

    loudness = st.slider(
        "🔊 loudness (dB)", -20.0, 0.0, -8.0,
        key=f"loud_{st.session_state['reset_nonce']}",
        help="Overall loudness. Closer to 0 means louder mastering. Many tracks are around -14 to -6 dB."
    )

with c2:
    tempo = st.slider(
        "🥁 tempo (BPM)", 40.0, 220.0, 120.0,
        key=f"tempo_{st.session_state['reset_nonce']}",
        help="Beat speed in BPM (beats per minute). 120 BPM is common for pop/dance."
    )

    artist_popularity = st.slider(
        "⭐ artist_popularity", 0, 100, 50,
        key=f"apop_{st.session_state['reset_nonce']}",
        help="Spotify popularity score (0–100). Higher means the artist is currently listened to more."
    )

    valence = st.slider(
        "😊 valence", 0.0, 1.0, 0.50,
        key=f"val_{st.session_state['reset_nonce']}",
        help="Musical positivity (0–1). Higher = happier/bright; lower = sadder/darker."
    )

    release_year = st.slider(
        "📅 release_year", 1950, 2025, 2020,
        key=f"year_{st.session_state['reset_nonce']}",
        help="Release year of the track."
    )

with st.expander("🧪 Advanced (optional)", expanded=False):
    speechiness = st.slider(
        "🗣️ speechiness", 0.0, 1.0, 0.05,
        key=f"sp_{st.session_state['reset_nonce']}",
        help="Spoken-word content (0–1). Higher values often happen in rap or talk-like vocals."
    )

    acousticness = st.slider(
        "🎻 acousticness", 0.0, 1.0, 0.20,
        key=f"ac_{st.session_state['reset_nonce']}",
        help="Acoustic feel (0–1). Higher values usually mean more natural/unplugged instruments."
    )

    instrumentalness = st.slider(
        "🎼 instrumentalness", 0.0, 1.0, 0.00,
        key=f"ins_{st.session_state['reset_nonce']}",
        help="Likelihood of no vocals (0–1). Higher means more likely instrumental."
    )

    liveness = st.slider(
        "🎤 liveness", 0.0, 1.0, 0.15,
        key=f"liv_{st.session_state['reset_nonce']}",
        help="Live recording likelihood (0–1). Higher can indicate an audience or live setting."
    )

row = {
    "duration_ms": float(duration_sec) * 1000.0,
    "danceability": float(danceability),
    "energy": float(energy),
    "loudness": float(loudness),
    "speechiness": float(speechiness),
    "acousticness": float(acousticness),
    "instrumentalness": float(instrumentalness),
    "liveness": float(liveness),
    "valence": float(valence),
    "tempo": float(tempo),
    "release_year": float(release_year),
    "artist_followers": float(artist_followers_k) * 1000.0,
    "artist_popularity": float(artist_popularity),
    "track_genre_freq": float(track_genre_freq),
}

X = pd.DataFrame([row])
for c in FEATURES:
    if c not in X.columns:
        X[c] = 0.0
X = X[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

st.divider()

if st.button("🎯 Predict", key=f"pred_{st.session_state['reset_nonce']}", help="Run the prediction using the current inputs."):
    if hasattr(model, "predict_proba"):
        hit_prob = float(model.predict_proba(X)[:, 1][0])
    else:
        hit_prob = float(model.decision_function(X)[0])

    non_hit_prob = float(1.0 - hit_prob)

    if hit_prob >= TH:
        st.success(f"✅ This song would be a HIT! (Hit probability: {hit_prob:.3f})")
    else:
        st.warning(f"❌ This song would NOT be a hit. (Non-hit probability: {non_hit_prob:.3f})")
