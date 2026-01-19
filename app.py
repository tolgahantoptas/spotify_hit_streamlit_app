import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os

# =========================
# Paths / Files
# =========================
ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "final_model.joblib")
META_PATH = os.path.join(ART_DIR, "meta.json")
GENRE_MAP_PATH = os.path.join(ART_DIR, "genre_freq_map.json")

# Put an "icon-only" Spotify logo here (official asset):
# repo/
#   assets/spotify_icon.png
LOGO_PATH = os.path.join("assets", "spotify_icon.png")

# =========================
# Page config (must be early)
# =========================
page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🎵"
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide", page_icon=page_icon)

# =========================
# CSS
# =========================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
      h1 {margin-bottom: 0.2rem;}
      h2 {margin-top: 1.2rem;}
      .stButton>button {border-radius: 10px; padding: 0.55rem 1.1rem;}
      .stExpander {border-radius: 14px;}
      [data-baseweb="slider"] {padding-top: 0.35rem;}
      .small-muted {color: rgba(255,255,255,0.65); font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Cache artifacts
# =========================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing meta: {META_PATH}")
    if not os.path.exists(GENRE_MAP_PATH):
        raise FileNotFoundError(f"Missing genre map: {GENRE_MAP_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_freq_map = json.load(f)

    # normalize keys (safety)
    genre_freq_map = {str(k).strip().lower(): float(v) for k, v in genre_freq_map.items()}
    return model, meta, genre_freq_map

try:
    model, meta, GENRE_FREQ_MAP = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

TH = float(meta.get("threshold", 0.67))
FEATURES = meta.get("feature_columns", []) or meta.get("feature_cols", [])
if not FEATURES:
    st.error("feature_columns not found in artifacts/meta.json")
    st.stop()
if "track_genre_freq" not in FEATURES:
    st.error("Model does not include track_genre_freq.")
    st.stop()

# =========================
# Reset logic
# =========================
if "reset_nonce" not in st.session_state:
    st.session_state["reset_nonce"] = 0

def do_reset():
    keep = {"reset_nonce"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state["reset_nonce"] += 1

# =========================
# Header with logo
# =========================
h1, h2 = st.columns([0.08, 0.92], vertical_alignment="center")
with h1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=72)  # <- bigger logo
    else:
        st.write("🎵")
with h2:
    st.title("Spotify Hit Predictor")
    st.caption("Adjust the inputs and press Predict. Use the ? icons next to each variable for a short explanation.")

# =========================
# Keep-alive (client-side refresh)
# =========================
with st.sidebar:
    st.header("🛠️ App Settings")
    keep_alive = st.checkbox(
        "Keep alive (auto-refresh every 5 min)",
        value=True,
        help=(
            "If you keep the page open, this auto-refresh helps keep your session active. "
            "To keep the app awake with no visitors on Streamlit Cloud, use an external ping (e.g., UptimeRobot every 5 min)."
        ),
        key=f"keepalive_{st.session_state['reset_nonce']}"
    )

    st.markdown(
        """
        <div class="small-muted">
        <b>Streamlit Cloud note:</b> Apps may sleep when inactive.  
        Best fix: set up an external HTTP monitor (UptimeRobot) to ping your app URL every 5 minutes.
        </div>
        """,
        unsafe_allow_html=True
    )

if keep_alive:
    st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

st.button("🔄 Reset", on_click=do_reset, key="reset_btn")

# =========================
# Super mapping (UI only)
# Labels:
#   Genre  -> broad category
#   Alt genre -> specific sub-genre
# =========================
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

# =========================
# Genre selection
# =========================
st.header("🎛️ Genre Selection")
g1, g2 = st.columns(2)

with g1:
    default_group = "Pop" if "Pop" in genre_groups else genre_groups[0]
    chosen_super = st.selectbox(
        "🎧 Genre",
        genre_groups,
        index=genre_groups.index(default_group),
        key=f"genre_group_{st.session_state['reset_nonce']}",
        help="A broad category. This only filters the list of alt genres below."
    )

with g2:
    sub_list = [g for g in genres if super_map.get(g, "Other") == chosen_super]
    sub_list = sub_list if sub_list else genres
    chosen_genre = st.selectbox(
        "🏷️ Alt genre",
        sub_list,
        index=0,
        key=f"alt_genre_{st.session_state['reset_nonce']}",
        help="Specific genre label used by the model mapping."
    )

track_genre_freq = float(GENRE_FREQ_MAP.get(chosen_genre, 0.0))

# =========================
# Basic Features
# =========================
st.header("📌 Basic Features")
c1, c2 = st.columns(2)

with c1:
    duration_sec = st.slider(
        "⏱️ duration (sec)", 30, 900, 180,
        key=f"duration_{st.session_state['reset_nonce']}",
        help="Track length in seconds."
    )
    st.caption(f"Selected: {duration_sec//60}:{duration_sec%60:02d}")

    artist_followers_k = st.slider(
        "👥 artist_followers (K)", 0, 150_000, 100,
        step=100,
        key=f"followers_{st.session_state['reset_nonce']}",
        help="Artist followers in thousands (K)."
    )
    st.caption(f"{artist_followers_k:,}K = {artist_followers_k*1000:,} followers")

    danceability = st.slider(
        "💃 danceability", 0.0, 1.0, 0.50,
        key=f"dance_{st.session_state['reset_nonce']}",
        help="How suitable the track is for dancing (0–1)."
    )

    energy = st.slider(
        "🔋 energy", 0.0, 1.0, 0.50,
        key=f"energy_{st.session_state['reset_nonce']}",
        help="Perceived intensity and activity (0–1)."
    )

    loudness = st.slider(
        "🔊 loudness (dB)", -20.0, 0.0, -8.0,  # <- starts at -20 now
        key=f"loud_{st.session_state['reset_nonce']}",
        help="Average loudness in dB (typical Spotify tracks are between -14 and -5 dB). Closer to 0 means louder."
    )

with c2:
    tempo = st.slider(
        "🥁 tempo (BPM)", 40.0, 220.0, 120.0,
        key=f"tempo_{st.session_state['reset_nonce']}",
        help="Estimated tempo in beats per minute."
    )

    artist_popularity = st.slider(
        "⭐ artist_popularity", 0, 100, 50,
        key=f"apop_{st.session_state['reset_nonce']}",
        help="Spotify popularity score for the artist (0–100)."
    )

    valence = st.slider(
        "😊 valence", 0.0, 1.0, 0.50,
        key=f"val_{st.session_state['reset_nonce']}",
        help="Musical positivity (0–1). Higher is happier."
    )

    release_year = st.slider(
        "📅 release_year", 1950, 2025, 2020,
        key=f"year_{st.session_state['reset_nonce']}",
        help="Release year of the track."
    )

# =========================
# Advanced Features
# =========================
with st.expander("🧪 Advanced (optional)", expanded=False):
    use_exact_duration = st.checkbox(
        "✍️ Enter exact duration (seconds)",
        value=False,
        key=f"exdur_{st.session_state['reset_nonce']}"
    )
    use_exact_followers = st.checkbox(
        "✍️ Enter exact followers",
        value=False,
        key=f"exfol_{st.session_state['reset_nonce']}"
    )

    if use_exact_duration:
        duration_sec = int(
            st.number_input(
                "⏱️ Exact duration (seconds)",
                min_value=1, max_value=36000,
                value=int(duration_sec), step=1,
                key=f"dur_in_{st.session_state['reset_nonce']}"
            )
        )

    if use_exact_followers:
        followers_exact = int(
            st.number_input(
                "👥 Exact followers",
                min_value=0, max_value=2_000_000_000,
                value=int(artist_followers_k * 1000),
                step=1000,
                key=f"fol_in_{st.session_state['reset_nonce']}"
            )
        )
        artist_followers_k = followers_exact // 1000

    speechiness = st.slider(
        "🗣️ speechiness", 0.0, 1.0, 0.05,
        key=f"sp_{st.session_state['reset_nonce']}"
    )
    acousticness = st.slider(
        "🎻 acousticness", 0.0, 1.0, 0.20,
        key=f"ac_{st.session_state['reset_nonce']}"
    )
    instrumentalness = st.slider(
        "🎼 instrumentalness", 0.0, 1.0, 0.00,
        key=f"ins_{st.session_state['reset_nonce']}"
    )
    liveness = st.slider(
        "🎤 liveness", 0.0, 1.0, 0.15,
        key=f"liv_{st.session_state['reset_nonce']}"
    )

if "speechiness" not in locals():
    speechiness, acousticness, instrumentalness, liveness = 0.05, 0.20, 0.00, 0.15

# =========================
# Build feature row
# =========================
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

# =========================
# Predict
# =========================
if st.button("🎯 Predict", key=f"pred_{st.session_state['reset_nonce']}"):
    if hasattr(model, "predict_proba"):
        hit_prob = float(model.predict_proba(X)[:, 1][0])
    else:
        score = float(model.decision_function(X)[0])
        hit_prob = 1.0 / (1.0 + np.exp(-score))

    if hit_prob >= TH:
        st.success(f"✅ HIT  |  P(hit)={hit_prob:.3f}  |  Threshold={TH:.3f}")
    else:
        st.warning(f"❌ NOT HIT  |  P(hit)={hit_prob:.3f}  |  Threshold={TH:.3f}")

    st.progress(min(max(hit_prob, 0.0), 1.0))
