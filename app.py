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

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
      h1 {margin-bottom: 0.2rem;}
      h2 {margin-top: 1.2rem;}
      .stButton>button {border-radius: 10px; padding: 0.55rem 1.1rem;}
      .stExpander {border-radius: 14px;}
      [data-baseweb="slider"] {padding-top: 0.35rem;}
      .metric-hint {color: rgba(255,255,255,0.75); font-size: 0.95rem; margin-top: -0.25rem;}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_freq_map = json.load(f)
    genre_freq_map = {str(k).strip().lower(): float(v) for k, v in genre_freq_map.items()}
    return model, meta, genre_freq_map

model, meta, GENRE_FREQ_MAP = load_artifacts()

TH = float(meta.get("threshold", 0.67))
FEATURES = meta.get("feature_columns", []) or meta.get("feature_cols", [])
if not FEATURES or "track_genre_freq" not in FEATURES:
    st.error("Invalid model metadata.")
    st.stop()

if "reset_nonce" not in st.session_state:
    st.session_state["reset_nonce"] = 0

def do_reset():
    keep = {"reset_nonce"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state["reset_nonce"] += 1

h1, h2 = st.columns([0.08, 0.92], vertical_alignment="center")
with h1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=72)
    else:
        st.write("🎵")
with h2:
    st.title("Spotify Hit Predictor")
    st.caption("Choose a genre, set the audio/artist features, then click Predict. Use the ? icons for short, practical explanations.")

st.button("🔄 Reset", on_click=do_reset, key="reset_btn")

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
        help="Pick a broad category to make the next list shorter (this does NOT go into the model directly)."
    )

with g2:
    sub_list = [g for g in genres if super_map.get(g, "Other") == chosen_super]
    chosen_genre = st.selectbox(
        "🏷️ Alt genre",
        sub_list if sub_list else genres,
        index=0,
        key=f"alt_genre_{st.session_state['reset_nonce']}",
        help="This is the genre label used by the model. The app converts it into track_genre_freq (how common the genre is in the training data)."
    )

track_genre_freq = float(GENRE_FREQ_MAP.get(chosen_genre, 0.0))

st.header("📌 Basic Features")
c1, c2 = st.columns(2)

with c1:
    duration_sec = st.slider(
        "⏱️ duration (sec)", 30, 900, 180,
        key=f"duration_{st.session_state['reset_nonce']}",
        help="Track length in seconds. Most songs are ~120–240 seconds (2–4 minutes)."
    )
    st.markdown(f"<div class='metric-hint'>≈ {duration_sec//60} min {duration_sec%60:02d} sec</div>", unsafe_allow_html=True)

    artist_followers_k = st.slider(
        "👥 artist_followers (K)", 0, 150_000, 100, step=100,
        key=f"followers_{st.session_state['reset_nonce']}",
        help="Artist followers in thousands (K). Example: 250K means 250,000 followers."
    )
    st.markdown(f"<div class='metric-hint'>≈ {artist_followers_k*1000:,} followers</div>", unsafe_allow_html=True)

    danceability = st.slider(
        "💃 danceability", 0.0, 1.0, 0.50,
        key=f"dance_{st.session_state['reset_nonce']}",
        help="How dance-friendly the track is (0–1). Higher values usually mean a steadier beat and easier rhythm for dancing."
    )

    energy = st.slider(
        "🔋 energy", 0.0, 1.0, 0.50,
        key=f"energy_{st.session_state['reset_nonce']}",
        help="Perceived intensity and activity (0–1). Higher values often feel louder, faster, and more energetic."
    )

    loudness = st.slider(
        "🔊 loudness (dB)", -20.0, 0.0, -8.0,
        key=f"loud_{st.session_state['reset_nonce']}",
        help="Average loudness in dB. Typical Spotify tracks are around -14 to -5 dB. Closer to 0 means louder mastering."
    )

with c2:
    tempo = st.slider(
        "🥁 tempo (BPM)", 40.0, 220.0, 120.0,
        key=f"tempo_{st.session_state['reset_nonce']}",
        help="Estimated tempo in beats per minute (BPM). Pop is often ~90–140 BPM."
    )

    artist_popularity = st.slider(
        "⭐ artist_popularity", 0, 100, 50,
        key=f"apop_{st.session_state['reset_nonce']}",
        help="Spotify popularity score (0–100). Higher usually means the artist is more widely listened to right now."
    )

    valence = st.slider(
        "😊 valence", 0.0, 1.0, 0.50,
        key=f"val_{st.session_state['reset_nonce']}",
        help="Musical positivity (0–1). Higher = happier/cheerful; lower = sad/angry."
    )

    release_year = st.slider(
        "📅 release_year", 1950, 2025, 2020,
        key=f"year_{st.session_state['reset_nonce']}",
        help="Release year of the track. Some years/eras may correlate with hit likelihood depending on the dataset."
    )

with st.expander("🧪 Advanced (optional)", expanded=False):
    speechiness = st.slider(
        "🗣️ speechiness", 0.0, 1.0, 0.05,
        key=f"sp_{st.session_state['reset_nonce']}",
        help="Amount of spoken words (0–1). Higher values often appear in rap or spoken-word tracks."
    )
    acousticness = st.slider(
        "🎻 acousticness", 0.0, 1.0, 0.20,
        key=f"ac_{st.session_state['reset_nonce']}",
        help="Probability the track is acoustic (0–1). Higher values usually mean more unplugged / acoustic sound."
    )
    instrumentalness = st.slider(
        "🎼 instrumentalness", 0.0, 1.0, 0.00,
        key=f"ins_{st.session_state['reset_nonce']}",
        help="Likelihood of no vocals (0–1). Near 0 means vocals present; higher means more instrumental."
    )
    liveness = st.slider(
        "🎤 liveness", 0.0, 1.0, 0.15,
        key=f"liv_{st.session_state['reset_nonce']}",
        help="Detects audience / live performance feeling (0–1). Higher values are more likely recorded live."
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

if st.button("Predict", key=f"pred_{st.session_state['reset_nonce']}"):
    if hasattr(model, "predict_proba"):
        hit_prob = float(model.predict_proba(X)[:, 1][0])
    else:
        hit_prob = float(model.decision_function(X)[0])

    non_hit_prob = float(1.0 - hit_prob)

    if hit_prob >= TH:
        st.success(f"This song would be a HIT! (Hit probability: {hit_prob:.3f})")
    else:
        st.warning(f"This song would NOT be a hit. (Non-hit probability: {non_hit_prob:.3f})")
