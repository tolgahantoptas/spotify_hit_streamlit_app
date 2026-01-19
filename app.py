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
h2 {margin-top: 1.2rem;}
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
 --bg:#0e1117; --txt:#ffffff; --muted:rgba(255,255,255,0.85);
 --control-bg:#141a24; --control-border:rgba(255,255,255,0.28);
 --menu-bg:#141a24; --menu-hover:rgba(255,255,255,0.14);
 --btn-bg:#1b2330; --btn-hover:#263244;
}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
h1,h2,h3,h4,h5,h6,p,span,label,small{color:var(--txt)!important;opacity:1!important;}
.metric-hint{color:var(--muted)!important;}
div[data-baseweb="select"]>div{background:var(--control-bg)!important;border-color:var(--control-border)!important;}
div[data-baseweb="select"] span,div[data-baseweb="select"] input,div[data-baseweb="select"] svg{color:#ffffff!important;fill:#ffffff!important;}
div[role="listbox"],ul[role="listbox"]{background:var(--menu-bg)!important;border:1px solid var(--control-border)!important;}
div[role="option"],li[role="option"]{color:#ffffff!important;background:var(--menu-bg)!important;}
div[role="option"][aria-selected="true"],li[role="option"][aria-selected="true"],
div[role="option"]:hover,li[role="option"]:hover{background:var(--menu-hover)!important;}
.stButton>button{background:var(--btn-bg)!important;color:#ffffff!important;border-radius:10px!important;font-weight:650!important;}
.stButton>button:hover{background:var(--btn-hover)!important;}
[data-baseweb="toggle"] span{color:#ffffff!important;font-weight:700!important;}
[data-baseweb="toggle"] [role="switch"][aria-checked="true"]{background:#1db954!important;}
</style>
"""

LIGHT_CSS = """
<style>
:root{
 --bg:#ffffff; --txt:#000000; --muted:rgba(0,0,0,0.80);
 --control-bg:#ffffff; --control-border:rgba(0,0,0,0.30);
 --menu-bg:#ffffff; --menu-hover:rgba(0,0,0,0.06);
 --btn-bg:#f3f4f6; --btn-hover:#e5e7eb;
}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
h1,h2,h3,h4,h5,h6,p,span,label,small{color:var(--txt)!important;opacity:1!important;}
.metric-hint{color:var(--muted)!important;}
div[data-baseweb="select"]>div{background:var(--control-bg)!important;border-color:var(--control-border)!important;}
div[data-baseweb="select"] span,div[data-baseweb="select"] input,div[data-baseweb="select"] svg{color:#000000!important;fill:#000000!important;}
div[role="listbox"],ul[role="listbox"]{background:var(--menu-bg)!important;border:1px solid var(--control-border)!important;}
div[role="option"],li[role="option"]{color:#000000!important;background:var(--menu-bg)!important;}
div[role="option"][aria-selected="true"],li[role="option"][aria-selected="true"],
div[role="option"]:hover,li[role="option"]:hover{background:var(--menu-hover)!important;}
.stButton>button{background:var(--btn-bg)!important;color:#000000!important;border-radius:10px!important;font-weight:650!important;}
.stButton>button:hover{background:var(--btn-hover)!important;}
[data-baseweb="toggle"] span{color:#000000!important;font-weight:800!important;}
[data-baseweb="toggle"] [role="switch"][aria-checked="true"]{background:#1db954!important;}
</style>
"""

st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.markdown(DARK_CSS if st.session_state["theme_mode"]=="dark" else LIGHT_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH))
    genre_freq = json.load(open(GENRE_MAP_PATH))
    genre_freq = {k.lower(): float(v) for k,v in genre_freq.items()}
    return model, meta, genre_freq

model, meta, GENRE_FREQ_MAP = load_artifacts()
TH = float(meta["threshold"])
FEATURES = meta["feature_columns"]

if "reset_nonce" not in st.session_state:
    st.session_state["reset_nonce"] = 0

def do_reset():
    keep={"reset_nonce","theme_mode"}
    for k in list(st.session_state.keys()):
        if k not in keep: del st.session_state[k]
    st.session_state["reset_nonce"]+=1

top_l, top_r = st.columns([0.72,0.28])
with top_r:
    is_dark = st.toggle("🌙 Dark mode", value=st.session_state["theme_mode"]=="dark")
    new_mode = "dark" if is_dark else "light"
    if new_mode!=st.session_state["theme_mode"]:
        st.session_state["theme_mode"]=new_mode
        st.rerun()

with top_l:
    c1,c2=st.columns([0.1,0.9])
    with c1:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH,width=72)
    with c2:
        st.title("Spotify Hit Predictor")
        st.caption("Choose a genre, set the audio and artist features, then click Predict.")

st.button("🔄 Reset",on_click=do_reset)

super_map={
"pop":"Pop","indie-pop":"Pop","k-pop":"Pop","j-pop":"Pop","mandopop":"Pop","cantopop":"Pop","british":"Pop",
"rock":"Rock/Metal","alt-rock":"Rock/Metal","metal":"Rock/Metal","punk":"Rock/Metal","grunge":"Rock/Metal","emo":"Rock/Metal",
"hip-hop":"Hip-Hop/R&B","r-n-b":"Hip-Hop/R&B","soul":"Hip-Hop/R&B","funk":"Hip-Hop/R&B",
"edm":"Electronic/Dance","electronic":"Electronic/Dance","house":"Electronic/Dance","techno":"Electronic/Dance",
"latin":"Latin/Reggae","reggae":"Latin/Reggae","reggaeton":"Latin/Reggae",
"acoustic":"Acoustic/Folk/Country","folk":"Acoustic/Folk/Country","country":"Acoustic/Folk/Country",
"classical":"Classical/Jazz","piano":"Classical/Jazz","jazz":"Classical/Jazz"
}

genres=sorted(GENRE_FREQ_MAP.keys())
groups=sorted(set(super_map.get(g,"Other") for g in genres))

st.header("🎛️ Genre Selection")
g1,g2=st.columns(2)
with g1:
    chosen_super=st.selectbox("🎧 Genre",groups,index=groups.index("Pop") if "Pop" in groups else 0)
with g2:
    subs=[g for g in genres if super_map.get(g,"Other")==chosen_super]
    chosen_genre=st.selectbox("🏷️ Alt genre",subs)

track_genre_freq=GENRE_FREQ_MAP.get(chosen_genre,0.0)

st.header("📌 Basic Features")
c1,c2=st.columns(2)
with c1:
    duration_sec=st.slider("⏱️ duration (sec)",30,900,180)
    st.markdown(f"<div class='metric-hint'>= {duration_sec//60} min {duration_sec%60:02d} sec</div>",unsafe_allow_html=True)
    artist_followers_k=st.slider("👥 artist_followers (K)",0,150000,100,step=100)
    st.markdown(f"<div class='metric-hint'>= {artist_followers_k*1000:,} followers</div>",unsafe_allow_html=True)
    danceability=st.slider("💃 danceability",0.0,1.0,0.5)
    energy=st.slider("🔋 energy",0.0,1.0,0.5)
    loudness=st.slider("🔊 loudness (dB)",-20.0,0.0,-8.0)
with c2:
    tempo=st.slider("🥁 tempo (BPM)",40.0,220.0,120.0)
    artist_popularity=st.slider("⭐ artist_popularity",0,100,50)
    valence=st.slider("😊 valence",0.0,1.0,0.5)
    release_year=st.slider("📅 release_year",1950,2025,2020)

with st.expander("🧪 Advanced (optional)"):
    speechiness=st.slider("🗣️ speechiness",0.0,1.0,0.05)
    acousticness=st.slider("🎻 acousticness",0.0,1.0,0.2)
    instrumentalness=st.slider("🎼 instrumentalness",0.0,1.0,0.0)
    liveness=st.slider("🎤 liveness",0.0,1.0,0.15)

row={
"duration_ms":duration_sec*1000,
"danceability":danceability,"energy":energy,"loudness":loudness,
"speechiness":speechiness,"acousticness":acousticness,
"instrumentalness":instrumentalness,"liveness":liveness,
"valence":valence,"tempo":tempo,"release_year":release_year,
"artist_followers":artist_followers_k*1000,
"artist_popularity":artist_popularity,
"track_genre_freq":track_genre_freq
}

X=pd.DataFrame([row])
for c in FEATURES:
    if c not in X: X[c]=0.0
X=X[FEATURES].fillna(0.0)

st.divider()

if st.button("🎯 Predict"):
    hit_prob=float(model.predict_proba(X)[:,1][0])
    if hit_prob>=TH:
        st.success(f"✅ This song would be a HIT! (Hit probability: {hit_prob:.3f})")
    else:
        st.warning(f"❌ This song would NOT be a hit. (Non-hit probability: {1-hit_prob:.3f})")
