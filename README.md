# 🎵 Spotify Hit Predictor

A Streamlit-based web application that predicts whether a Spotify track is likely to become a **hit**, using a machine learning model trained on audio features, artist-level information, and genre popularity signals.

---

## 📌 Project Description

This project formulates **Spotify hit prediction** as a **binary classification problem**.

A track is predicted as a **Hit (1)** if its estimated probability exceeds a predefined threshold; otherwise, it is classified as **Non-Hit (0)**.  
The threshold and feature configuration are stored externally and reused consistently during deployment.

The main goal is to provide an **interpretable, reproducible, and deployment-ready** machine learning application rather than only a standalone model.

---

## 🧠 Model

- **Algorithm:** HistGradientBoostingClassifier (scikit-learn)
- **Model type:** Histogram-based Gradient Boosting for tabular data
- **Inference:** Probability-based decision with a fixed threshold

### Final Hyperparameters

```json
{
  "max_iter": 300,
  "max_depth": 10,
  "learning_rate": 0.03,
  "min_samples_leaf": 20
}
```

**Why HistGradientBoostingClassifier?**
- Strong performance on structured/tabular datasets
- Efficient training and inference
- Captures non-linear interactions between features
- Does not require feature scaling
- Well-suited for production deployment

---

## 🎛 Features Used

The model uses the following features during inference:

- duration_ms
- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- release_year
- artist_followers
- artist_popularity
- track_genre_freq

---

## 🎼 Genre Handling Strategy

Instead of one-hot encoding genres, the model uses a compact numeric representation:

- **track_genre_freq**  
  A continuous value representing how frequently a given genre appears in the training dataset.

For usability in the Streamlit interface:
1. Users select a **genre group** (e.g., Pop, Rock/Metal).
2. Users then select a **sub-genre**.
3. The application automatically maps the selection to the corresponding track_genre_freq.

This design:
- Avoids high-dimensional sparse encoding
- Preserves statistical genre information
- Keeps the deployed model unchanged

---

## 📦 Artifacts

The trained model and metadata are stored under the `artifacts/` directory.

Key files:
- final_model.joblib – trained model
- meta.json – feature list, threshold, and configuration
- genre_freq_map.json – genre → frequency mapping

These artifacts ensure consistent behavior between training and deployment.

---

## 🌐 Streamlit Application

The Streamlit app provides:
- Interactive feature sliders
- Two-step genre selection
- Dark / Light mode support
- Contextual help for each input
- Probability-based prediction output

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Repository Structure

```
.
├── app.py
├── artifacts/
│   ├── final_model.joblib
│   ├── meta.json
│   └── genre_freq_map.json
├── assets/
│   └── spotify_icon.png
├── requirements.txt
└── README.md
```

---

## 📊 Evaluation & Reproducibility

- Fixed train/validation/test splits
- Stored decision threshold
- Externalized feature order
- Reproducible inference using saved artifacts

This makes the project suitable for:
- Machine learning portfolios
- End-to-end ML deployment demonstrations
- Academic or applied ML studies

---
## 👤 Author

**Tolgahan Toptaş**  
Machine Learning / Data Science

## 📄 License

This project is intended for **educational and research purposes**.  
Commercial use requires permission.