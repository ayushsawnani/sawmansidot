# app.py
import os, io, joblib, tempfile
from io import BytesIO
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from diarization import diarize_two_speakers, to_mono_16k
from features_extraction import extract_window_features

st.set_page_config(page_title="Sawman's IDOT", layout="centered")


@st.cache_resource
def load_model():
    # auto-pick newest model in models/
    import glob

    files = glob.glob("models/*.pkl")
    if not files:
        st.error("No model found in models/. Upload xgb_interest.pkl.")
        st.stop()
    latest = max(files, key=os.path.getmtime)
    bundle = joblib.load(latest)
    return bundle, os.path.basename(latest)


st.title("🎧 Sawman's Interest Detection Over Time")

bundle, model_name = load_model()
# st.caption(f"Loaded model: `{model_name}`")

uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded:
    st.audio(uploaded, format="audio/wav")
win = st.number_input("Window (s)", value=5.0, min_value=2.0, max_value=15.0, step=0.5)
hop = st.number_input("Hop (s)", value=2.5, min_value=0.5, max_value=10.0, step=0.5)
thresh = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded is not None:
    # read bytes → array
    raw_bytes = uploaded.read()
    data, sr = sf.read(BytesIO(raw_bytes))
    y, sr = to_mono_16k(data, sr)

    st.write("**Step 1:** Diarizing…")
    with st.spinner("Running diarization (2 speakers)…"):
        # If your diarizer expects a file path, write a temp WAV and pass the path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, y, sr)
            tmp_path = tmp.name
        try:
            turns = diarize_two_speakers(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not turns:
        st.warning("No speaker turns found.")
        st.stop()

    dur0 = sum(e - s for spk, s, e in turns if spk == 0)
    dur1 = sum(e - s for spk, s, e in turns if spk == 1)
    default_spk = 0 if dur0 >= dur1 else 1

    # --- Snippet audition for each diarized speaker ---
    def _make_snippet_bytes(y_arr, sr_val, turns_list, speaker_id, max_len=2.0):
        """Return a small WAV bytes sample for the first turn of a given speaker."""
        for sp, s, e in turns_list:
            if sp == speaker_id:
                s_i = int(max(0.0, s) * sr_val)
                e_i = int(min(e, s + max_len) * sr_val)
                seg = y_arr[s_i:e_i]
                if seg.size < int(0.2 * sr_val):  # need at least 0.2s to preview
                    continue
                buf = BytesIO()
                sf.write(buf, seg, sr_val, format="WAV", subtype="PCM_16")
                buf.seek(0)
                return buf.read()
        return None

    snip0 = _make_snippet_bytes(y, sr, turns, 0)
    snip1 = _make_snippet_bytes(y, sr, turns, 1)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Preview: Speaker 0")
        if snip0:
            st.audio(snip0, format="audio/wav")
        else:
            st.caption("No usable snippet for Speaker 0")
    with c2:
        st.write("Preview: Speaker 1")
        if snip1:
            st.audio(snip1, format="audio/wav")
        else:
            st.caption("No usable snippet for Speaker 1")

    st.write("**Step 2:** Choose target speaker")
    target = st.radio(
        "Target speaker", [0, 1], index=(0 if default_spk == 0 else 1), horizontal=True
    )

    if st.button("Extract features & Predict"):
        with st.spinner("Extracting features…"):
            df = extract_window_features(
                y, sr, turns, target_speaker=int(target), win=win, hop=hop
            )
        df["file_id"] = os.path.splitext(uploaded.name)[0]

        model = bundle["model"]
        feat_cols = bundle["features"]
        scaler = bundle.get("scaler", None)

        with st.spinner("Predicting interest…"):
            try:
                X = df[feat_cols].astype(float).values
                if scaler is not None:
                    X = scaler.transform(X)
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X)[:, 1]
                elif hasattr(model, "decision_function"):
                    from sklearn.preprocessing import MinMaxScaler

                    scores = (
                        MinMaxScaler()
                        .fit_transform(model.decision_function(X).reshape(-1, 1))
                        .ravel()
                    )
                else:
                    pred = model.predict(X)
                    scores = pred if getattr(pred, "ndim", 1) == 1 else pred.ravel()
            except KeyError as e:
                missing = [c for c in feat_cols if c not in df.columns]
                st.error(f"Feature mismatch. Missing columns: {missing}")
                st.stop()

        if len(df):
            df["pred_score"] = scores
            d = df.sort_values("t_start")
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(d["t_start"], d["pred_score"])
            ax.axhline(thresh, linestyle="--", alpha=0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Score (0–1)")
            ax.set_title(f"Interest over time — {uploaded.name} (Speaker_{target})")
            st.pyplot(fig)
            st.caption(
                f"Target speaker: Speaker_{target} | Windows: {len(df)} | Win={win}s Hop={hop}s"
            )

            st.metric("Mean score", f"{df['pred_score'].mean():.3f}")

            st.download_button(
                "Download window predictions (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_preds.csv",
                mime="text/csv",
            )
        else:
            st.warning("Nothing extracted from the target speaker.")
