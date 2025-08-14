# features_extraction.py
# Windowed prosody + spectral + simple conversation-dynamics features.
# Depends on: numpy, librosa, pandas, soundfile
# Uses diarization.py for: diarize_two_speakers, to_mono_16k

from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os
import tempfile
from tkinter import filedialog, messagebox
import sounddevice as sd

from diarization import diarize_two_speakers, to_mono_16k


def play_audio_segment(y, sr, start_s, end_s):
    """Play a short segment of audio."""
    start_idx = int(start_s * sr)
    end_idx = int(end_s * sr)
    sd.play(y[start_idx:end_idx], sr)
    sd.wait()


# ---------- timeline helpers ----------
def build_timelines(turns, step=0.01):
    """
    turns: list of [spk, start, end]
    Returns:
      times: [T] grid in seconds
      who:   [T] -1 none, 0 spk0, 1 spk1
      spans: dict {0:[(s,e),...], 1:[(s,e),...]}
    """
    spans = {0: [], 1: []}
    T_end = 0.0
    for spk, s, e in turns:
        spans[int(spk)].append((float(s), float(e)))
        T_end = max(T_end, float(e))
    step = float(step)
    times = np.arange(0, T_end + step, step)
    who = -1 * np.ones_like(times, dtype=int)
    for spk in (0, 1):
        for s, e in spans[spk]:
            i0, i1 = int(np.floor(s / step)), int(np.ceil(e / step))
            who[i0:i1] = spk
    # ensure spans are sorted
    spans[0].sort(key=lambda x: x[0])
    spans[1].sort(key=lambda x: x[0])
    return times, who, spans


def window_overlap_fraction(win_s, win_e, spans):
    total = 0.0
    for s, e in spans:
        a = max(s, win_s)
        b = min(e, win_e)
        if b > a:
            total += b - a
    return total / max(1e-6, (win_e - win_s))


def compute_response_latencies(spans_target, spans_other, max_gap=5.0):
    lat = []
    j = 0
    for os, oe in spans_other:
        while j < len(spans_target) and spans_target[j][1] <= oe:
            j += 1
        if j < len(spans_target):
            ts = spans_target[j][0]
            if ts >= oe and (ts - oe) <= max_gap:
                lat.append(ts - oe)
    return lat


def short_utterance_count(spans_target, spans_other, max_len=0.6):
    cnt = 0
    for s, e in spans_target:
        if (e - s) <= max_len:
            # near the other speaker (overlap or within 1s)
            near = any(
                (abs(s - so) < 1.0 or abs(e - eo) < 1.0 or (s < eo and e > so))
                for (so, eo) in spans_other
            )
            if near:
                cnt += 1
    return cnt


# ---------- audio features ----------
def slice_audio(y, sr, start_s, end_s):
    s = int(start_s * sr)
    e = min(int(end_s * sr), len(y))
    return y[s:e]


def f0_stats(y, sr, fmin=70, fmax=350):
    f0, _, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256
    )
    if f0 is None:
        return dict(f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0, f0_range=0.0)
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return dict(f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0, f0_range=0.0)
    return dict(
        f0_mean=float(np.mean(f0)),
        f0_std=float(np.std(f0)),
        f0_min=float(np.min(f0)),
        f0_max=float(np.max(f0)),
        f0_range=float(np.max(f0) - np.min(f0)),
    )


def prosody_spectral(y, sr):
    # Energy / rhythm / timbre-ish stats over the window
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    feats = dict(
        rms_mean=float(rms.mean()),
        rms_std=float(rms.std()),
        zcr_mean=float(zcr.mean()),
        zcr_std=float(zcr.std()),
        cent_mean=float(cent.mean()),
        cent_std=float(cent.std()),
        roll_mean=float(roll.mean()),
        roll_std=float(roll.std()),
        flat_mean=float(flat.mean()),
        flat_std=float(flat.std()),
    )
    feats.update(f0_stats(y, sr))
    # very rough speaking-rate proxy (no ASR): onset peaks per second
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=5
    )
    dur_s = len(y) / float(sr)
    feats["onset_rate_hz"] = float(len(peaks) / max(dur_s, 1e-6))
    return feats


# ---------- main extractor ----------
def extract_window_features(y, sr, turns, target_speaker=1, win=5.0, hop=2.5):
    """
    Returns a pandas DataFrame with one row per window:
      [t_start, t_end, frac_target, frac_other, frac_silence,
       rms/zcr/cent/roll/flat stats, f0 stats, onset_rate_hz,
       resp_latency_mean_global, short_backchannel_count_global]
    """
    times, who, spans = build_timelines(turns, step=0.01)
    target = int(target_speaker)
    other = 1 - target

    spans_t = spans[target]
    spans_o = spans[other]

    # global conversation dynamics (not window-specific)
    latencies = compute_response_latencies(spans_t, spans_o)
    global_latency_mean = float(np.mean(latencies)) if latencies else 0.0
    global_short_bkch = short_utterance_count(spans_t, spans_o, max_len=0.6)

    rows = []
    T = times[-1] if len(times) else 0.0
    t = 0.0
    while t + win <= T + 1e-9:
        t0, t1 = t, t + win

        # fractions of time in window
        frac_target = window_overlap_fraction(t0, t1, spans_t)
        frac_other = window_overlap_fraction(t0, t1, spans_o)
        frac_sil = max(0.0, 1.0 - frac_target - frac_other)

        # simple: use full window audio for features (fast + robust enough)
        y_win = slice_audio(y, sr, t0, t1)
        feats = prosody_spectral(y_win, sr)
        feats.update(
            dict(
                t_start=t0,
                t_end=t1,
                frac_target=frac_target,
                frac_other=frac_other,
                frac_silence=frac_sil,
                resp_latency_mean_global=global_latency_mean,
                short_backchannel_count_global=global_short_bkch,
            )
        )
        rows.append(feats)
        t += hop

    return pd.DataFrame(rows)
    root = tk.Tk()
    root.title("Interest Feature Extraction with Playback")
    root.geometry("580x500")

    state = {"wav": None, "turns": None, "y": None, "sr": None}

    def pick_file():
        f = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav")],
        )
        if not f:
            return
        state["wav"] = f
        lbl_file.config(text=f"File: {os.path.basename(f)}")

        # Load audio for later playback
        y, sr = sf.read(f)
        y, sr = to_mono_16k(y, sr)
        state["y"], state["sr"] = y, sr

        turns = diarize_two_speakers(f)
        state["turns"] = turns
        dur0 = sum(e - s for spk, s, e in turns if spk == 0)
        dur1 = sum(e - s for spk, s, e in turns if spk == 1)

        txt = []
        txt.append(f"Speaker_0 total={dur0:.2f}s, Speaker_1 total={dur1:.2f}s\n")
        for spk, s, e in turns[:6]:
            txt.append(f"Speaker_{spk}: {s:.2f}-{e:.2f}s ({e-s:.2f}s)")
        txt_turns.config(state="normal")
        txt_turns.delete("1.0", "end")
        txt_turns.insert("1.0", "\n".join(txt))
        txt_turns.config(state="disabled")

    def play_sample(spk):
        if not state["turns"] or state["y"] is None:
            messagebox.showwarning("No audio", "Load a WAV first.")
            return
        for spk_id, s, e in state["turns"]:
            if spk_id == spk:
                play_audio_segment(state["y"], state["sr"], s, e)
                break

    def run_extract():
        if not state["wav"] or not state["turns"]:
            messagebox.showwarning("Missing", "Pick a file first.")
            return
        target = var_target.get()
        out_csv = filedialog.asksaveasfilename(
            title="Save features CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not out_csv:
            return
        df = extract_window_features(
            state["y"],
            state["sr"],
            state["turns"],
            target_speaker=target,
            win=float(ent_win.get()),
            hop=float(ent_hop.get()),
        )
        df.to_csv(out_csv, index=False)
        messagebox.showinfo("Done", f"Saved to {out_csv}")

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    tk.Button(frm, text="Select WAV…", command=pick_file).grid(
        row=0, column=0, sticky="w"
    )
    lbl_file = tk.Label(frm, text="File: (none)")
    lbl_file.grid(row=0, column=1, columnspan=2, sticky="w")

    tk.Label(frm, text="Window (s):").grid(row=1, column=0, sticky="w", pady=(10, 0))
    ent_win = tk.Entry(frm, width=6)
    ent_win.insert(0, str(default_win))
    ent_win.grid(row=1, column=1, sticky="w")

    tk.Label(frm, text="Hop (s):").grid(row=2, column=0, sticky="w")
    ent_hop = tk.Entry(frm, width=6)
    ent_hop.insert(0, str(default_hop))
    ent_hop.grid(row=2, column=1, sticky="w")

    tk.Label(frm, text="Target speaker:").grid(
        row=3, column=0, sticky="w", pady=(10, 0)
    )
    var_target = tk.IntVar(value=1)
    rb0 = tk.Radiobutton(frm, text="Speaker_0", variable=var_target, value=0)
    rb1 = tk.Radiobutton(frm, text="Speaker_1", variable=var_target, value=1)
    rb0.grid(row=3, column=1, sticky="w", pady=(10, 0))
    rb1.grid(row=3, column=2, sticky="w", pady=(10, 0))

    tk.Button(frm, text="Play sample 0", command=lambda: play_sample(0)).grid(
        row=4, column=1
    )
    tk.Button(frm, text="Play sample 1", command=lambda: play_sample(1)).grid(
        row=4, column=2
    )

    txt_turns = tk.Text(frm, height=12, width=65, state="disabled")
    txt_turns.grid(row=5, column=0, columnspan=3, pady=(10, 0))

    tk.Button(frm, text="Extract & Save CSV…", command=run_extract).grid(
        row=6, column=0, columnspan=3, pady=12
    )

    root.mainloop()


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Extract windowed features after diarization"
    )
    ap.add_argument(
        "wav", help="Path to WAV (mono/stereo ok; will be converted to mono 16k)"
    )
    ap.add_argument(
        "--target", type=int, default=1, help="Target speaker id to track (0 or 1)"
    )
    ap.add_argument("--win", type=float, default=5.0, help="Window length (s)")
    ap.add_argument("--hop", type=float, default=2.5, help="Hop length (s)")
    ap.add_argument("--out_csv", default=None, help="Optional: write features to CSV")
    ap.add_argument(
        "--gui_playback",
        action="store_true",
        help="Launch GUI to choose WAV, select speaker with audio playback, and extract features",
    )
    args = ap.parse_args()

    # 1) Diarize
    turns = diarize_two_speakers(args.wav)
    print("Diarized turns:")
    for spk, s, e in turns:
        print(f"  Speaker_{spk}: {s:.2f}s – {e:.2f}s ({e-s:.2f}s)")

    # 2) Load audio once (mono 16k)
    y, sr = sf.read(args.wav)
    y, sr = to_mono_16k(y, sr)

    # 3) Features
    df = extract_window_features(
        y, sr, turns, target_speaker=args.target, win=args.win, hop=args.hop
    )

    print("\nFeature rows:", len(df))
    print(df.head(8).to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote features to {args.out_csv}")

    # Optional: also save turns JSON so everything’s side-by-side
    turns_json = [
        {"speaker": int(s), "start": float(a), "end": float(b)} for (s, a, b) in turns
    ]
    with open("turns.json", "w") as f:
        json.dump(turns_json, f, indent=2)
        print("Saved diarized turns to turns.json")


if __name__ == "__main__":
    main()
