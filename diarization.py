import numpy as np
import librosa
from scipy.signal import resample_poly, medfilt
from sklearn.cluster import KMeans
import soundfile as sf
import matplotlib as plt


# ---------------- VAD ---------------- #
def to_mono_16k(x, sr):
    """Ensure mono float32 at 16kHz."""
    if x.dtype.kind in "iu":
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    target_sr = 16000
    if sr != target_sr:
        from math import gcd

        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up, down)
        sr = target_sr
    return x, sr


def frame_signal(x, sr, win=0.025, hop=0.010):
    wlen = int(round(win * sr))
    hlen = int(round(hop * sr))
    num_frames = 1 + max(0, (len(x) - wlen) // hlen)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num_frames, wlen),
        strides=(x.strides[0] * hlen, x.strides[0]),
        writeable=False,
    ).copy()
    frames *= np.hanning(wlen)[None, :]
    return frames


def log_energy(frames, eps=1e-10):
    e = (frames.astype(np.float32) ** 2).sum(axis=1) + eps
    return np.log(e).astype(np.float32)


def mask_to_regions(mask, hop_s, min_speech=0.1, min_gap=0.2):
    mask = mask.astype(np.uint8)
    diffs = np.diff(mask, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    regions = []
    for s, e in zip(starts, ends):
        if e > s:
            regions.append([s * hop_s, e * hop_s])
    merged = []
    for seg in regions:
        if not merged:
            merged.append(seg)
        else:
            if seg[0] - merged[-1][1] < min_gap:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
    return [[s, e] for s, e in merged if (e - s) >= min_speech]


def vad_energy(
    x,
    sr,
    win=0.025,
    hop=0.010,
    perc=30,
    offset=0.5,
    median_k=2,
    min_speech=0.3,
    min_gap=0.2,
):
    frames = frame_signal(x, sr, win, hop)
    e = log_energy(frames)
    thr = np.percentile(e, perc) - offset
    raw_mask = (e > thr).astype(np.uint8)
    if median_k % 2 == 0:
        median_k += 1
    smooth_mask = medfilt(raw_mask, median_k).astype(bool)
    regions = mask_to_regions(smooth_mask, hop, min_speech=min_speech, min_gap=min_gap)
    return regions


# ---------------- Diarization ---------------- #
def chop_regions(regions, seg_len=2.0, hop=1.0):
    segs = []
    for start, end in regions:
        t = start
        while t + seg_len <= end:
            segs.append((t, t + seg_len))
            t += hop
    return segs


def embed_mfcc(sig, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def merge_segments_by_label(segments, labels, min_dur=0.4):
    merged = []
    for seg, lbl in zip(segments, labels):
        if not merged:
            merged.append([lbl, seg[0], seg[1]])
        else:
            if lbl == merged[-1][0]:
                merged[-1][2] = seg[1]
            else:
                merged.append([lbl, seg[0], seg[1]])
    return [[lbl, s, e] for (lbl, s, e) in merged if (e - s) >= min_dur]


def diarize_two_speakers(wav_path):
    x, sr = sf.read(wav_path)
    x, sr = to_mono_16k(x, sr)

    # VAD
    speech_regions = vad_energy(x, sr)

    # Chop into segments
    segments = chop_regions(speech_regions, seg_len=2.0, hop=1.0)

    # Extract embeddings
    feats = []
    for s, e in segments:
        seg_audio = x[int(s * sr) : int(e * sr)]
        feats.append(embed_mfcc(seg_audio, sr))
    feats = np.vstack(feats)

    # K-means clustering into 2 speakers
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(feats)

    # Merge segments into continuous turns
    turns = merge_segments_by_label(segments, labels)
    return turns


# ---------------- Run Example ---------------- #
if __name__ == "__main__":
    wav_path = "audiotest_2.wav"  # replace with your audio
    turns = diarize_two_speakers(wav_path)
    for lbl, s, e in turns:
        time = e - s
        print(f"Speaker_{lbl}: {s:.2f}s - {e:.2f}s: {time:.2f}s")
