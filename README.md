# 🎧 Sawman's Interest Detection Over Time

## Overview

Interest Detection Over Time is a machine learning–powered tool that processes speech recordings, isolates a target speaker, extracts relevant acoustic features, and predicts the level of listener interest (or engagement) over time.

The application is built with:
- Python for audio processing and ML training
- XGBoost for classification
- Streamlit for a web-based demo

### Features
- Speaker diarization → separates two speakers so you can choose the target one
- Feature extraction from target speaker windows
- Real-time predictions with visual score plots
- CSV export for predictions
- Adjustable window size, hop size, and prediction threshold

## Pipeline
1.	Upload Audio (WAV)
- The app resamples audio to mono @ 16 kHz.
2.	Diarization (2 speakers)
- Splits the recording into turns for Speaker 0 and Speaker 1.
- Plays short snippets for each speaker so the user can identify the target.
3.	Target Speaker Selection
- User selects which speaker to analyze.
4.	Feature Extraction
- Computes acoustic features per time window.
5.	Prediction
- Applies the XGBoost model to each window.
- Produces a time-series plot of interest scores.
6.	Download Results
- Exports a CSV of timestamps + predicted interest scores.

## Installation

### Install Dependencies
  `pip install -r requirements.txt`

### Key dependencies:
- numpy, pandas
- librosa, soundfile
- matplotlib
- scikit-learn, xgboost
- streamlit


## Roadmap
- Add support for >2 speakers
- Cloud inference API for mobile/remote clients

## License

MIT License. See LICENSE for details.
