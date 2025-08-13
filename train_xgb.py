# train_xgb.py — robust training with non-augmented test split and numeric-only features
import argparse, os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

META_COLS = {
    "label",
    "file_id",
    "t_start",
    "t_end",
    "target_speaker",
    "source",
    "is_augmented",
    "aug_variant",
    "orig_row_id",
}
LABEL_COL = "label"  # 0/1 expected


def build_split_real_only(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Create file-level stratified split using ONLY real rows (is_augmented==0)."""
    if "is_augmented" not in df.columns:
        df["is_augmented"] = 0
    df["is_augmented"] = df["is_augmented"].fillna(0).astype(int)

    assert "file_id" in df.columns, "file_id column required"
    assert LABEL_COL in df.columns, f"{LABEL_COL} column required"

    real = df[df["is_augmented"] == 0].copy()
    file_labels = real.groupby("file_id")[LABEL_COL].first().astype(int)

    file_ids = file_labels.index.to_numpy()
    y_files = file_labels.values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(sss.split(file_ids, y_files))
    train_files = set(file_ids[tr_idx])
    test_files = set(file_ids[te_idx])

    # Masks:
    train_mask = df["file_id"].isin(train_files).values  # real + augmented
    test_mask = (
        df["file_id"].isin(test_files).values & (df["is_augmented"] == 0).values
    )  # ONLY real

    return train_files, test_files, train_mask, test_mask


def select_feature_columns(df: pd.DataFrame):
    """Keep numeric-only feature columns, drop metadata/labels."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in META_COLS]
    if not feat_cols:
        raise RuntimeError(
            "No numeric feature columns found after dropping metadata. Check your parquet schema."
        )
    # Helpful log of skipped non-numeric candidates
    non_numeric_candidates = [
        c for c in df.columns if c not in META_COLS and c not in numeric_cols
    ]
    if non_numeric_candidates:
        print("Skipping non-numeric columns:", non_numeric_candidates)
    return feat_cols


def main():
    ap = argparse.ArgumentParser(
        description="Train XGBoost on window features with non-augmented test split."
    )
    ap.add_argument(
        "--data",
        default="data/combined/features_all.parquet",
        help="Combined features parquet",
    )
    ap.add_argument(
        "--model_out",
        default="models/xgb_interest.pkl",
        help="Path to save model bundle",
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--scale",
        action="store_true",
        help="Apply StandardScaler (optional; XGB doesn't require it)",
    )
    ap.add_argument(
        "--save_splits",
        action="store_true",
        help="Save train/test file lists to splits/",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if LABEL_COL not in df.columns:
        raise SystemExit(f"Missing '{LABEL_COL}' in {args.data}")

    # Split by file_id using ONLY real rows for the split/test set
    train_files, test_files, train_mask, test_mask = build_split_real_only(
        df, args.test_size, args.seed
    )
    print("Train files:", len(train_files), "Test files:", len(test_files))
    print(
        "Train rows -> real/aug:",
        int((train_mask & (df.is_augmented == 0)).sum()),
        int((train_mask & (df.is_augmented == 1)).sum()),
    )
    print(
        "Test  rows -> real/aug:",
        int((test_mask & (df.is_augmented == 0)).sum()),
        int((test_mask & (df.is_augmented == 1)).sum()),  # should be 0
    )

    # Feature columns: numeric-only, drop metadata
    feat_cols = select_feature_columns(df)

    # Matrices
    Xtr = df.loc[train_mask, feat_cols].to_numpy(dtype=float)
    Xte = df.loc[test_mask, feat_cols].to_numpy(dtype=float)
    ytr = df.loc[train_mask, LABEL_COL].astype(int).to_numpy()
    yte = df.loc[test_mask, LABEL_COL].astype(int).to_numpy()

    # Sanity checks on class presence
    print("Train class counts:", np.bincount(ytr))
    print("Test  class counts:", np.bincount(yte))
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        raise RuntimeError(
            "Split produced a single-class set. Ensure both classes exist across files in your dataset."
        )

    # Optional scaling (kept for future models; XGB is fine without)
    scaler = None
    if args.scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    # Model
    clf = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        base_score=0.5,
        n_jobs=8,
        random_state=args.seed,
    )
    clf.fit(Xtr, ytr)

    # Evaluate
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(Xte)[:, 1]
    else:
        p = clf.predict(Xte)
        if p.ndim > 1:
            p = p.ravel()
    pred = (p >= 0.5).astype(int)

    try:
        auc = roc_auc_score(yte, p)
        print(f"AUC: {auc:.3f}")
    except ValueError:
        print("AUC: n/a (single-class test set)")
    print(f"F1 : {f1_score(yte, pred, zero_division=0):.3f}")
    print(f"Acc: {accuracy_score(yte, pred):.3f}")
    print(f"Features used: {len(feat_cols)}")

    # Save bundle
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    bundle = {"model": clf, "features": feat_cols}
    if args.scale and scaler is not None:
        bundle["scaler"] = scaler
    joblib.dump(bundle, args.model_out)
    print(f"Saved model → {args.model_out}")

    # Save splits for reproducibility
    if args.save_splits:
        os.makedirs("splits", exist_ok=True)
        pd.Series(sorted(train_files)).to_csv(
            "splits/train_files.csv", index=False, header=False
        )
        pd.Series(sorted(test_files)).to_csv(
            "splits/test_files.csv", index=False, header=False
        )
        print("Saved splits to splits/train_files.csv and splits/test_files.csv")


if __name__ == "__main__":
    main()
