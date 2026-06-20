"""Stage 5: ML Classification — Random Forest on physics features."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

PHYSICS_FEATURES = [
    "depth_score",
    "shape_score",
    "nosec_score",
    "oddeven_score",
    "achrom_score",
    "atmos_score",
    "det_quality",
    "oot_std",
    "skewness",
]

MODEL_PATH = Path(__file__).parent.parent / "models" / "rf_planet_classifier.pkl"


def _get_col(row, *candidates, default=0.5):
    for c in candidates:
        if c in row.index:
            val = row[c]
            if pd.notna(val):
                return float(val)
    return default


def compute_detection_quality(row: pd.Series) -> float:
    bls_sde = _get_col(row, "sde", "bls_sde", "bls_sde_x", "bls_sde_y", default=0.0)
    tls_sde = _get_col(row, "tls_sde", "tls_sde_x", "tls_sde_y", default=0.0)
    return float(min((bls_sde + tls_sde) / 30.0, 1.0))


def apply_hard_vetoes(row: pd.Series) -> tuple[bool, str, float]:
    depth_score = _get_col(row, "depth_score", "depth_consistency_score", default=0.5)
    shape_score = _get_col(row, "shape_score", default=0.5)
    nosec_score = _get_col(row, "nosec_score", "secondary_eclipse_score", default=0.5)
    oddeven_score = _get_col(row, "oddeven_score", "odd_even_score", default=0.5)
    achrom_score = _get_col(row, "achrom_score", "achromaticity_score", default=0.5)
    atmos_score = _get_col(row, "atmos_score", "atmosphere_proxy_score", default=0.5)
    centroid_score = _get_col(row, "centroid_score", "centroid_check_score", default=0.5)
    ellipsoidal_score = _get_col(row, "ellipsoidal_score", default=1.0)

    if depth_score < 0.1:
        return True, "FALSE_POSITIVE", 0.05
    if depth_score < 0.4 and oddeven_score < 0.3:
        return True, "ECLIPSING_BINARY", 0.15
    if nosec_score < 0.2:
        return True, "ECLIPSING_BINARY", 0.10
    if oddeven_score < 0.2 and nosec_score < 0.6:
        return True, "ECLIPSING_BINARY", 0.20
    if shape_score < 0.3 and centroid_score < 0.3:
        return True, "BLEND", 0.15
    if achrom_score < 0.25 and shape_score < 0.4 and atmos_score < 0.4:
        return True, "STELLAR_ACTIVITY", 0.20
    # Ellipsoidal variation veto
    if ellipsoidal_score < 0.3:
        return True, "ECLIPSING_BINARY", 0.25
    return False, "", 0.0


def extract_ml_features(row: pd.Series) -> np.ndarray:
    det_quality = compute_detection_quality(row)
    features = {
        "depth_score": _get_col(row, "depth_score", "depth_consistency_score", default=0.5),
        "shape_score": _get_col(row, "shape_score", default=0.5),
        "nosec_score": _get_col(row, "nosec_score", "secondary_eclipse_score", default=0.5),
        "oddeven_score": _get_col(row, "oddeven_score", "odd_even_score", default=0.5),
        "achrom_score": _get_col(row, "achrom_score", "achromaticity_score", default=0.5),
        "atmos_score": _get_col(row, "atmos_score", "atmosphere_proxy_score", default=0.5),
        "det_quality": det_quality,
        "oot_std": _get_col(row, "oot_std", default=0.01),
        "skewness": _get_col(row, "skewness", default=0.0),
    }
    return np.array([features[f] for f in PHYSICS_FEATURES]).reshape(1, -1)


def build_training_data(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    tic_list = []
    CONFIRMED_PLANETS = {
        98796344, 452866790, 307809773, 188589164, 237913194, 149603524,
        183374187,  # Hot Jupiter
    }
    KNOWN_EBS = {
        33419790, 5674169, 31690845, 11046410, 10891640, 24935204, 436869712,
        1616408, 2438442, 466376085, 470710327,
    }

    for _, row in features_df.iterrows():
        tic_id = int(row["tic_id"])
        vetoed, veto_cls, _ = apply_hard_vetoes(row)
        if vetoed:
            X_list.append(extract_ml_features(row).flatten())
            y_list.append(0)
            tic_list.append(tic_id)
            continue
        if tic_id in CONFIRMED_PLANETS:
            X_list.append(extract_ml_features(row).flatten())
            y_list.append(1)
            tic_list.append(tic_id)
            continue
        if tic_id in KNOWN_EBS:
            X_list.append(extract_ml_features(row).flatten())
            y_list.append(0)
            tic_list.append(tic_id)
            continue

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\n  📊 Training Data:")
    print(f"     Planets (label=1): {sum(y == 1)}")
    print(f"     Non-planets (label=0): {sum(y == 0)}")
    print(f"     Total: {len(y)}")
    print(f"     TICs: {tic_list}")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    if len(np.unique(y)) < 2:
        print("  ⚠️ Only one class. Using dummy model.")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    n_planets = sum(y == 1)
    n_non = sum(y == 0)
    if n_planets < n_non:
        class_weight = {0: 1.0, 1: n_non / max(n_planets, 1)}
    elif n_non < n_planets:
        class_weight = {0: n_planets / max(n_non, 1), 1: 1.0}
    else:
        class_weight = "balanced"

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        min_samples_leaf=3, max_features="sqrt",
        class_weight=class_weight, random_state=42, n_jobs=-1,
    )

    if len(y) >= 10:
        cv = StratifiedKFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"\n  🔬 CV AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X, y)
    print(f"\n  📈 Feature Importance:")
    for feat, imp in zip(PHYSICS_FEATURES, model.feature_importances_):
        bar = "█" * int(imp * 30)
        print(f"     {feat:<20} {imp:.3f} {bar}")
    return model


def save_model(model: RandomForestClassifier, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  💾 Model saved to {path}")


def load_model(path: Path) -> RandomForestClassifier | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_ml_classification(candidates: pd.DataFrame, config: dict, force_retrain: bool = False) -> pd.DataFrame:
    logger.info("Stage 5: ML Classification on %d candidates", len(candidates))
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame()

    features_path = Path(config['paths']['processed']) / 'features.csv'
    if not features_path.exists():
        logger.error("Features not found: %s", features_path)
        return candidates

    features_df = pd.read_csv(features_path)
    print(f"\n{'='*60}")
    print(f"  STAGE 5: ML CLASSIFICATION")
    print(f"{'='*60}")

    merged = candidates.merge(features_df, on='tic_id', how='left', suffixes=('', '_feat'))

    # Step 1: Apply hard vetoes
    veto_results = []
    for _, row in merged.iterrows():
        vetoed, cls, conf = apply_hard_vetoes(row)
        veto_results.append({"vetoed": vetoed, "veto_cls": cls, "veto_conf": conf})

    veto_df = pd.DataFrame(veto_results)
    merged = pd.concat([merged.reset_index(drop=True), veto_df], axis=1)

    # Step 2: Train/load model
    model = None
    if not force_retrain:
        model = load_model(MODEL_PATH)

    if model is None:
        print(f"\n  🔄 Training new model...")
        X_train, y_train = build_training_data(features_df)
        if len(X_train) > 0:
            model = train_model(X_train, y_train)
            save_model(model, MODEL_PATH)
        else:
            print("  ⚠️ No training data. Using fallback.")

    # Step 3: Predict ML scores
    ml_scores = []
    for _, row in merged.iterrows():
        if row["vetoed"]:
            ml_scores.append(0.0)
        elif model is not None:
            X = extract_ml_features(row)
            proba = model.predict_proba(X)[0]
            ml_scores.append(float(proba[1]))
        else:
            ml_scores.append(0.5)

    merged['ml_score'] = ml_scores

    # Step 4: Final classification WITH VETOES
    final_classifications = []
    final_confidences = []

    for _, row in merged.iterrows():
        # CHECK VETO FIRST
        vetoed, veto_cls, veto_conf = apply_hard_vetoes(row)
        if vetoed:
            final_classifications.append(veto_cls)
            final_confidences.append(veto_conf)
            continue

        ml = row["ml_score"]
        det_quality = compute_detection_quality(row)

        if ml >= 0.60 and det_quality >= 0.25:
            final_classifications.append("TRANSIT")
            final_confidences.append(ml)
        elif ml >= 0.35:
            final_classifications.append("UNCERTAIN")
            final_confidences.append(ml)
        else:
            final_classifications.append("REJECT")
            final_confidences.append(ml)

    merged['classification'] = final_classifications
    merged['ml_confidence'] = final_confidences
    merged['noise_percent'] = ((1.0 - merged['ml_confidence'].clip(0, 1)) * 100).round(1)

    print(f"\n  Classification Results:")
    for cls, count in merged['classification'].value_counts().items():
        print(f"    {cls}: {count}")
    print()
    print(f"  ML Score Distribution:")
    print(f"    Mean: {merged['ml_score'].mean():.3f}")
    print(f"    Std:  {merged['ml_score'].std():.3f}")
    print(f"    Min:  {merged['ml_score'].min():.3f}")
    print(f"    Max:  {merged['ml_score'].max():.3f}")

    out_path = Path(config['paths']['results']) / 'ml_predictions.csv'
    merged.to_csv(out_path, index=False)
    logger.info("Stage 5 complete: %s", out_path)
    return merged