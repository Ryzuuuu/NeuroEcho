"""Track A: explainable biomarker model on UCI Parkinson's data."""
import sys
from pathlib import Path

# Add ml/ to sys.path for config import
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend: the plot is only written to disk
import matplotlib.pyplot as plt  # noqa: E402
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from ml_config import RAW_DIR, MODELS_DIR, RANDOM_STATE


def main():
    df = pd.read_csv(RAW_DIR / "parkinsons.data")
    # Train using only 3 gender-neutral micro-perturbation features.
    # We explicitly exclude Fo/Fhi/Flo because absolute pitch is heavily gender-biased,
    # and the UCI dataset mistakenly learned that 'deep voices = sick'.
    features = ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR"]
    X = df[features]
    y = df["status"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Balance classes (there are ~3x more sick patients than healthy)
    # scale_pos_weight = negative_samples / positive_samples
    num_neg = (y_tr == 0).sum()
    num_pos = (y_tr == 1).sum()
    scale_weight = num_neg / num_pos

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_weight,
            eval_metric="logloss", random_state=RANDOM_STATE)),
    ])

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {auc.mean():.3f} ± {auc.std():.3f}")

    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(classification_report(y_te, pred))
    print("Test ROC-AUC:", round(roc_auc_score(y_te, proba), 3))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Explainability plot
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    X_te_scaled = model.named_steps["scaler"].transform(X_te)
    shap_values = explainer.shap_values(X_te_scaled)
    shap.summary_plot(shap_values, X_te, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150)
    print("Saved SHAP plot -> models/shap_summary.png")

    joblib.dump(model, MODELS_DIR / "biomarker_model.pkl")
    print("Saved model -> models/biomarker_model.pkl")


if __name__ == "__main__":
    main()
