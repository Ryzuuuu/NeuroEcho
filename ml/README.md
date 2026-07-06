# NeuroEcho — ML

## Setup
1. Download `parkinsons.data` from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons) → place in `data/raw/`.
2. (Optional) Add WAV audio to `data/raw/audio/pd/` and `data/raw/audio/healthy/`.
3. `pip install -r requirements.txt`

## Train
```bash
python src/train_biomarker.py   # explainable model + SHAP plot
python src/train_cnn.py         # deep audio model
```

## Output
- `models/biomarker_model.pkl`
- `models/cnn_model.pt`
- `models/shap_summary.png` ← use in your pitch deck!

## Metrics we report
ROC-AUC · Sensitivity · Specificity · F1 · Confusion matrix · SHAP importance
