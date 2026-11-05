# NeuroEcho — ML

## Setup
1. `pip install -r requirements.txt`
2. `data/raw/parkinsons.data` (UCI Parkinson's dataset, CC BY 4.0) is included. Download it again from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons) if needed.
3. (Optional, for the CNN) add WAV audio to `data/raw/audio/pd/` and `data/raw/audio/healthy/`.

## Train
```bash
python src/train_biomarker.py   # explainable model + SHAP plot
python src/train_cnn.py         # deep audio model (needs audio from step 3)
```

## Output
- `models/biomarker_model.pkl` — tracked, small enough to ship with the repo
- `models/shap_summary.png` — feature-importance plot shown in the main README
- `models/cnn_model.pt` — only created when the CNN is trained (git-ignored)

Set `MODEL_DIR` to read or write models in a different directory.

## Tests
```bash
pytest ml/tests
```

## Metrics we report
ROC-AUC · Sensitivity · Specificity · F1 · Confusion matrix · SHAP importance
