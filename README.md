# Predicting Smartphone Addiction — Kaggle Playground Series S6E8

Feature engineering + a 7-model ensemble pipeline for the Kaggle competition
[Playground Series S6E8: Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8),
evaluated on ROC AUC.

The pipeline is split into independent, sequentially-numbered Jupyter notebooks so each model can be run,
tuned, or re-run on its own without re-executing the others.

## Repo structure

```
01_feature_engineering.ipynb   # builds the shared feature set, saves train_features.csv / test_features.csv
02_model_histgb.ipynb          # sklearn HistGradientBoostingClassifier   (no extra installs)
03_model_lightgbm.ipynb        # LightGBM
04_model_xgboost.ipynb         # XGBoost
05_model_catboost.ipynb        # CatBoost
06_model_random_forest.ipynb   # Random Forest
07_model_extra_trees.ipynb     # Extra Trees
08_model_neural_net.ipynb      # PyTorch MLP with categorical embeddings
09_ensemble_blend.ipynb        # blends whichever model outputs exist, writes submission.csv
```

Each model notebook (`02`–`08`) trains with 5-fold `StratifiedKFold` cross-validation, prints per-fold and
overall out-of-fold (OOF) AUC, and saves two files:

- `oof_<model>.csv` — out-of-fold predictions on the training set (`id`, `oof_pred`)
- `test_pred_<model>.csv` — predictions on the test set (`id`, `test_pred`)

`09_ensemble_blend.ipynb` auto-discovers whatever `oof_*.csv` / `test_pred_*.csv` files are present, so you
don't need to run every model — it blends whichever ones you've generated.

## Data

Not included in this repo (competition data, subject to Kaggle's terms). Download from the
[competition data page](https://www.kaggle.com/competitions/playground-series-s6e8/data) — either via the
web UI or the Kaggle CLI:

```bash
kaggle competitions download -c playground-series-s6e8
unzip playground-series-s6e8.zip -d data/
```

You should end up with `train.csv`, `test.csv`, and `sample_submission.csv`. Point `DATA_DIR` at the top of
each notebook to wherever you put them.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn lightgbm xgboost catboost torch jupyter
```

(Notebooks `03`–`05` and `08` also have a `!pip install` cell at the top if you'd rather install as you go.)

## How to run

1. **Run `01_feature_engineering.ipynb` first.** It reads the raw CSVs, builds the engineered feature set,
   and saves `train_features.csv` / `test_features.csv`. Every other notebook loads those two files rather
   than re-deriving features, so all models train on identical columns and — via a fixed random seed —
   identical CV fold splits.
2. **Run any subset of `02`–`08`.** Each is fully self-contained; run them in any order, on any machine,
   in parallel across separate kernels, or skip ones you don't care about.
3. **Run `09_ensemble_blend.ipynb` last.** It merges all available OOF predictions on `id`, blends them
   with a greedy forward-selection search (adds whichever model most improves OOF AUC at each step,
   allowing repeats, until nothing helps), and writes the final `submission.csv`.

## What the feature engineering found

- `daily_screen_time_hours`, `social_media_hours`, and `weekend_screen_time` are by far the strongest
  individual predictors. There's a sharp, sigmoid-shaped jump in addiction rate between roughly 6 and 9.5
  daily screen-time hours (~23% → ~99%+).
- `gender`, `stress_level`, and `academic_work_impact` carry essentially **no** signal — the screen-time
  threshold curve is identical across every group of each. They're kept as low-cost categorical inputs
  since boosted trees can occasionally pick up minor conditional splits, but don't expect much from them.
- Missingness is not informative (missing-value indicators have ~zero correlation with the target).
- The 18 engineered features (ratios like `notif_per_hour`, `sm_ratio`; sums like `screen_plus_weekend`;
  and pairwise interaction products) gave a measurable lift over raw features alone — around
  **0.955 → 0.963 OOF AUC** with sklearn's `HistGradientBoostingClassifier`.
- A purely additive/linear composite of the correlated proxy variables (screen time, social media, sleep,
  etc.) tops out well below what boosted trees achieve, meaning the true signal involves real feature
  interactions rather than a simple weighted sum.

## Results

| Model | Notebook | Notes |
|---|---|---|
| HistGradientBoosting | `02` | Baseline, no extra installs — OOF AUC ≈ 0.963 |
| LightGBM | `03` | Native categorical support |
| XGBoost | `04` | `enable_categorical=True`, `tree_method='hist'` |
| CatBoost | `05` | Native categorical support, no encoding needed |
| Random Forest | `06` | Bagging — lower solo AUC, but decorrelated errors help the blend |
| Extra Trees | `07` | Bagging with randomized split thresholds |
| PyTorch NN | `08` | Embeddings for categoricals + dense layers — different error profile than trees |
| **Blend** | `09` | Greedy forward-selection blend of whichever models were run |


## Further improvement ideas

- Hyperparameter search (Optuna) on LightGBM/XGBoost/CatBoost — the configs shipped here are reasonable
  defaults, not tuned. `num_leaves`, `max_depth`, `min_child_samples`/`min_child_weight`, and `reg_lambda`
  are the highest-leverage knobs.
- Multiple seeds per model, averaged, before blending — cheap variance reduction.
- Stacking (a meta-model on the OOF columns) instead of the greedy blend.
- Out-of-fold target/frequency encoding for the categorical columns.
- Pseudo-labeling high-confidence test predictions back into training.
- A deeper/wider or better-tuned NN architecture in `08`.

## License

Data is subject to the Kaggle competition's own terms — see the
[competition rules](https://www.kaggle.com/competitions/playground-series-s6e8/rules). Code in this repo:
MIT (adjust as you prefer).
