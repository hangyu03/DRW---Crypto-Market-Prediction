# xgb_mlp_ensemble_with_outlier_adjusted – Strategy Overview

Below you will find concise English explanations of each key strategy implemented in `xgb_mlp_ensemble_with_outlier_adjusted.ipynb`. Copy‑paste or extend these sections as needed for your project documentation.

## 1  Feature Engineering
The notebook begins by enriching the raw limit‑order‑book feed with domain‑specific signals. It derives price‑depth interactions (e.g., `bid_qty × ask_qty`), logarithmic volume measures, buy‑sell pressure ratios, and a configurable set of lag/lead features that capture short‑term temporal structure. All missing values introduced by these transformations are imputed with column means to maintain a dense input matrix. This engineered feature space gives subsequent models both instantaneous and lagged context on market microstructure behaviour.

## 2  Time‑Decay Sample Weighting
Because crypto‑market dynamics drift over time, the notebook assigns exponentially decaying weights to samples so that newer observations influence the loss function more than stale ones. The decay constant is user‑controlled; higher decay accelerates forgetting. These weights are fed consistently into every cross‑validation fold and final refit, ensuring that model parameters are optimised for the most recent regime without discarding historic data entirely.

## 3  Outlier‑Aware Re‑weighting Strategies
A light RandomForest first fits the labelled training set and computes residuals; the top 0.1 % largest residuals are flagged as candidate outliers. Four mutually exclusive strategies then adjust the sample weights before XGBoost training:
- **reduce** (default): down‑scales outlier weights by a random factor in \[0.2, 0.8\], lessening—but not eliminating—their impact.
- **remove**: sets outlier weights to 0, effectively dropping those rows from the loss calculation.
- **double**: up‑weights outliers by 2×, useful when extreme moves are considered especially informative.
- **none**: keeps original time‑decay weights, treating all samples equally.
Each strategy is evaluated via time‑series cross‑validation, and the one delivering the best out‑of‑fold correlation is selected automatically for the final model.

## 4  XGBoost Training
The gradient‑boosted tree model is trained with the GPU‑accelerated `hist` algorithm and `gpu_predictor` for fast iteration. Hyper‑parameters such as learning rate, depth, and number of trees are tuned either manually or through Optuna (if enabled). The model inherits both the time‑decay weights and the outlier‑adjusted weights, allowing it to focus on recent, reliable patterns while ignoring uninformative noise.

## 5  MLP Training
A compact PyTorch MLP provides complementary non‑linear capacity. After `StandardScaler` normalisation, Gaussian noise (0.5 %) is injected to improve generalisation. The architecture defaults to `[input → 256 → 64 → 1]` with ReLU activations, dropout 0.6, and early‑stopping checkpoints based on validation correlation. Optional k‑fold training is available to stabilise performance on non‑stationary data.

## 6  Model Ensembling
The notebook finishes by linearly blending the predictions from the best XGBoost and the MLP. The default mix is 0.9 × XGB + 0.1 × MLP, reflecting the typically stronger single‑model performance of boosted trees while still harvesting incremental signal from the neural network. Both out‑of‑fold and test‑set predictions are saved to CSV so they can be uploaded directly to the Kaggle DRW Crypto Market Prediction leaderboard.

---
Feel free to adjust decay rates, outlier fractions, network depth, or ensemble weights to suit your own experiments. Pull requests welcome!

