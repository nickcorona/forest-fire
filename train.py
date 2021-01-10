import lightgbm as lgb
from helpers import preprocess
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    r"data\training.csv",
    parse_dates=[],
    index_col=[],
)
X, y = preprocess(df, encode=True, categorize=True, preran=True)
d = lgb.Dataset(X, y, silent=True)

BEST_ITERATION = 144
params = {
    "objective": "tweedie",
    "metric": "rmse",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.03625071824840758,
    "feature_pre_filter": False,
    "lambda_l1": 1.2978470049705244e-05,
    "lambda_l2": 8.83472293209571e-05,
    "num_leaves": 30,
    "feature_fraction": 0.652,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
    # "categorical_column": [2, 5, 6, 7, 8, 11, 15, 16],
}

model = lgb.train(
    params,
    d,
    num_boost_round=BEST_ITERATION,
)

Path("models").mkdir(exist_ok=True)
model.save_model("trained_model.pickle", num_iteration=BEST_ITERATION, importance_type="gain")
