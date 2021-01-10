import lightgbm as lgb
from helpers import preprocess
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    r"data\forestfires.csv",
    parse_dates=[],
    index_col=[],
)
X, y = preprocess(df, encode=False, categorize=True, preran=False)
d = lgb.Dataset(X, y, silent=True)

params = {
    "objective": "rmse",
    "metric": "rmse",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.004090619790710353,
    "feature_pre_filter": False,
    "lambda_l1": 1.846285949957257e-08,
    "lambda_l2": 9.447939398508748,
    "num_leaves": 31,
    "feature_fraction": 0.8999999999999999,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
    "num_boost_round": 434,
}

model = lgb.train(
    params,
    d,
)

Path("models").mkdir(exist_ok=True)
model.save_model(
    "models/model.pkl",
    num_iteration=params["num_boost_round"],
    importance_type="gain",
)
