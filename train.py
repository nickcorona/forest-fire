import lightgbm as lgb
from helpers import preprocess
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"data\forestfires.csv",
    parse_dates=[],
    index_col=[],
)
X, y = preprocess(df, encode=False, categorize=True, preran=False)
X = X.drop("rain", axis=1)
d = lgb.Dataset(X, y, silent=True)

# rmse: 98.23778701225294
params = {
    "objective": "rmse",
    "metric": "rmse",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.004090619790710353,
    "feature_pre_filter": False,
    "lambda_l1": 1.1286303015208678e-08,
    "lambda_l2": 9.694463982242468,
    "num_leaves": 11,
    "feature_fraction": 0.8999999999999999,
    "bagging_fraction": 0.9949731326113778,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "num_boost_round": 443,
}

model = lgb.train(
    params,
    d,
)

Path("figures").mkdir(exist_ok=True)
lgb.plot_importance(model, grid=False)
plt.savefig("figures/feature_importange.png")

Path("models").mkdir(exist_ok=True)
model.save_model(
    "models/model.pkl",
    num_iteration=params["num_boost_round"],
    importance_type="gain",
)
