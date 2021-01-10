"""In this module, we ask you to define your pricing model, in Python."""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from dirty_cat import SimilarityEncoder

BEST_ITERATION = 144

# Don't forget to add them to requirements.txt before submitting.
# Feel free to create any number of other functions, constants and classes to use
# in your model (e.g., a preprocessing function).
def preprocess(X_raw, encode, categorize, preran):
    if encode:
        encode_columns = [
            "id_policy",
            "vh_make_model",
        ]
        n_prototypes = 5
        if not preran:
            enc = SimilarityEncoder(
                similarity="ngram", categories="k-means", n_prototypes=n_prototypes
            )
            enc.fit(X_raw[encode_columns].values)
            pd.to_pickle(enc, "encoders/similarity_encoder.pickle")
        else:
            enc = pd.read_pickle("encoders/similarity_encoder.pickle")
        transformed_values = enc.transform(X_raw[encode_columns].values)

        transformed_values = pd.DataFrame(transformed_values, index=X_raw.index)
        transformed_columns = []
        for col in encode_columns:
            for i in range(0, n_prototypes):
                transformed_columns.append(col + "_" + str(i))
        transformed_values.columns = transformed_columns
        X_raw = pd.concat([X_raw, transformed_values], axis=1)
        X_raw = X_raw.drop(encode_columns, axis=1)

    if categorize:
        obj_cols = X_raw.select_dtypes("object").columns
        X_raw[obj_cols] = X_raw[obj_cols].astype("category")
    return X_raw


def fit_model(X_raw, y_raw):
    """Model training function: given training data (X_raw, y_raw), train this pricing model.
    Parameters
    ----------
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.
    y_raw : a Numpy array, with the value of the claims, in the same order as contracts in X_raw.
        A one dimensional array, with values either 0 (most entries) or >0.
    Returns
    -------
    self: this instance of the fitted model. This can be anything, as long as it is compatible
        with your prediction methods.
    """
    X = preprocess(X_raw, True, True, False)
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
    }
    d = lgb.Dataset(X, y_raw)
    model = lgb.train(
        params,
        d,
        num_boost_round=BEST_ITERATION,
    )
    return model


def predict_expected_claim(model, X_raw):
    """Model prediction function: predicts the expected claim based on the pricing model.
    This functions estimates the expected claim made by a contract (typically, as the product
    of the probability of having a claim multiplied by the expected cost of a claim if it occurs),
    for each contract in the dataset X_raw.
    This is the function used in the RMSE leaderboard, and hence the output should be as close
    as possible to the expected cost of a contract.
    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs.
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.
    Returns
    -------
    avg_claims: a one-dimensional Numpy array of the same length as X_raw, with one
        expected claim per contract (in same order). These expected claims must be POSITIVE (>0).
    """
    X = preprocess(X_raw, True, True, False)
    return model.predict(X, num_iteration=BEST_ITERATION)


def predict_premium(model, X_raw):
    """Model prediction function: predicts premiums based on the pricing model.
    This function outputs the prices that will be offered to the contracts in X_raw.
    premium will typically depend on the expected claim predicted in
    predict_expected_claim, and will add some pricing strategy on top.
    This is the function used in the expected profit leaderboard. Prices output here will
    be used in competition with other models, so feel free to use a pricing strategy.
    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs.
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
            Each row is a different contract. This data has not been processed.
    Returns
    -------
    prices: a one-dimensional Numpy array of the same length as X_raw, with one
        price per contract (in same order). These prices must be POSITIVE (>0).
    """
    return predict_expected_claim(model, X_raw) * 2


def save_model(model):
    """Saves this trained model to a file.
    This is used to save the model after training, so that it can be used for prediction later.
    Do not touch this unless necessary (if you need specific features). If you do, do not
     forget to update the load_model method to be compatible.
    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs."""
    model.save_model(
        "trained_model.pickle", num_iteration=BEST_ITERATION, important_type="gain"
    )
    # with open("trained_model.pickle", "wb") as target:
    #     pickle.dump(model, target)


def load_model():
    """Load a saved trained model from the file.
    This is called by the server to evaluate your submission on hidden data.
    Only modify this *if* you modified save_model."""
    # with open("trained_model.pickle", "rb") as target:
    #     trained_model = pickle.load(target)
    model = lgb.Booster(model_file="trained_model.pickle")
    return model
