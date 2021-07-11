#!/bin/env python3

import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def dataset(infile, stars_min=None, stars_max=None):
    data = pd.read_csv(infile)
    data = data.drop(columns=["index", "id", "owner", "name"])
    data = data.drop(columns=["watches", "forks"])
    if stars_min is not None:
        data = data.loc[data["stars"] >= stars_min]
    else:
        stars_min = data["stars"].min()
    if stars_max is not None:
        data = data.loc[data["stars"] < stars_max]
    else:
        stars_max = data["stars"].max()
    data = data.fillna(
        value={"description": "", "homepage": "", "language": "", "license": ""}
    )

    le = LabelEncoder()
    for c in [
        "owner_type",
        "fork",
        "has_issues",
        "has_projects",
        "has_downloads",
        "has_wiki",
        "has_pages",
        "language",
        "archived",
        "license",
    ]:
        data[c] = le.fit_transform(data[c])

    for c in ["description", "homepage"]:
        data[[c]] = data[[c]].applymap(lambda x: len(x))

    for c in ["created_at", "updated_at"]:
        data[[c]] = data[[c]].applymap(
            lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()
        )

    for c in [
        "description",
        "created_at",
        "updated_at",
        "homepage",
        "size",
        "readme_len",
    ]:
        data[c] -= data[c].values.mean()

    x = data.drop(columns=["stars"])
    y = data["stars"]

    stars = np.arange(stars_max - stars_min + 1)
    freq = np.zeros((stars.shape[0]))
    y_uniq, count = np.unique(y, return_counts=True)
    freq[y_uniq - stars_min] += count
    f = lambda x, a, b, c, d, e: np.exp(-a * x) * b + np.exp(-c * x) * d + e
    params, _ = curve_fit(f, stars, freq, p0=[1, 10, 1, 20, 1])
    w = y / f(y, *params)

    print(params)
    #  plt.scatter(stars, freq)
    #  plt.plot(stars, f(stars, *params))
    #  plt.show()
    #  exit()

    return (x, y, w)


def split(x, y, w, split_seed=None):
    return train_test_split(x, y, w, test_size=0.2, random_state=split_seed)


def fit(x, y, w, rf_seed=None, n_estimators=100):
    rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=rf_seed)
    rfr.fit(x, y, w)
    return rfr


def eval(rfr, x, y, w):
    score = rfr.score(x, y, w)

    importances = pd.DataFrame(
        {"importance": rfr.feature_importances_}, index=x.columns
    )
    importances = importances.sort_values("importance", ascending=False)

    return (score, importances)


def search(x, y, w):
    result = pd.DataFrame(
        columns=("split_seed", "n_estimators", "rf_seed", "score", "importances")
    )

    for split_seed in range(50):
        for n_estimators in [10, 30, 100, 300, 1000]:
            for rf_seed in range(5):
                print(
                    f"Fit split_seed={split_seed}, n_estimators={n_estimators}, rf_seed={rf_seed}: ",
                    end="",
                    flush=True,
                )

                x_train, x_test, y_train, y_test, w_train, w_test = split(
                    x, y, w, split_seed=split_seed
                )
                rfr = fit(
                    x_train,
                    y_train,
                    w_train,
                    n_estimators=n_estimators,
                    rf_seed=rf_seed,
                )
                score, importances = eval(rfr, x_test, y_test, w_test)

                print(f"{score:.4f}")
                result = result.append(
                    {
                        "split_seed": split_seed,
                        "n_estimators": n_estimators,
                        "rf_seed": rf_seed,
                        "score": score,
                        "importances": importances,
                    },
                    ignore_index=True,
                )
    print()

    n = 20
    print(f"Top {n} result:")
    for c in ["split_seed", "n_estimators", "rf_seed"]:
        result[c] = result[c].astype("int64")

    result = result.sort_values("score", ascending=False, ignore_index=True)
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.max_rows = None
    print(result.drop(columns=["importances"]).loc[:n])
    print()

    split_seed = result.loc[0, "split_seed"]
    n_estimators = result.loc[0, "n_estimators"]
    rf_seed = result.loc[0, "rf_seed"]
    score = result.loc[0, "score"]
    importances = result.loc[0, "importances"]
    print(
        f"Param split_seed={split_seed}, n_estimators={n_estimators}, rf_seed={rf_seed}:"
    )
    print(f"Score: {score:.4f}")
    print(importances)


def show(x, y, w, split_seed, n_estimators, rf_seed):
    x_train, x_test, y_train, y_test, w_train, w_test = split(
        x, y, w, split_seed=split_seed
    )

    rfr = fit(x_train, y_train, w_train, n_estimators=n_estimators, rf_seed=rf_seed)

    score, importances = eval(rfr, x_test, y_test, w_test)

    pd.options.display.float_format = "{:.4f}".format
    print(f"Score: {score:.4}")
    print(importances)

    y_pred = rfr.predict(x_test)
    limit = max(y_test.max(), y_pred.max())
    plt.scatter(y_test, y_pred)
    plt.plot([0, limit], [0, limit])
    plt.show()


def main(infile):
    x, y, w = dataset(infile, stars_min=1)
    #  search(x, y, w)
    show(x, y, w, 39, 100, 3)


if __name__ == "__main__":
    main(sys.argv[1])
