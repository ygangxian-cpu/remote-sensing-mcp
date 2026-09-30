from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

import run_parent_bias_correction_v4_pilot as pilot

# Faster mechanism pilot: the two Landsat evaluation dates are not present in
# the correction corpus at all. Cross-validation remains blocked by whole date.
pilot.TRAIN_DATES = [
    "2019-09-16",
    "2019-09-17",
    "2019-09-19",
    "2019-09-20",
    "2019-09-21",
    "2019-09-22",
    "2019-09-23",
    "2019-09-25",
    "2019-09-27",
    "2019-09-29",
]
pilot.PARENT_RF = {
    "n_estimators": 300,
    "min_samples_leaf": 12,
    "max_features": 0.8,
    "n_jobs": -1,
    "random_state": 42,
}
pilot.LOCAL_RF = {
    "n_estimators": 250,
    "min_samples_leaf": 25,
    "max_features": 0.8,
    "n_jobs": -1,
    "random_state": 42,
}


def grouped_date_cv(df, features, target, params):
    clean = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=features + [target, "date"])
        .reset_index(drop=True)
    )
    X = clean[features].to_numpy(dtype="float32")
    y = clean[target].to_numpy(dtype="float32")
    groups = clean["date"].astype(str).to_numpy()
    unique = np.unique(groups)
    k = min(5, unique.size)
    if k < 2:
        raise RuntimeError("Need at least two whole dates for grouped CV")

    splitter = GroupKFold(n_splits=k)
    oof = np.full(len(clean), np.nan, dtype="float32")
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        model = RandomForestRegressor(**params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te]).astype("float32")
        held_dates = sorted(set(groups[te].tolist()))
        fold_rows.append({
            "date": ",".join(held_dates),
            "fold": fold,
            **pilot.score(y[te], oof[te]),
        })

    full = RandomForestRegressor(**params)
    full.fit(X, y)
    return clean, oof, full, pilot.score(y, oof), fold_rows


pilot.fit_date_blocked = grouped_date_cv


if __name__ == "__main__":
    pilot.main()
