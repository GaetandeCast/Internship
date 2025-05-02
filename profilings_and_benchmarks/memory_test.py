# import UFI
from time import time

import numpy as np
from memory_profiler import memory_usage

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=10)
    X, y = np.array(X), np.array(y)

    n_trains = 1
    n_estimators = 100
    results = {}
    n_jobs = 1

    def run_base_fit():
        for i in range(n_trains):
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=5, random_state=i, n_jobs=n_jobs
            )
            rf.fit(X, y)

    def run_base_fit_then_mdi():
        for i in range(n_trains):
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=5, random_state=i, n_jobs=n_jobs
            )
            rf.fit(X, y)
            _ = rf.feature_importances_

    def run_fit_with_oob():
        for i in range(n_trains):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=i,
                oob_score=True,
                n_jobs=n_jobs,
            )
            rf.fit(X, y)
            _ = rf.oob_ufi_feature_importance_

    def run_base_fit_then_ufi():
        for i in range(n_trains):
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=5, random_state=i, n_jobs=n_jobs
            )
            rf.fit(X, y)
            rf._compute_unbiased_feature_importance_and_oob_predictions(X, y)

    # def run_base_fit_then_mdi_oob():
    #     for i in range(n_trains):
    #         rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=i)
    #         rf.fit(X, y)
    #         rf._compute_unbiased_feature_importance_oob(X, y, method="mdi_oob")

    # def run_ufi_paper():
    #     for i in range(n_trains):
    #         rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=i)
    #         rf.fit(X, y)
    #         UFI.cls(rf, X, y, mix="mixed")

    def run_permutation_importance():
        for i in range(n_trains):
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=5, random_state=i, n_jobs=n_jobs
            )
            rf.fit(X, y)
            _ = permutation_importance(rf, X, y, random_state=i)

    def profile_block(label, func):
        start = time()
        mem = memory_usage(func)
        end = time()
        results[label] = {
            "runtime (s)": round(end - start, 3),
            "max_memory (MB)": round(max(mem) - min(mem), 3),
        }

    profile_block("base fit", run_base_fit)
    profile_block("base fit then MDI calcs", run_base_fit_then_mdi)
    profile_block("fit with oob", run_fit_with_oob)
    profile_block("base fit then ufi", run_base_fit_then_ufi)
    # profile_block("base fit then mdi_oob", run_base_fit_then_mdi_oob)
    # profile_block("base fit then UFI-paper", run_ufi_paper)
    profile_block("base fit then permutation importance", run_permutation_importance)

    for label, metrics in results.items():
        print(
            f"{label:<50} | Runtime: {metrics['runtime (s)']:.3f} s | Max Memory: {metrics['max_memory (MB)']:.3f} MB"
        )
