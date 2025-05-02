import sys

# import UFI
from time import time

import numpy as np

# from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=10, random_state=1)
    X = X.astype(np.float32)

    method = sys.argv[1]
    n_trains = 1
    n_estimators = 100
    results = {}

    for i in range(n_trains):
        # with memray.Tracker(f"memray_{method}.bin", native_traces=True):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=i, n_jobs=1)

        start_time = time()
        rf.fit(X, y)
        run_time = time() - start_time
        print(f"fit : {run_time:.3f}s")
        start_time = time()
        print(
            rf._compute_unbiased_feature_importance_and_oob_predictions(
                X, y, method=method
            )[0]
        )
        run_time = time() - start_time
        print(f"method = {method} : {run_time:.3f}s")
