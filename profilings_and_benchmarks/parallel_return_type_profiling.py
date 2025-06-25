import sys

from time import time

import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=10, random_state=1)
    X = X.astype(np.float32)

    return_type = sys.argv[1]
    assert return_type in ["list", "generator", "no-oob"]
    n_estimators = 100

    if return_type == "no-oob":
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=False,
            random_state=42,
            n_jobs=1,
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            return_type=return_type,
            random_state=42,
            n_jobs=1,
        )

    start_time = time()
    rf.fit(X, y)
    run_time = time() - start_time
    if return_type!= "no-oob":
        print(rf.unbiased_feature_importances_)
    print(f"Fitting took : {run_time:.3f}s")
