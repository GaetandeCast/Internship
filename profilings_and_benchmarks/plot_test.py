import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state


def gen_noise_cls(n, random_state=None, null=True):
    rng = check_random_state(random_state)
    x1 = rng.normal(size=n).reshape((n, 1))
    x2 = rng.choice(2, n).reshape((n, 1))
    x3 = rng.choice(4, n).reshape((n, 1))
    x4 = rng.choice(10, n).reshape((n, 1))
    x5 = rng.choice(20, n).reshape((n, 1))

    y = x2.copy()
    rho = 0.2 * (1 - null)
    flip_prob = (1 + rho) / 2
    flip_mask = rng.uniform(0, 1, size=y.shape) > flip_prob
    y[flip_mask] = 1 - y[flip_mask]

    y = y.ravel()

    X = np.concatenate((x1, x2, x3, x4, x5), axis=1)

    return [X, y]


n = 1000
n_estimators = 100
null = False
m = 10


score = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5"])
score_sk = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5"])


print("sklearn-ufi")
for i in range(m):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=5, random_state=i, oob_score=True
    )
    X, y = gen_noise_cls(n, null=null, random_state=i)
    clf.fit(X, y)
    ufi_importance = clf.oob_ufi_feature_importance_
    score_sk.loc[i] = ufi_importance

ax = score_sk.plot.box(vert=False, whis=10)
ax.set_title("sklearn-ufi")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in gini")
ax.figure.tight_layout()
ax.figure.savefig("fig1")

print("sklearn-mdi")
for i in range(m):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=5, random_state=i, oob_score=False
    )
    X, y = gen_noise_cls(n, null=null, random_state=i)
    clf.fit(X, y)
    mdi_importance = clf.feature_importances_
    score_sk.loc[i] = mdi_importance

ax = score_sk.plot.box(vert=False, whis=10)
ax.set_title("sklearn-ufi")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in gini")
ax.figure.tight_layout()
ax.figure.savefig("fig2")
