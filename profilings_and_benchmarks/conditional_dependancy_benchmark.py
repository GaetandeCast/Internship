import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state


def create_dataset_with_conditional_dependances(n_samples, random_state=None):
    rng = check_random_state(random_state)

    x_info_high_card = rng.choice(20, n_samples).reshape((n_samples, 1))
    x_noise_high_card = rng.choice(20, n_samples).reshape((n_samples, 1))

    x_info_low_card = x_info_high_card // 5
    x_noise_low_card = x_noise_high_card // 5

    rho = 0.2
    y = np.random.binomial(n=1, p=0.5 + rho * (2 * x_info_low_card >= 2 - 1)).ravel()
    X = np.concatenate(
        (x_info_high_card, x_info_low_card, x_noise_high_card, x_noise_low_card), axis=1
    )
    return (X, y)


n_samples = 10000
n_experiments = 10

importances_ufi = pd.DataFrame(
    columns=["info_high", "info_low", "noise_high", "noise_low"]
)
importances_mdi_oob = pd.DataFrame(
    columns=["info_high", "info_low", "noise_high", "noise_low"]
)

for i in range(n_experiments):
    X, y = create_dataset_with_conditional_dependances(
        n_samples=n_samples, random_state=i
    )
    rf = RandomForestClassifier(random_state=i)
    rf.fit(X, y)
    importances_ufi.loc[i] = (
        rf._compute_unbiased_feature_importance_and_oob_predictions(X, y, method="ufi")[
            0
        ]
    )
    importances_mdi_oob.loc[i] = (
        rf._compute_unbiased_feature_importance_and_oob_predictions(
            X, y, method="mdi_oob"
        )[0]
    )

ax = importances_ufi.plot.box(vert=False, whis=10)
ax.set_title("ufi")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Mean Decrease Impurity")
ax.figure.tight_layout()
plt.show()

ax = importances_mdi_oob.plot.box(vert=False, whis=10)
ax.set_title("mdi_oob")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Mean Decrease Impurity")
ax.figure.tight_layout()
plt.show()
