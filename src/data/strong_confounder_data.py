import numpy as np
import pandas as pd
from scipy.special import expit


def generate_synthetic_data(n_samples=5000, seed=42):
    np.random.seed(seed)

    n_features = 10
    W = np.random.normal(0, 1, size=(n_samples, n_features))
    W[:, 1] = 2 + 3 * W[:, 1]
    W1 = W[:, 0]
    W2 = W[:, 1]
    W3 = W[:, 2]
    W4 = W[:, 3]
    W5 = W[:, 4]
    W6 = W[:, 5]
    # W7, W8, W9, W10 are noise variables for Y and A

    logit_A = 0.0 + 2.5 * W1 + 0.7 * W2 - 0.5 * W3 + 0.9 * W4
    prob_A = expit(logit_A)
    A = np.random.binomial(1, prob_A, size=n_samples)

    true_cate_intercept = 2.0
    true_cate_w1_interaction = 0.3

    mean_Y = (
        1.0
        + true_cate_intercept * A
        + 2.0 * W1
        + true_cate_w1_interaction * A * (W1 > 0.5)
        + 1.0 * W2
        - 0.8 * W3
        + 1.2 * W5
        - 1.5 * W6
    )

    noise_std = 1.0
    Y = mean_Y + np.random.normal(0, noise_std, size=n_samples)

    feature_cols = [f"W{i + 1}" for i in range(10)]
    data = pd.DataFrame(W, columns=feature_cols)
    data["A"] = A
    data["Y"] = Y

    return data, feature_cols
