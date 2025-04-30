import numpy as np
import pandas as pd
from scipy.special import expit


def generate_no_effect_confounders_data(n_samples=5000, seed=44):
    """
    Generates synthetic data where some confounders (W3, W4) affect treatment A
    but not outcome Y.
    - W1, W2 affect both A and Y.
    - W3, W4 affect A but not Y.
    - W5 affects Y but not A.
    - W6 affects neither (pure noise).
    """
    np.random.seed(seed)

    # 1. Generate Covariates (W)
    n_features = 6
    W1 = np.random.normal(0, 1, size=n_samples)
    W2 = np.random.normal(0, 1, size=n_samples)
    W3_noise_A = np.random.normal(0, 1, size=n_samples)  # Affects A only
    W4_noise_A = np.random.binomial(1, 0.4, size=n_samples)  # Affects A only (binary)
    W5_noise_Y = np.random.normal(0, 1, size=n_samples)  # Affects Y only
    W6_pure_noise = np.random.normal(0, 1, size=n_samples)  # Affects neither

    W = np.stack([W1, W2, W3_noise_A, W4_noise_A, W5_noise_Y, W6_pure_noise], axis=1)

    # 2. Generate Treatment Assignment (A) - Depends on W1, W2, W3_noise_A, W4_noise_A
    logit_A = (
        0.0  # Intercept
        + 1.2 * W1
        - 0.9 * W2
        + 1.5 * W3_noise_A  # Confounder with no Y effect
        - 1.1 * W4_noise_A  # Binary confounder with no Y effect
        # No effect from W5_noise_Y or W6_pure_noise
    )
    prob_A = expit(logit_A)
    A = np.random.binomial(1, prob_A, size=n_samples)

    # 3. Generate Outcome (Y) - Depends on A, W1, W2, W5_noise_Y
    true_cate = 2.0  # Constant treatment effect

    mean_Y = (
        1.0  # Intercept
        + true_cate * A
        + 1.5 * W1
        + 1.0 * W2
        # No direct effect of W3_noise_A
        # No direct effect of W4_noise_A
        - 1.3 * W5_noise_Y  # Variable affecting Y only
        # No effect of W6_pure_noise
    )

    noise_std = 1.0
    Y = mean_Y + np.random.normal(0, noise_std, size=n_samples)

    # Create DataFrame
    feature_cols = [f"W{i + 1}" for i in range(n_features)]
    data = pd.DataFrame(W, columns=feature_cols)
    data["A"] = A
    data["Y"] = Y

    return data, feature_cols
