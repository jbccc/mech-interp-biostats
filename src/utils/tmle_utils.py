import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression


def create_super_learner(model_type="regressor", random_state=None):
    """Creates a scikit-learn Stacking ensemble as a Super Learner stand-in."""
    if model_type == "regressor":
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=50, random_state=random_state)),
            (
                "gb",
                GradientBoostingRegressor(n_estimators=50, random_state=random_state),
            ),
            ("lr", LinearRegression()),
        ]
        final_estimator = LinearRegression()
        sl = StackingRegressor(
            estimators=estimators, final_estimator=final_estimator, cv=3
        )  # cv=3 for internal stacking estimates
    else:
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=50, random_state=random_state)),
            (
                "gb",
                GradientBoostingClassifier(n_estimators=50, random_state=random_state),
            ),
            ("lr", LogisticRegression(solver="liblinear", random_state=random_state)),
        ]
        final_estimator = LogisticRegression(
            solver="liblinear", random_state=random_state
        )
        sl = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=3,
            stack_method="predict_proba",
        )
    return sl


def run_tmle_with_models(
    W, A, Y, Q_model_instance, g_model_instance, p_score_clip=0.01
):
    """
    Performs the TMLE steps for ATE estimation with a continuous outcome using models.

    Args:
        W (pd.DataFrame): Covariates
        A (pd.Series): Treatment assignment
        Y (pd.Series): Outcome variable
        Q_model_instance: Scikit-learn model for outcome prediction
        g_model_instance: Scikit-learn model for propensity score prediction
        p_score_clip (float): Clipping threshold for propensity scores

    Returns:
        dict: TMLE results including ATE estimate and confidence interval
    """
    n = len(Y)
    # Convert inputs to FLAT numpy arrays (shape (n,)) where needed
    A_np = A.to_numpy().flatten()
    Y_np = Y.to_numpy().flatten()

    # --- Step 1: Estimate Propensity Score g(A=1|W) ---
    g_model = g_model_instance
    g_model.fit(W, A)  # Use original A Series for fit
    g_hat_all_probs = g_model.predict_proba(W)
    g_hat_1 = g_hat_all_probs[:, 1].flatten()  # Ensure flat (n,)
    g_hat_1 = np.clip(g_hat_1, p_score_clip, 1 - p_score_clip)
    g_hat_0 = 1 - g_hat_1  # Also flat (n,)

    # --- Step 2: Estimate Initial Outcome Regression Q(A,W) = E[Y|A,W] ---
    Q_model = Q_model_instance
    X_q = W.copy()
    X_q["A"] = A  # Use original A Series for constructing features
    Q_model.fit(X_q, Y)  # Use original Y Series for fit

    Q_hat_A = Q_model.predict(X_q).flatten()  # Ensure flat (n,)

    X_q1 = W.copy()
    X_q1["A"] = 1
    Q_hat_1 = Q_model.predict(X_q1).flatten()  # Ensure flat (n,)

    X_q0 = W.copy()
    X_q0["A"] = 0
    Q_hat_0 = Q_model.predict(X_q0).flatten()  # Ensure flat (n,)

    # Use the targeting step with the predicted values
    return run_tmle_targeting(A_np, Y_np, Q_hat_A, Q_hat_1, Q_hat_0, g_hat_1, g_hat_0)


def run_tmle_with_predictions(
    A, Y, Q_preds, g_preds, Q_preds_a1=None, Q_preds_a0=None, p_score_clip=0.01
):
    """
    Performs the TMLE steps for ATE estimation with a continuous outcome using pre-computed predictions.

    Args:
        A (array-like): Treatment assignment
        Y (array-like): Outcome variable
        Q_preds (array-like): Predicted outcomes for the observed treatment
        g_preds (array-like): Predicted propensity scores P(A=1|W)
        Q_preds_a1 (array-like, optional): Predicted outcomes if everyone had A=1
        Q_preds_a0 (array-like, optional): Predicted outcomes if everyone had A=0
        p_score_clip (float): Clipping threshold for propensity scores

    Returns:
        dict: TMLE results including ATE estimate and confidence interval
    """
    # Ensure inputs are numpy arrays and flat
    A_np = np.asarray(A).flatten()
    Y_np = np.asarray(Y).flatten()
    Q_hat_A = np.asarray(Q_preds).flatten()
    g_hat_1 = np.asarray(g_preds).flatten()

    # Clip propensity scores to avoid extreme values
    g_hat_1 = np.clip(g_hat_1, p_score_clip, 1 - p_score_clip)
    g_hat_0 = 1 - g_hat_1

    # If counterfactual predictions aren't provided, we can't compute ATE
    if Q_preds_a1 is None or Q_preds_a0 is None:
        raise ValueError(
            "Both Q_preds_a1 and Q_preds_a0 must be provided for ATE estimation"
        )

    Q_hat_1 = np.asarray(Q_preds_a1).flatten()
    Q_hat_0 = np.asarray(Q_preds_a0).flatten()

    # Use the targeting step with the provided predictions
    return run_tmle_targeting(A_np, Y_np, Q_hat_A, Q_hat_1, Q_hat_0, g_hat_1, g_hat_0)


def calculate_tmle_from_preds(W, A, Y, Q_preds, g_preds, p_score_clip=0.01):
    """
    Simplified TMLE calculation using pre-computed Q and g predictions.
    This version approximates counterfactual predictions by averaging over treatment groups.

    Args:
        W (array-like): Covariates (unused in this function but kept for API consistency)
        A (array-like): Treatment assignment
        Y (array-like): Outcome variable
        Q_preds (array-like): Predicted outcomes for observed treatment
        g_preds (array-like): Predicted propensity scores
        p_score_clip (float): Clipping threshold for propensity scores

    Returns:
        dict: TMLE results including ATE estimate and confidence interval
    """
    # Convert inputs to numpy arrays
    A_np = np.asarray(A).flatten()
    Y_np = np.asarray(Y).flatten()
    Q_hat_A = np.asarray(Q_preds).flatten()
    g_hat_1 = np.asarray(g_preds).flatten()

    # Clip propensity scores
    g_hat_1 = np.clip(g_hat_1, p_score_clip, 1 - p_score_clip)
    g_hat_0 = 1 - g_hat_1

    n = len(Y_np)

    # Calculate clever covariate H
    H_1 = A_np / g_hat_1
    H_0 = (1 - A_np) / g_hat_0
    H = H_1 - H_0

    # Estimate fluctuation parameter epsilon
    fluctuation_model = LinearRegression(fit_intercept=False)
    H_reshaped = H.reshape(-1, 1)
    residuals = Y_np - Q_hat_A
    fluctuation_model.fit(H_reshaped, residuals)
    epsilon_hat = fluctuation_model.coef_[0]

    # Update the predicted outcomes
    Q_star_A = Q_hat_A + epsilon_hat * H

    # For this simplified version, we approximate Q(1,W) and Q(0,W)
    # by averaging over the treatment groups after targeting
    Q_star_1_mean = np.mean(Q_star_A[A_np == 1])
    Q_star_0_mean = np.mean(Q_star_A[A_np == 0])

    # Estimated ATE
    ate_tmle = Q_star_1_mean - Q_star_0_mean

    # Simplified variance estimation
    n1 = np.sum(A_np == 1)
    n0 = np.sum(A_np == 0)
    var1 = np.var(Q_star_A[A_np == 1], ddof=1) if n1 > 1 else 0
    var0 = np.var(Q_star_A[A_np == 0], ddof=1) if n0 > 1 else 0
    var_tmle = var1 / n1 + var0 / n0 if (n1 > 0 and n0 > 0) else np.nan

    sd_tmle = np.sqrt(var_tmle) if not np.isnan(var_tmle) else np.nan
    ci_lower = ate_tmle - 1.96 * sd_tmle if not np.isnan(sd_tmle) else np.nan
    ci_upper = ate_tmle + 1.96 * sd_tmle if not np.isnan(sd_tmle) else np.nan

    return {
        "ATE": ate_tmle,
        "StdDev": sd_tmle,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
        "CI_Width": (ci_upper - ci_lower)
        if not np.isnan(ci_lower) and not np.isnan(ci_upper)
        else np.nan,
        "Epsilon": epsilon_hat,
    }


def run_tmle_targeting(A_np, Y_np, Q_hat_A, Q_hat_1, Q_hat_0, g_hat_1, g_hat_0):
    """
    Core TMLE targeting step that updates initial predictions.

    Args:
        A_np (np.ndarray): Treatment assignment as flat numpy array
        Y_np (np.ndarray): Outcome variable as flat numpy array
        Q_hat_A (np.ndarray): Initial outcome predictions for observed treatment
        Q_hat_1 (np.ndarray): Initial outcome predictions if everyone had A=1
        Q_hat_0 (np.ndarray): Initial outcome predictions if everyone had A=0
        g_hat_1 (np.ndarray): Propensity score predictions P(A=1|W)
        g_hat_0 (np.ndarray): Propensity score predictions P(A=0|W)

    Returns:
        dict: TMLE results including ATE estimate and confidence interval
    """
    n = len(Y_np)

    # --- Step 3: Calculate Clever Covariate H ---
    H_1 = A_np / g_hat_1
    H_0 = (1 - A_np) / g_hat_0
    H = H_1 - H_0

    # --- Step 4: Estimate Fluctuation Parameter epsilon ---
    fluctuation_model = LinearRegression(fit_intercept=False)
    H_reshaped = H.reshape(-1, 1)
    residuals = Y_np - Q_hat_A

    fluctuation_model.fit(H_reshaped, residuals)
    epsilon_hat = fluctuation_model.coef_[0]

    # --- Step 5: Update Q estimates to Q_star ---
    H_1_W = (1 / g_hat_1).flatten()
    H_0_W = (-1 / g_hat_0).flatten()

    Q_star_1 = Q_hat_1 + epsilon_hat * H_1_W
    Q_star_0 = Q_hat_0 + epsilon_hat * H_0_W

    # --- Step 6: Calculate TMLE ATE Estimate ---
    ate_tmle = np.mean(Q_star_1 - Q_star_0)

    # --- Step 7: Estimate Influence Curve (EIC) ---
    Q_star_A = Q_hat_A + epsilon_hat * H
    eic = H * (Y_np - Q_star_A) + (Q_star_1 - Q_star_0) - ate_tmle

    # --- Step 8: Calculate Variance and Confidence Interval ---
    var_tmle = np.var(eic) / n
    sd_tmle = np.sqrt(var_tmle)
    ci_lower = ate_tmle - 1.96 * sd_tmle
    ci_upper = ate_tmle + 1.96 * sd_tmle

    return {
        "ATE": ate_tmle,
        "StdDev": sd_tmle,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
        "CI_Width": ci_upper - ci_lower,
        "Epsilon": epsilon_hat,
    }


def get_tmle(W, A, Y, seed=42):
    """
    Convenience function to run TMLE with Super Learner models.

    Args:
        W (pd.DataFrame): Covariates
        A (pd.Series): Treatment assignment
        Y (pd.Series): Outcome variable
        seed (int): Random seed for reproducibility

    Returns:
        dict: TMLE results
    """
    sl_q_model = create_super_learner(model_type="regressor", random_state=seed)
    sl_g_model = create_super_learner(model_type="classifier", random_state=seed)
    tmle_results_sl = run_tmle_with_models(W, A, Y, sl_q_model, sl_g_model)
    return tmle_results_sl
