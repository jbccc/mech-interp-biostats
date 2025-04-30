import os
import sys

sys.path.append(".")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.strong_confounder_data import generate_synthetic_data
from src.models.toy_tmle_strong_confounder import ToyTMLEStrongConfounder
from src.utils.tmle_utils import calculate_tmle_from_preds

SAVE_DIR = "exp1"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 231
ABLATION_PERCENTAGE = 10
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data(n_samples=5000):
    """Loads or generates synthetic data and returns dataloader and full tensors."""
    data, feature_cols = generate_synthetic_data(n_samples, seed=SEED)
    W = data[feature_cols]
    A = data["A"]
    Y = data["Y"]
    W1 = data["W1"]

    scaler = StandardScaler()
    W_scaled = scaler.fit_transform(W)

    W_full_tensor = torch.FloatTensor(W_scaled)
    A_full_tensor = torch.FloatTensor(A.values).unsqueeze(-1)
    Y_full_tensor = torch.FloatTensor(Y.values).unsqueeze(1)
    W1_full_tensor = torch.FloatTensor(W1.values)

    dataset = TensorDataset(
        W_full_tensor,
        A_full_tensor,
        Y_full_tensor,
        W1_full_tensor,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    return dataloader, W, A, Y, W1, W_full_tensor, A_full_tensor, Y_full_tensor


def load_model():
    """Loads the model, adapting input dimension."""
    model_trainer = ToyTMLEStrongConfounder()
    model = model_trainer.load_model()
    print(model)
    return model


def extract_activations_and_outputs(model, dataloader, layer_hooks):
    """Runs model, extracts activations via hooks, gets original predictions using dataloader."""
    print("Extracting activations and original predictions...")
    model.eval()
    activations = {name: [] for name in layer_hooks.keys()}
    true_W1_list = []
    original_Q_preds = []
    original_g_preds = []
    hooks = {}

    def get_activation_hook(name):
        def hook(module, input, output):
            activations[name].append(output.cpu())

        return hook

    for name, (module_name, layer_index) in layer_hooks.items():
        module = getattr(model, module_name)
        layer = module[layer_index]
        hooks[name] = layer.register_forward_hook(get_activation_hook(name))

    with torch.no_grad():
        for W_b, A_b, _, W1_b in dataloader:
            q_pred, g_pred = model(W_b, A_b)
            original_Q_preds.append(q_pred.cpu())
            original_g_preds.append(g_pred.cpu())
            true_W1_list.append(W1_b.cpu())

    for handle in hooks.values():
        handle.remove()

    processed_activations = {}
    for name in activations:
        if activations[name]:
            processed_activations[name] = torch.cat(activations[name], dim=0).numpy()
        else:
            print(f"Warning: No activations captured for {name}")

    cat_Q_orig = torch.cat(original_Q_preds, dim=0).squeeze().numpy()
    cat_g_orig = torch.cat(original_g_preds, dim=0).squeeze().numpy()
    cat_W1_true = torch.cat(true_W1_list, dim=0).numpy()

    print("Activation extraction complete.")
    return processed_activations, cat_W1_true, cat_Q_orig, cat_g_orig


def train_all_probes(activations, true_W1):
    """Trains linear probes for multiple layers."""
    print("--- Starting Probe Training for All Layers ---")
    probe_results = {}
    y_probe = true_W1

    for layer_name, X_probe in activations.items():
        print(f"\nTraining probe for layer: {layer_name}")
        if X_probe.shape[0] != y_probe.shape[0]:
            print(f"Skipping layer {layer_name}: Shape mismatch")
            probe_results[layer_name] = {"error": "Shape mismatch"}
            continue
        if X_probe.shape[1] == 0:
            print(f"Skipping layer {layer_name}: Zero neurons")
            probe_results[layer_name] = {"error": "Zero neurons"}
            continue

        try:
            probe_model = LinearRegression()
            probe_model.fit(X_probe, y_probe)
            y_pred = probe_model.predict(X_probe)
            r2 = r2_score(y_probe, y_pred)
            print(f" Probe R2 score for {layer_name}: {r2:.4f}")

            if probe_model.coef_.ndim > 1:
                importances = np.mean(np.abs(probe_model.coef_), axis=0)
            else:
                importances = np.abs(probe_model.coef_)

            if importances.shape[0] != X_probe.shape[1]:
                raise ValueError(f"Internal error: Importance/Neuron mismatch")

            sorted_indices = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_indices]

            probe_results[layer_name] = {
                "r2": r2,
                "sorted_indices": sorted_indices,
                "sorted_importances": sorted_importances,
                "importances": importances,
                "num_neurons": X_probe.shape[1],
            }
        except Exception as e:
            print(f"Error training probe for layer {layer_name}: {e}")
            probe_results[layer_name] = {"error": str(e)}

    print("\n--- Probe Training Completed ---")
    return probe_results


# --- TMLE Calculation ---


# --- Loss Calculation ---
def calculate_losses(q_preds, g_preds, A_tensor, Y_tensor):
    """Calculates Q (MSE) and g (BCE/log) losses."""
    try:
        y_true = Y_tensor.squeeze().numpy()
        a_true = A_tensor.squeeze().numpy()
        q_preds_np = q_preds.squeeze()
        g_preds_np = g_preds.squeeze()

        q_loss = mean_squared_error(y_true, q_preds_np)
        g_preds_clipped = np.clip(g_preds_np, 1e-9, 1 - 1e-9)
        g_loss = log_loss(a_true, g_preds_clipped)
        return {"Q_loss": q_loss, "g_loss": g_loss}
    except Exception as e:
        print(f"Error calculating losses: {e}")
        return {"Q_loss": np.nan, "g_loss": np.nan}


# --- Ablation ---
current_ablated_layer = None
current_indices_to_zero_tensor = None


def ablation_hook(module, input, output):
    """Hook function that performs ablation based on global state."""
    global current_ablated_layer, current_indices_to_zero_tensor
    if current_indices_to_zero_tensor is not None:
        mask = torch.ones_like(output)
        if current_indices_to_zero_tensor.numel() > 0:
            indices = current_indices_to_zero_tensor.long()
            indices = torch.clamp(indices, 0, output.shape[1] - 1)
            mask[:, indices] = 0
        return output * mask
    return output


def run_single_ablation_pass(
    model, W_full_tensor, A_full_tensor, layer_hooks, layer_to_ablate, indices_to_ablate
):
    """Runs a single forward pass with specified neurons ablated."""
    global current_ablated_layer, current_indices_to_zero_tensor

    indices_to_zero = np.array(indices_to_ablate).copy()
    indices_to_zero = indices_to_zero[~np.isnan(indices_to_zero)].astype(int)

    current_ablated_layer = layer_to_ablate
    current_indices_to_zero_tensor = torch.tensor(indices_to_zero, dtype=torch.long)

    model.eval()
    hook_handle = None
    q_ablated, g_ablated = None, None

    try:
        target_module_name, target_layer_index = layer_hooks[layer_to_ablate]
        target_module = getattr(model, target_module_name)
        target_layer = target_module[target_layer_index]
        hook_handle = target_layer.register_forward_hook(ablation_hook)

        with torch.no_grad():
            q_ablated, g_ablated = model(W_full_tensor, A_full_tensor)

    except Exception as e:
        print(f"Error during ablation forward pass for layer {layer_to_ablate}: {e}")

    finally:
        if hook_handle:
            hook_handle.remove()
        current_ablated_layer = None
        current_indices_to_zero_tensor = None

    if q_ablated is None or g_ablated is None:
        print(f"Ablation pass failed for layer {layer_to_ablate}, returning NaNs.")
        num_samples = W_full_tensor.shape[0]
        q_ablated = torch.full((num_samples, 1), float("nan"))
        g_ablated = torch.full((num_samples, 1), float("nan"))

    q_ablated_np = q_ablated.detach().cpu().squeeze().numpy()
    g_ablated_np = g_ablated.detach().cpu().squeeze().numpy()

    return q_ablated_np, g_ablated_np


def run_ablation_experiments(
    model,
    W_full_tensor,
    A_full_tensor,
    Y_full_tensor,
    W_pd,
    A_pd,
    Y_pd,
    q_orig_np,
    g_orig_np,
    probe_results,
    layer_hooks,
    layers_to_analyze,
):
    """
    Runs ablation experiments 1 and 2 for specified layers.

    Args:
        model: The trained PyTorch model.
        W_full_tensor, A_full_tensor, Y_full_tensor: Full dataset tensors.
        W_pd, A_pd, Y_pd: Original pandas data for TMLE calculation.
        q_orig_np, g_orig_np: Original model predictions (numpy).
        probe_results: Dictionary from train_all_probes.
        layer_hooks: Dictionary mapping layer names to model modules/indices.
        layers_to_analyze: List of layer names to perform ablation on.

    Returns:
        dict: Nested dictionary containing ablation results.
              results[layer_name]['baseline'] = {'tmle': tmle_dict, 'losses': loss_dict}
              results[layer_name]['experiment1'][ablation_type] = {'tmle': tmle_dict, 'losses': loss_dict, 'ate_change': ..., 'q_loss_change': ..., 'g_loss_change': ...}
              results[layer_name]['experiment2'] = {'ate_sweep': [ate1, ate2,...], 'percentages': [p1, p2,...]}
    """
    print("\n--- Running Ablation Experiments ---")
    ablation_results = {}

    # --- Calculate Baseline Metrics ---
    print("Calculating Baseline TMLE and Losses...")
    baseline_tmle = calculate_tmle_from_preds(W_pd, A_pd, Y_pd, q_orig_np, g_orig_np)
    baseline_losses = calculate_losses(
        q_orig_np, g_orig_np, A_full_tensor, Y_full_tensor
    )
    print(
        f"Baseline ATE: {baseline_tmle.get('ATE', np.nan):.4f}, Q-Loss: {baseline_losses.get('Q_loss', np.nan):.4f}, G-Loss: {baseline_losses.get('g_loss', np.nan):.4f}"
    )

    for layer_name in layers_to_analyze:
        print(f"\n--- Ablating Layer: {layer_name} ---")
        if layer_name not in probe_results or "error" in probe_results[layer_name]:
            print(
                f"Skipping layer {layer_name} due to probing error or missing results."
            )
            ablation_results[layer_name] = {
                "error": "Skipped in ablation due to probe issues."
            }
            continue

        layer_probe_res = probe_results[layer_name]
        sorted_indices = layer_probe_res.get("sorted_indices")
        num_neurons = layer_probe_res.get("num_neurons")

        if sorted_indices is None or num_neurons is None:
            print(
                f"Skipping layer {layer_name}: Missing 'sorted_indices' or 'num_neurons' in probe results."
            )
            ablation_results[layer_name] = {"error": "Missing essential probe results."}
            continue

        if num_neurons == 0:
            print(f"Skipping layer {layer_name}: No neurons found.")
            ablation_results[layer_name] = {"error": "No neurons."}
            continue

        ablation_results[layer_name] = {
            "baseline": {"tmle": baseline_tmle, "losses": baseline_losses},
            "experiment1": {},
            "experiment2": {"ate_sweep": [], "percentages": []},
        }

        # --- Experiment 1: Fixed Percentage Ablation (Top, Bottom, Random) ---
        print(f"  Running Experiment 1 (Ablating {ABLATION_PERCENTAGE}% Neurons)...")
        num_to_ablate = max(1, int(num_neurons * (ABLATION_PERCENTAGE / 100.0)))
        num_to_ablate = min(num_to_ablate, num_neurons)
        print(
            f"    Number of neurons to ablate: {num_to_ablate} (out of {num_neurons})"
        )

        indices_exp1 = {
            "Top": sorted_indices[:num_to_ablate],
            "Bottom": sorted_indices[-num_to_ablate:],
            "Random": np.random.choice(
                np.arange(num_neurons), num_to_ablate, replace=False
            ),
        }

        for ablation_type, indices in indices_exp1.items():
            if indices is None or len(indices) == 0 and num_to_ablate > 0:
                print(
                    f"    Warning: Empty or invalid indices generated for {ablation_type} {ABLATION_PERCENTAGE}%. Skipping."
                )
                ablated_tmle = {
                    "ATE": np.nan,
                    "StdDev": np.nan,
                    "CI_Width": np.nan,
                    "error": "Invalid indices",
                }
                ablated_losses = {
                    "Q_loss": np.nan,
                    "g_loss": np.nan,
                    "error": "Invalid indices",
                }
                ate_change, q_loss_change, g_loss_change = np.nan, np.nan, np.nan
            else:
                print(f"    Ablating {ablation_type} {ABLATION_PERCENTAGE}%...")
                q_abl, g_abl = run_single_ablation_pass(
                    model,
                    W_full_tensor,
                    A_full_tensor,
                    layer_hooks,
                    layer_name,
                    indices,
                )

                if np.isnan(q_abl).any() or np.isnan(g_abl).any():
                    print(
                        f"      Skipping TMLE/Loss calculation due to NaNs in ablation output for {ablation_type}."
                    )
                    ablated_tmle = {
                        "ATE": np.nan,
                        "StdDev": np.nan,
                        "CI_Width": np.nan,
                        "error": "NaN from ablation pass",
                    }
                    ablated_losses = {
                        "Q_loss": np.nan,
                        "g_loss": np.nan,
                        "error": "NaN from ablation pass",
                    }
                    ate_change, q_loss_change, g_loss_change = np.nan, np.nan, np.nan
                else:
                    ablated_tmle = calculate_tmle_from_preds(
                        W_pd, A_pd, Y_pd, q_abl, g_abl
                    )
                    ablated_losses = calculate_losses(
                        q_abl, g_abl, A_full_tensor, Y_full_tensor
                    )
                    ate_change = ablated_tmle.get("ATE", np.nan) - baseline_tmle.get(
                        "ATE", np.nan
                    )
                    q_loss_change = ablated_losses.get(
                        "Q_loss", np.nan
                    ) - baseline_losses.get("Q_loss", np.nan)
                    g_loss_change = ablated_losses.get(
                        "g_loss", np.nan
                    ) - baseline_losses.get("g_loss", np.nan)

                print(
                    f"      ATE Change: {ate_change:.4f}, Q-Loss Change: {q_loss_change:.4f}, G-Loss Change: {g_loss_change:.4f}"
                )

            ablation_results[layer_name]["experiment1"][ablation_type] = {
                "tmle": ablated_tmle,
                "losses": ablated_losses,
                "ate_change": ate_change,
                "q_loss_change": q_loss_change,
                "g_loss_change": g_loss_change,
                "indices_ablated": indices,
            }

        # --- Experiment 2: Ablation by Importance Percentile Bands ---
        band = 10
        print(f"  Running Experiment 2 (Ablating by {band}% Importance Bands)...")
        num_bands = 100 // band
        band_labels = [f"{i * band}-{(i + 1) * band}%" for i in range(num_bands)]
        ate_bands = []

        for i in range(num_bands):
            band_start_perc = i * band
            band_end_perc = (i + 1) * band

            # Recalculate indices based on sorted order
            # most important frist, for example with 20% band
            # sorted_indices goes from MOST important [0] to LEAST important [-1]
            # Band 80-100% (most important) = sorted_indices[0 : ceil(0.05*N)]
            # Band 0-20% (least important) = sorted_indices[floor(0.95*N) : N]
            idx_for_band_start = int(
                np.floor(num_neurons * ((100 - band_end_perc) / 100.0))
            )
            idx_for_band_end = int(
                np.ceil(num_neurons * ((100 - band_start_perc) / 100.0))
            )

            idx_for_band_start = min(max(0, idx_for_band_start), num_neurons)
            idx_for_band_end = min(max(0, idx_for_band_end), num_neurons)

            if idx_for_band_start >= idx_for_band_end:
                print(
                    f"    Skipping band {band_labels[i]} ({idx_for_band_start}:{idx_for_band_end}): No neurons in this range based on sorted importance."
                )
                ate_p = np.nan
                ate_bands.append(ate_p)
                continue

            indices_sweep = sorted_indices[idx_for_band_start:idx_for_band_end]

            print(
                f"    Ablating band {band_labels[i]} (Indices {idx_for_band_start} to {idx_for_band_end - 1} of sorted list, {len(indices_sweep)} neurons)..."
            )

            if len(indices_sweep) == 0:
                print(
                    f"    Warning: Empty indices_sweep for band {band_labels[i]}. Setting ATE to NaN."
                )
                ate_p = np.nan
            else:
                q_abl_sweep, g_abl_sweep = run_single_ablation_pass(
                    model,
                    W_full_tensor,
                    A_full_tensor,
                    layer_hooks,
                    layer_name,
                    indices_sweep,
                )

                if np.isnan(q_abl_sweep).any() or np.isnan(g_abl_sweep).any():
                    print(
                        f"      Skipping TMLE calculation for band {band_labels[i]} due to NaNs in ablation output."
                    )
                    ate_p = np.nan
                else:
                    tmle_sweep = calculate_tmle_from_preds(
                        W_pd, A_pd, Y_pd, q_abl_sweep, g_abl_sweep
                    )
                    ate_p = tmle_sweep.get("ATE", np.nan)

            ate_bands.append(ate_p)

        ablation_results[layer_name]["experiment2"] = {
            "bands": band_labels,
            "ate_bands": ate_bands,
        }
        print(f"  Experiment 2 complete for layer {layer_name}.")

    print("\n--- Ablation Experiments Completed ---")
    return ablation_results


# --- Plotting Probe Analysis ---
def plot_probes_analysis(probe_results: dict[str, dict]):
    """Generates summary plots for probe results."""
    plots = {}
    valid_layers_data = []
    print("\n--- Processing data for probe summary plots ---")
    layer_names_all = list(probe_results.keys())

    for layer_name in layer_names_all:
        results = probe_results[layer_name]

        if (
            isinstance(results, dict)
            and "error" not in results
            and all(k in results for k in ["r2", "sorted_importances", "num_neurons"])
        ):
            num_total_neurons = results["num_neurons"]
            if num_total_neurons == 0:
                print(f"Skipping layer {layer_name} for plots: No neurons.")
                continue

            sorted_importances = results["sorted_importances"]

            if not isinstance(sorted_importances, np.ndarray):
                print(
                    f"Warning: sorted_importances for {layer_name} is not a numpy array. Type: {type(sorted_importances)}"
                )
                sorted_importances = np.array(sorted_importances)

            total_importance = (
                np.sum(sorted_importances) if len(sorted_importances) > 0 else 0.0
            )
            num_neurons_95, num_neurons_75, num_neurons_50 = 0, 0, 0

            if len(sorted_importances) > 0 and total_importance > 1e-9:
                cumulative_importances = np.cumsum(sorted_importances)

                cutoff_index_95 = np.searchsorted(
                    cumulative_importances, 0.95 * total_importance, side="left"
                )
                cutoff_index_75 = np.searchsorted(
                    cumulative_importances, 0.75 * total_importance, side="left"
                )
                cutoff_index_50 = np.searchsorted(
                    cumulative_importances, 0.50 * total_importance, side="left"
                )

                num_neurons_95 = min(int(cutoff_index_95 + 1), int(num_total_neurons))
                num_neurons_75 = min(int(cutoff_index_75 + 1), int(num_total_neurons))
                num_neurons_50 = min(int(cutoff_index_50 + 1), int(num_total_neurons))

            elif num_total_neurons > 0:
                print(
                    f"Warning: Layer {layer_name} has zero total importance or only one importance value."
                )
                num_neurons_95 = num_neurons_75 = num_neurons_50 = 0

            valid_layers_data.append(
                {
                    "name": layer_name,
                    "r2": results["r2"],
                    "num_neurons_95": num_neurons_95,
                    "num_neurons_75": num_neurons_75,
                    "num_neurons_50": num_neurons_50,
                    "sorted_importances": sorted_importances,
                    "num_total_neurons": num_total_neurons,
                }
            )
        else:
            error_msg = (
                results.get("error", "Invalid structure")
                if isinstance(results, dict)
                else "Not a dict"
            )
            print(
                f"Skipping layer {layer_name} for plots: Invalid/Error results ({error_msg})."
            )

    # Plot 1: R2 and Neuron Count Evolution
    print("\n--- Generating R2 & Neuron Counts Evolution Plot ---")
    if valid_layers_data:
        layer_names = [d["name"] for d in valid_layers_data]
        r2_scores = [d["r2"] for d in valid_layers_data]
        neurons_95 = [d["num_neurons_95"] for d in valid_layers_data]
        neurons_75 = [d["num_neurons_75"] for d in valid_layers_data]
        neurons_50 = [d["num_neurons_50"] for d in valid_layers_data]
        x_indices = np.arange(len(layer_names))

        fig1, ax1 = plt.subplots(figsize=(14, 7))
        fig1.subplots_adjust(right=0.80)
        color1 = "tab:blue"
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("R² Score", color=color1, fontsize=12)
        ln1 = ax1.plot(
            x_indices,
            r2_scores,
            marker="o",
            linestyle="-",
            markersize=8,
            linewidth=2,
            color=color1,
            label="R² Score",
        )
        ax1.tick_params(axis="y", labelcolor=color1, labelsize=11)
        ax1.tick_params(axis="x", labelsize=11)
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(layer_names, rotation=45, ha="right")
        ax1.grid(True, axis="y", linestyle="--", linewidth=0.5)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Neurons for X% Importance", color=color2, fontsize=12)
        ln2 = ax2.plot(
            x_indices,
            neurons_95,
            marker="s",
            linestyle="-",
            markersize=7,
            color=color2,
            label="# Neurons (95%)",
        )
        ln3 = ax2.plot(
            x_indices,
            neurons_75,
            marker="^",
            linestyle="--",
            markersize=7,
            color=color2,
            label="# Neurons (75%)",
        )
        ln4 = ax2.plot(
            x_indices,
            neurons_50,
            marker="d",
            linestyle=":",
            markersize=7,
            color=color2,
            label="# Neurons (50%)",
        )
        ax2.tick_params(axis="y", labelcolor=color2, labelsize=11)

        lns = ln1 + ln2 + ln3 + ln4
        labs = [str(l.get_label()) for l in lns]
        ax1.legend(lns, labs, loc="best", fontsize=10)
        ax1.set_title(
            "Probe R² and Neuron Counts for Importance Thresholds Across Layers",
            fontsize=14,
        )
        fig1.tight_layout(rect=(0, 0, 0.95, 1))
        plots["r2_neurons_evolution"] = fig1
        plt.savefig(f"{SAVE_DIR}/r2_neurons_evolution.png")
        print("Saved r2_neurons_evolution.png")
    else:
        print("No valid data for R2 & Neuron Counts Evolution plot.")

    # Plot 2: Cumulative Importance Curves
    print("\n--- Generating Cumulative Importance Plot ---")
    fig2, ax2_cum = plt.subplots(figsize=(10, 8))
    layers_plotted = 0
    valid_layers_plot2 = [
        d
        for d in valid_layers_data
        if d["num_total_neurons"] > 0
        and isinstance(d["sorted_importances"], np.ndarray)
        and len(d["sorted_importances"]) > 0
    ]

    if valid_layers_plot2:
        try:
            if len(valid_layers_plot2) > 0:
                cmap = plt.get_cmap("viridis")
                colors = [cmap(i) for i in np.linspace(0, 1, len(valid_layers_plot2))]
            else:
                colors = []
        except Exception as e:
            print(
                f"Could not generate colors using matplotlib.cm.viridis: {e}. Using fallback."
            )
            cmap_name = "tab10" if len(valid_layers_plot2) <= 10 else "tab20c"
            cmap = plt.get_cmap(cmap_name)
            num_colors_needed = len(valid_layers_plot2)
            colors = (
                [cmap(i) for i in np.linspace(0, 1, num_colors_needed)]
                if num_colors_needed > 0
                else []
            )

        color_idx = 0
        for layer_data in valid_layers_plot2:
            layer_name = layer_data["name"]
            sorted_importances = layer_data["sorted_importances"]
            num_neurons = layer_data["num_total_neurons"]
            total_importance = np.sum(sorted_importances)

            if num_neurons <= 0:
                continue

            current_color = colors[color_idx % len(colors)] if colors else "blue"

            if total_importance > 1e-9:
                cumulative_importances = np.cumsum(sorted_importances)
                percentage_importance = np.concatenate(
                    (
                        [0],
                        (cumulative_importances / total_importance) * 100
                        if total_importance > 1e-9
                        else np.zeros_like(cumulative_importances),
                    )
                )
                percentage_neurons = np.concatenate(
                    (
                        [0],
                        (np.arange(1, num_neurons + 1) / num_neurons) * 100
                        if num_neurons > 0
                        else [0],
                    )
                )

                if len(percentage_neurons) == len(percentage_importance):
                    ax2_cum.plot(
                        percentage_neurons,
                        percentage_importance,
                        label=layer_name,
                        marker=".",
                        linestyle="-",
                        markersize=4,
                        color=current_color,
                    )
                    layers_plotted += 1
                else:
                    print(
                        f"Warning: Mismatch in length for cumulative plot data for layer {layer_name}. Skipping."
                    )

            else:
                percentage_neurons = np.concatenate(
                    (
                        [0],
                        (np.arange(1, num_neurons + 1) / num_neurons) * 100
                        if num_neurons > 0
                        else [0],
                    )
                )
                percentage_importance = np.zeros_like(percentage_neurons)
                if len(percentage_neurons) == len(percentage_importance):
                    ax2_cum.plot(
                        percentage_neurons,
                        percentage_importance,
                        label=f"{layer_name} (Zero Imp.)",
                        linestyle="--",
                        color=current_color,
                    )
                    layers_plotted += 1
                else:
                    print(
                        f"Warning: Mismatch in length for zero importance plot data for layer {layer_name}. Skipping."
                    )

            color_idx += 1

    if layers_plotted > 0:
        ax2_cum.plot(
            [0, 100], [0, 100], color="grey", linestyle=":", label="Baseline (y=x)"
        )
        ax2_cum.set_title(
            "Cumulative Importance vs. % Neurons Across Layers", fontsize=14
        )
        ax2_cum.set_xlabel("% of Neurons (Sorted by Importance)", fontsize=12)
        ax2_cum.set_ylabel("% of Total Importance Explained", fontsize=12)
        ax2_cum.legend(loc="best", fontsize="small")
        ax2_cum.grid(True, linestyle="--", linewidth=0.5)
        ax2_cum.tick_params(axis="both", which="major", labelsize=11)
        ax2_cum.set_ylim(0, 105)
        ax2_cum.set_xlim(0, 100)
        fig2.tight_layout()
        plots["cumulative_importance"] = fig2
        plt.savefig(f"{SAVE_DIR}/cumulative_importance_neurons.png")
        print("Saved cumulative_importance_neurons.png")
    else:
        print("No valid data available to generate Cumulative Importance plot.")
        plt.close(fig2)

    return plots


def plot_ablation_experiment1(ablation_results, layers_to_analyse):
    """Plots Experiment 1 results: Absolute ATE with CIs for Baseline, Top, Bottom, Random ablation."""
    print("\n--- Generating Ablation Experiment 1 Plot (Absolute ATE with CIs) ---")
    plot_data = []
    layers_analyzed = []

    ordered_layers = [layer for layer in layers_to_analyse if layer in ablation_results]

    for layer in ordered_layers:
        results = ablation_results[layer]
        if (
            isinstance(results, dict)
            and "baseline" in results
            and "experiment1" in results
            and isinstance(results["baseline"], dict)
            and isinstance(results["experiment1"], dict)
        ):
            layers_analyzed.append(layer)

            baseline_tmle = results["baseline"].get("tmle", {})
            if baseline_tmle and not baseline_tmle.get("error"):
                plot_data.append(
                    {
                        "Layer": layer,
                        "Ablation Type": "TMLE",
                        "ATE": baseline_tmle.get("ATE", np.nan),
                        "CI_Lower": baseline_tmle.get("CI_Lower", np.nan),
                        "CI_Upper": baseline_tmle.get("CI_Upper", np.nan),
                    }
                )
            else:
                plot_data.append(
                    {
                        "Layer": layer,
                        "Ablation Type": "TMLE",
                        "ATE": np.nan,
                        "CI_Lower": np.nan,
                        "CI_Upper": np.nan,
                    }
                )
                print(f"Warning: Missing or invalid baseline TMLE for {layer}.")

            ablation_order = ["Top", "Bottom", "Random"]
            for ablation_type in ablation_order:
                if ablation_type in results["experiment1"]:
                    exp1_res = results["experiment1"][ablation_type]
                    exp1_tmle = exp1_res.get("tmle", {})
                    if (
                        isinstance(exp1_res, dict)
                        and exp1_tmle
                        and not exp1_tmle.get("error")
                    ):
                        plot_data.append(
                            {
                                "Layer": layer,
                                "Ablation Type": ablation_type,
                                "ATE": exp1_tmle.get("ATE", np.nan),
                                "CI_Lower": exp1_tmle.get("CI_Lower", np.nan),
                                "CI_Upper": exp1_tmle.get("CI_Upper", np.nan),
                            }
                        )
                    else:
                        plot_data.append(
                            {
                                "Layer": layer,
                                "Ablation Type": ablation_type,
                                "ATE": np.nan,
                                "CI_Lower": np.nan,
                                "CI_Upper": np.nan,
                            }
                        )
                        print(
                            f"Warning: Missing or invalid TMLE for {layer} - {ablation_type}. Plotting as NaN."
                        )
                else:
                    plot_data.append(
                        {
                            "Layer": layer,
                            "Ablation Type": ablation_type,
                            "ATE": np.nan,
                            "CI_Lower": np.nan,
                            "CI_Upper": np.nan,
                        }
                    )
                    print(
                        f"Warning: Missing results for {layer} - {ablation_type}. Plotting as NaN."
                    )
        else:
            print(
                f"Skipping layer {layer} for Exp 1 plot: Missing baseline or experiment1 results."
            )

    if not plot_data:
        print("No valid data to plot for Ablation Experiment 1.")
        return None

    df_plot = pd.DataFrame(plot_data)

    err_lower = df_plot["ATE"] - df_plot["CI_Lower"]
    err_upper = df_plot["CI_Upper"] - df_plot["ATE"]

    asymmetric_error = [err_lower.fillna(0).tolist(), err_upper.fillna(0).tolist()]

    plt.figure(figsize=(14, 8))
    bar_width = 0.2
    x_indices = np.arange(len(layers_analyzed))
    ablation_types_plot = ["TMLE", "Top", "Bottom", "Random"]
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(ablation_types_plot))
    colors = plt.cm.get_cmap("viridis", len(ablation_types_plot))

    for i, ab_type in enumerate(ablation_types_plot):
        type_data = df_plot[df_plot["Ablation Type"] == ab_type]

        type_data = type_data.set_index("Layer").reindex(layers_analyzed).reset_index()

        ate_values = type_data["ATE"]

        ci_low = type_data["CI_Lower"]
        ci_up = type_data["CI_Upper"]
        yerr_lower = ate_values - ci_low
        yerr_upper = ci_up - ate_values
        asymmetric_error = np.array([yerr_lower.fillna(0), yerr_upper.fillna(0)])

        plt.bar(
            x_indices + offsets[i],
            ate_values,
            bar_width,
            label=ab_type,
            color=colors(i),
        )

        valid_idx = ate_values.notna()
        plt.errorbar(
            x_indices[valid_idx] + offsets[i],
            ate_values[valid_idx],
            yerr=asymmetric_error[:, valid_idx],
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1,
        )

    plt.title(f"Experiment 1: Absolute ATE ({ABLATION_PERCENTAGE}% Neurons Ablated)")
    plt.ylabel("TMLE ATE Estimate (with 95% CI)")
    plt.xlabel("Layer Ablated")
    plt.xticks(x_indices, layers_analyzed, rotation=45, ha="right")
    plt.legend(title="Ablation Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    filename = f"{SAVE_DIR}/ablation_exp1_absolute_ate_ci.png"
    plt.savefig(filename)
    print(f"Saved Experiment 1 plot to {filename}")
    return plt.gcf()


def plot_ablation_experiment2(ablation_results, layers_to_analyse):
    """Plots Experiment 2 results: ATE vs. importance percentile band."""
    print("\n--- Generating Ablation Experiment 2 Plot (ATE by Importance Band) ---")

    layers_with_data = []
    for layer in layers_to_analyse:  # Iterate in defined order
        results = ablation_results.get(layer, {})
        if isinstance(results, dict) and "experiment2" in results:
            exp2_res = results["experiment2"]
            if (
                isinstance(exp2_res, dict)
                and exp2_res.get("ate_bands")
                and exp2_res.get("bands")
                and len(exp2_res["ate_bands"]) == len(exp2_res["bands"])
            ):
                layers_with_data.append(layer)
            else:
                print(
                    f"Skipping layer {layer} for Exp 2 plot: Inconsistent/missing data."
                )
        else:
            print(
                f"Skipping layer {layer} for Exp 2 plot: Missing experiment2 results."
            )

    if not layers_with_data:
        print("No valid data to plot for Ablation Experiment 2.")
        return None

    plot_data = []
    baseline_ate = None
    for layer in layers_with_data:
        results = ablation_results[layer]
        if baseline_ate is None and "baseline" in results:
            baseline_ate = results["baseline"].get("tmle", {}).get("ATE")

        exp2_res = results["experiment2"]
        bands = exp2_res["bands"]
        ate_bands_vals = exp2_res["ate_bands"]

        for band_label, ate_val in zip(bands, ate_bands_vals):
            plot_data.append(
                {"Layer": layer, "Importance Band": band_label, "ATE": ate_val}
            )

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=df_plot,
        x="Importance Band",
        y="ATE",
        hue="Layer",
        hue_order=layers_with_data,
        marker="o",
        sort=False,
    )

    plt.title("Experiment 2: ATE vs. Neuron Importance Band")
    plt.ylabel("TMLE ATE Estimate")
    plt.xlabel("Neuron Importance Band (0-10% = Least Important)")

    if baseline_ate is not None and not np.isnan(baseline_ate):
        plt.axhline(
            baseline_ate,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Baseline ATE ({baseline_ate:.3f})",
        )

    plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    filename = f"{SAVE_DIR}/ablation_exp2_ate_bands_line.png"
    plt.savefig(filename)
    print(f"Saved Experiment 2 plot to {filename}")
    return plt.gcf()


# -- Display Plots --
def show_plots():
    """Displays generated plots."""
    print("\nDisplaying generated plots (if any)...")
    if plt.get_fignums():
        plt.show()
    else:
        print("No plots to display.")


# --- Main Experiment Runner ---
def run_experiment():
    all_plots = {}

    # 1. Load Data (including full tensors)
    print("--- Loading Data ---")
    dataloader, W_pd, A_pd, Y_pd, _, W_full_tensor, A_full_tensor, Y_full_tensor = (
        load_data()
    )

    # 2. Load Model
    print("--- Loading Model ---")
    model = load_model()
    LAYER_HOOK_POINTS = {
        f"h{i}": ("shared", 4 * i - 1) for i in range(len(model.shared) // 4 - 1)
    }
    LAYERS_TO_ANALYZE = LAYER_HOOK_POINTS.keys()

    # 3. Extract Activations and Original Predictions
    print("--- Extracting Activations & Original Predictions ---")

    activations, true_W1_np, q_orig_np, g_orig_np = extract_activations_and_outputs(
        model, dataloader, LAYER_HOOK_POINTS
    )

    # 4. Probing Phase
    print("\n--- Running Probing Analysis ---")
    probe_results = train_all_probes(activations, true_W1_np)
    probe_plots = plot_probes_analysis(probe_results)
    if probe_plots:
        all_plots.update(probe_plots)

    # --- Output Probe Table with Baseline Comparison ---
    print("\n--- Probe Accuracy Table (R2 for W1 Prediction) ---")
    print(f"{'Layer':<10} {'Probe R2':<10}")
    print("-" * 35)
    for layer_name, results in probe_results.items():
        if "r2" in results:
            print(f"{layer_name:<10} {results['r2']:.4f}{'':<5}")
        else:
            error_msg = results.get("error", "N/A")
            print(f"{layer_name:<10} {'Error':<10} ({error_msg})")
    print("-" * 35)

    # 5. Ablation Experiments
    ablation_results = run_ablation_experiments(
        model,
        W_full_tensor,
        A_full_tensor,
        Y_full_tensor,
        W_pd,
        A_pd,
        Y_pd,
        q_orig_np,
        g_orig_np,
        probe_results,
        LAYER_HOOK_POINTS,
        LAYERS_TO_ANALYZE,
    )

    # 6. Plot Ablation Results
    exp1_fig = plot_ablation_experiment1(ablation_results, LAYERS_TO_ANALYZE)
    if exp1_fig:
        all_plots["ablation_exp1"] = exp1_fig
    exp2_fig = plot_ablation_experiment2(ablation_results, LAYERS_TO_ANALYZE)
    if exp2_fig:
        all_plots["ablation_exp2"] = exp2_fig

    # 7. Show Plots (Optional - plots are saved)
    show_plots()

    print("\n--- Experiment 5.1 Complete ---")


if __name__ == "__main__":
    run_experiment()
