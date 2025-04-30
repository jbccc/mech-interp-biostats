import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.no_effect_confounders_data import \
    generate_no_effect_confounders_data
from src.models.toy_tmle_no_effect_confounder import ToyTMLENoEffectConfounder
from src.utils.causal_pathways_utils import multiplot_pathway_analyses

OUTPUT_DIR = "exp2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Model Loader --
def load_model():
    model_trainer = ToyTMLENoEffectConfounder()
    model = model_trainer.load_model()
    print(f"Model loaded successfully: {type(model).__name__}")
    return model


# --- Main Execution ---
if __name__ == "__main__":
    SEED = 42
    N_SAMPLES = 1000
    ACTIVATION_THRESHOLD = 1e-6

    TOP_K_VALUES = [5, 10, 15, 20]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Load Data to get Input Dimension ---
    print("Loading data to determine input dimension...")
    data, feature_cols = generate_no_effect_confounders_data(
        n_samples=N_SAMPLES, seed=SEED
    )
    INPUT_DIM = len(feature_cols)
    print(f"Input dimension determined: {INPUT_DIM}")

    model = load_model()

    ordered_layer_names = []
    if hasattr(model, "shared") and isinstance(model.shared, nn.Sequential):
        for i, layer in enumerate(model.shared):
            if isinstance(layer, nn.Dropout):
                ordered_layer_names.append(f"shared.{i}")
            elif isinstance(layer, nn.ReLU):
                ordered_layer_names.append(f"shared.{i}")
            elif isinstance(layer, nn.Linear):
                ordered_layer_names.append(f"shared.{i}")

    guidelines = {
        "input_dim": INPUT_DIM,
        "ordered_layer_names": ordered_layer_names,
    }
    print("Constructed Guidelines:")
    print(guidelines)
    print("-" * 30)

    # --- Run analysis with multiple top_k values ---
    print(f"Running pathway analyses with top_k values: {TOP_K_VALUES}")
    multiplot_pathway_analyses(
        model=model,
        input_dim=INPUT_DIM,
        top_k_values=TOP_K_VALUES,
        activation_threshold=ACTIVATION_THRESHOLD,
        guidelines=guidelines,
        output_dir=OUTPUT_DIR,
        show_plots=True,
    )

    print("\nAnalysis finished.")
    print(f"All plots saved to directory: {OUTPUT_DIR}")
