import os
from collections import defaultdict, deque
from typing import Any, Callable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


def register_hooks(
    model: nn.Module, layer_module_names: list[str]
) -> tuple[dict[str, torch.Tensor], list[Any], dict[str, nn.Module]]:
    """
    Registers forward hooks on specified modules using their attribute names.

    Args:
        model: The model containing the modules.
        layer_module_names: List of strings representing the attribute access path
                           (e.g., 'layer1', 'encoder.block0.ffn') to the modules.

    Returns:
        Tuple containing:
        - activations: Dictionary to store captured activations.
        - handles: List of hook handles.
        - layer_modules: Dictionary mapping the provided names to the actual module objects.
    """
    activations = {}
    handles = []
    layer_modules = {}

    def get_hook(name: str) -> Callable:
        def hook_fn(module: nn.Module, input: Any, output: Any):
            # Handle potential tuple outputs (e.g., from RNNs or Attention)
            # For simplicity, take the first tensor element if output is a tuple
            activation_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(activation_tensor, torch.Tensor):
                activations[name] = activation_tensor.detach().clone()
            else:
                print(
                    f"Warning: Output of hooked module '{name}' is not a Tensor or tuple containing a Tensor. Type: {type(output)}"
                )

        return hook_fn

    for name in layer_module_names:
        try:
            # Use get_submodule for robust access to nested modules
            module = model.get_submodule(name)
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
            layer_modules[name] = module
            # print(f"Successfully registered hook for: {name}") # Debug print
        except AttributeError:
            print(
                f"Warning: Module '{name}' not found in model. Skipping hook registration."
            )
        except Exception as e:
            print(f"Warning: Error registering hook for '{name}': {e}")

    if len(layer_modules) != len(layer_module_names):
        found_names = list(layer_modules.keys())
        expected_names = layer_module_names
        missing = [n for n in expected_names if n not in found_names]
        print(
            f"Warning: Mismatch in hooked modules. Expected {len(expected_names)}, got {len(found_names)}. Missing: {missing}"
        )

    return activations, handles, layer_modules


def remove_hooks(handles: list[Any]):
    """Removes registered hooks."""
    for handle in handles:
        handle.remove()


def trace_causal_pathways(
    model: nn.Module,
    base_input: torch.Tensor,
    source_input: torch.Tensor,
    top_k: int,
    activation_threshold: float,
    source_neuron_index: Optional[int] = None,
    causal_pathways_guidelines: Optional[dict] = None,
) -> list[
    tuple[Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool]
]:
    """
    Traces causal pathways using model-provided guidelines.

    Args:
        model: The PyTorch model (nn.Module). MUST have 'causal_pathways_guidelines'.
        base_input: The baseline input tensor (batch size 1).
        source_input: The source input tensor for patching (batch size 1).
        top_k: The number of top affected neurons to follow in the next layer.
        activation_threshold: Minimum activation change to consider a pathway.
        source_neuron_index: Optional index of the source neuron to use for the initial pathway source.
        causal_pathways_guidelines: Optional dictionary containing guidelines for causal pathway tracing.

    Returns:
        List of 4-tuples: (source, target, change, is_failed_source_marker)
          - Pathway: ( (layer_name, from_idx), (layer_name, to_idx), change, False )
          - Failed Source Marker: ( (layer_name, idx), None, None, True )
    """
    # --- 0. Check for and Extract Guidelines ---
    if causal_pathways_guidelines:
        guidelines = causal_pathways_guidelines
    elif hasattr(model, "causal_pathways_guidelines"):
        guidelines = model.causal_pathways_guidelines
    else:
        print(
            "Warning: No causal_pathways_guidelines provided or found on model. Using default."
        )
        guidelines = {
            "input_dim": 10,
            "ordered_layer_names": ["network.0", "network.2", "network.4"],
        }

    assert isinstance(guidelines, dict), "Guidelines must be a dictionary."

    required_keys = ["input_dim", "ordered_layer_names"]
    if not all(key in guidelines for key in required_keys):
        raise ValueError(
            f"'causal_pathways_guidelines' must contain keys: {required_keys}"
        )

    input_dim = guidelines.get("input_dim")
    ordered_layer_names = guidelines.get("ordered_layer_names")

    if not isinstance(input_dim, int):
        raise TypeError(
            f"'input_dim' in guidelines must be an integer, got {type(input_dim)}"
        )
    if not isinstance(ordered_layer_names, list) or not all(
        isinstance(name, str) for name in ordered_layer_names
    ):
        raise TypeError(
            f"'ordered_layer_names' in guidelines must be a list of strings, got {type(ordered_layer_names)}"
        )

    if not ordered_layer_names:
        print(
            "Warning: 'ordered_layer_names' in guidelines is empty. No tracing possible."
        )
        return []

    model.eval()
    intermediate_pathways: list[tuple[tuple[str, int], tuple[str, int], float]] = []
    visited_neurons = set()
    propagating_sources = set()
    processed_sources = set()

    # --- Setup Layers based on Guidelines ---
    ordered_layers = ["input"] + ordered_layer_names
    layer_name_to_idx = {name: i for i, name in enumerate(ordered_layers)}
    layer_sizes = {"input": input_dim}

    activations_base, handles_base, layer_modules = register_hooks(
        model, ordered_layer_names
    )
    if not layer_modules:
        print(
            "Error: No modules were successfully hooked based on guidelines. Aborting trace."
        )
        remove_hooks(handles_base)
        return []

    for name, module in layer_modules.items():
        if isinstance(module, nn.Linear):
            layer_sizes[name] = module.out_features
        elif isinstance(module, nn.Dropout) or isinstance(module, nn.ReLU):
            layer_sizes[name] = (
                activations_base[name].shape[-1] if name in activations_base else 1
            )
        else:
            if name in activations_base and activations_base[name].ndim > 1:
                layer_sizes[name] = activations_base[name].shape[-1]
                print(
                    f"Inferred size {layer_sizes[name]} for non-Linear layer '{name}' from activation shape."
                )
            else:
                print(
                    f"Warning: Could not determine output size for non-Linear layer '{name}'. Using size 1 as fallback."
                )
                layer_sizes[name] = 1

    # --- 1. Get Base Activations ---
    with torch.no_grad():
        try:
            _ = model(base_input, torch.zeros_like(base_input[:, :1]))
        except TypeError:
            _ = model(base_input)
    remove_hooks(handles_base)

    activations_source, handles_source, _ = register_hooks(model, ordered_layer_names)

    # --- 2. Get Source Activations ---
    with torch.no_grad():
        try:
            _ = model(source_input, torch.zeros_like(source_input[:, :1]))
        except TypeError:
            _ = model(source_input)
    remove_hooks(handles_source)

    # --- 3. Trace Initialization (Input -> First Real Layer) ---
    queue = deque()
    first_real_layer_name = ordered_layer_names[0]

    if (
        first_real_layer_name not in activations_base
        or first_real_layer_name not in activations_source
    ):
        print(
            f"Error: Activations for the first guideline layer '{first_real_layer_name}' not captured. Check hook registration."
        )
        return []
    if first_real_layer_name not in layer_sizes:
        print(
            f"Error: Size for the first guideline layer '{first_real_layer_name}' not determined."
        )
        return []

    base_target_act = activations_base[first_real_layer_name][0]
    source_target_act = activations_source[first_real_layer_name][0]
    initial_changes = torch.abs(source_target_act - base_target_act)
    target_layer_size = layer_sizes[first_real_layer_name]

    if initial_changes.numel() == 0:
        print(
            f"Warning: Initial activation changes for layer '{first_real_layer_name}' are empty."
        )
    else:
        flat_changes = initial_changes.flatten()
        num_elements = flat_changes.numel()
        current_top_k = min(top_k, num_elements)

        if num_elements > 0:
            top_k_flat_indices = torch.argsort(flat_changes, descending=True)[
                :current_top_k
            ]
            top_k_changes = flat_changes[top_k_flat_indices]

            for i in range(len(top_k_flat_indices)):
                target_neuron_idx = int(top_k_flat_indices[i].item())
                change = top_k_changes[i].item()
                target_neuron = (first_real_layer_name, target_neuron_idx)
                actual_source_index = (
                    source_neuron_index if source_neuron_index is not None else 0
                )
                input_neuron = ("input_source", actual_source_index)

                if change > activation_threshold:
                    intermediate_pathways.append((input_neuron, target_neuron, change))
                    if target_neuron not in visited_neurons:
                        queue.append(target_neuron)
                        visited_neurons.add(target_neuron)
        else:
            print(
                f"Warning: No elements found in activation changes for layer '{first_real_layer_name}'. Cannot initialize tracing."
            )

    # --- 4. Process Queue (Subsequent Layers) ---
    while queue:
        current_neuron = queue.popleft()
        processed_sources.add(current_neuron)
        current_layer_name, current_neuron_idx = current_neuron
        current_layer_idx_in_guidelines = ordered_layer_names.index(current_layer_name)

        if current_layer_idx_in_guidelines >= len(ordered_layer_names) - 1:
            continue

        target_layer_name = ordered_layer_names[current_layer_idx_in_guidelines + 1]

        if current_layer_name not in activations_source:
            print(
                f"Warning: Source activation for layer {current_layer_name} not found. Skipping patching from this node."
            )
            continue

        source_activation_layer = activations_source[current_layer_name][0]

        # --- 4a. Perform Patching Experiment ---
        patched_activations = {}
        patch_handles, current_layer_modules = [], {}

        def get_patch_hook(
            layer_to_patch: str, neuron_idx_to_patch: int, value_to_patch: torch.Tensor
        ) -> Callable:
            def patch_hook_fn(module: nn.Module, input: Any, output: Any):
                module_name = None
                for (
                    name,
                    mod,
                ) in layer_modules.items():
                    if mod == module:
                        module_name = name
                        break

                if module_name == layer_to_patch:
                    original_output = output[0] if isinstance(output, tuple) else output
                    new_output = original_output.clone()

                    new_output.view(-1)[neuron_idx_to_patch] = value_to_patch
                    # print(f"Patched {module_name} neuron {neuron_idx_to_patch}") # Debug
                    return (
                        (new_output,) + output[1:]
                        if isinstance(output, tuple)
                        else new_output
                    )
                elif module_name in ordered_layer_names:
                    activation_tensor = (
                        output[0] if isinstance(output, tuple) else output
                    )
                    if isinstance(activation_tensor, torch.Tensor):
                        patched_activations[module_name] = (
                            activation_tensor.detach().clone()
                        )
                return output

            return patch_hook_fn

        patch_value = source_activation_layer.view(-1)[current_neuron_idx]
        try:
            current_module_to_patch = model.get_submodule(current_layer_name)
            patch_hook_handle = current_module_to_patch.register_forward_hook(
                get_patch_hook(current_layer_name, current_neuron_idx, patch_value)
            )
            patch_handles.append(patch_hook_handle)
        except AttributeError:
            print(
                f"Error: Could not find module {current_layer_name} to apply patch hook."
            )
            continue

        for name in ordered_layer_names:
            if name != current_layer_name:
                try:
                    module = model.get_submodule(name)
                    capture_hook_handle = module.register_forward_hook(
                        get_patch_hook(
                            current_layer_name, current_neuron_idx, patch_value
                        )
                    )
                    patch_handles.append(capture_hook_handle)
                except AttributeError:
                    pass

        with torch.no_grad():
            try:
                _ = model(base_input, torch.zeros_like(base_input[:, :1]))
            except TypeError:
                _ = model(base_input)
        remove_hooks(patch_handles)

        # --- 4b. Compare Activations ---
        if target_layer_name not in patched_activations:
            print(
                f"Warning: Target layer {target_layer_name} activation not captured during patch of {current_layer_name}. Skipping."
            )
            continue
        if target_layer_name not in activations_base:
            print(
                f"Warning: Target layer {target_layer_name} activation not found in base run. Skipping."
            )
            continue

        base_target_act = activations_base[target_layer_name][0]
        patched_target_act = patched_activations[target_layer_name][0]
        activation_changes = torch.abs(patched_target_act - base_target_act)

        # --- 4c. Find Top K and Recurse ---
        if target_layer_name not in layer_sizes:
            print(
                f"Warning: Size for target layer {target_layer_name} not determined. Skipping recursion."
            )
            continue

        flat_changes = activation_changes.flatten()
        num_elements = flat_changes.numel()
        current_top_k = min(top_k, num_elements)

        found_propagation = False
        if num_elements > 0:
            top_k_flat_indices = torch.argsort(flat_changes, descending=True)[
                :current_top_k
            ]
            top_k_changes = flat_changes[top_k_flat_indices]

            for i in range(len(top_k_flat_indices)):
                target_neuron_idx = int(top_k_flat_indices[i].item())  # Use flat index
                change = top_k_changes[i].item()
                target_neuron = (target_layer_name, target_neuron_idx)

                if change > activation_threshold:
                    found_propagation = True
                    intermediate_pathways.append(
                        (current_neuron, target_neuron, change)
                    )
                    if target_neuron not in visited_neurons:
                        visited_neurons.add(target_neuron)
                        queue.append(target_neuron)
        else:
            print(
                f"Warning: Activation changes for layer {target_layer_name} are empty after patching {current_layer_name}. Cannot recurse."
            )

        if found_propagation:
            propagating_sources.add(current_neuron)

    # --- 5. Finalize Output list ---
    final_output_list = []
    for source, target, change in intermediate_pathways:
        if source:
            final_output_list.append((source, target, change, False))
    failed_sources = processed_sources - propagating_sources
    for node in failed_sources:
        if node:
            final_output_list.append((node, None, None, True))

    return final_output_list


def get_pathways_for_input_neuron(
    model: nn.Module,
    input_dim: int,
    neuron_idx: int,
    top_k: int,
    activation_threshold: float,
    guidelines: dict,
    base_value: float = 0.0,
    source_value: float = 1.0,
) -> list[
    tuple[Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool]
]:
    """
    Traces causal pathways originating from a single input neuron activation.
    """
    base_input = torch.full((1, input_dim), base_value, dtype=torch.float32)

    source_input = base_input.clone()
    source_input[0, neuron_idx] = source_value

    results = trace_causal_pathways(
        model=model,
        base_input=base_input,
        source_input=source_input,
        top_k=top_k,
        activation_threshold=activation_threshold,
        source_neuron_index=neuron_idx,
        causal_pathways_guidelines=guidelines,
    )
    return results


def extract_nodes_from_pathways(
    pathway_results: list[
        tuple[
            Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool
        ]
    ],
) -> set[tuple[str, int]]:
    """Extracts unique intermediate/target nodes from pathway results."""
    nodes = set()
    for source, target, _, is_failed_source in pathway_results:
        if source and source[0] != "input_source" and not is_failed_source:
            nodes.add(source)
        if target and not is_failed_source:
            nodes.add(target)
        elif source and is_failed_source:
            nodes.add(source)

    nodes.discard(None)
    return nodes


def calculate_pathway_overlap(
    nodes1: set[tuple[str, int]], nodes2: set[tuple[str, int]]
) -> float:
    """
    Calculates the Jaccard index between two sets of pathway nodes.
    """
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    return intersection / union if union > 0 else 0.0


def analyze_input_overlaps(
    model: nn.Module,
    input_dim: int,
    top_k: int,
    activation_threshold: float,
    guidelines: dict,
    show_plot: bool = True,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Calculates and optionally plots the overlap between pathways from each input neuron.

    Args:
        model: The neural network model
        input_dim: Dimension of the input
        top_k: Number of top nodes to consider in pathway tracing
        activation_threshold: Threshold for activation changes
        guidelines: Model structure guidelines
        show_plot: Whether to display the overlap heatmap
        output_dir: Directory to save the plot (if None, plot won't be saved)

    Returns:
        overlap_matrix: Matrix containing Jaccard indices between pathways
    """
    print("\n--- Analyzing Pathway Overlaps ---")
    all_pathways_nodes = {}
    for i in range(input_dim):
        print(f"  Tracing for input neuron {i}...")
        results = get_pathways_for_input_neuron(
            model, input_dim, i, top_k, activation_threshold, guidelines
        )
        all_pathways_nodes[i] = extract_nodes_from_pathways(results)
        print(f"    Found {len(all_pathways_nodes[i])} unique nodes.")

    overlap_matrix = np.zeros((input_dim, input_dim))
    for i in range(input_dim):
        for j in range(input_dim):
            overlap = calculate_pathway_overlap(
                all_pathways_nodes[i], all_pathways_nodes[j]
            )
            overlap_matrix[i, j] = overlap

    if show_plot:
        plt.figure(figsize=(8, 6))
        tick_labels = [str(i) for i in range(input_dim)]
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        plt.xlabel("Input Neuron Index")
        plt.ylabel("Input Neuron Index")
        plt.title("Pathway Overlap (Jaccard Index)")

        # Save the plot if output_dir is provided
        if output_dir is not None:
            import os

            output_path = os.path.join(output_dir, "overlap_matrix.png")
            plt.savefig(output_path)
            print(f"  Saved overlap matrix to {output_path}")

        plt.tight_layout()
        plt.show()

    return overlap_matrix


def calculate_pathway_quality(
    pathway_results: list[
        tuple[
            Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool
        ]
    ],
    guideline_layers: list[str],
) -> dict[str, float]:
    """
    Calculates quality components for a set of pathways.
    Higher scores are better.

    Args:
        pathway_results: The output from trace_causal_pathways.
        guideline_layers: The list of layer names from the model's guidelines.

    Returns:
        A dictionary containing 'sparsity' and 'success' scores.
    """
    nodes_per_layer = defaultdict(set)
    pathway_edges = []
    failed_sources = set()
    total_sources_traced = set()

    for source, target, change, is_failed_source in pathway_results:
        if is_failed_source:
            if source:
                failed_sources.add(source)
                total_sources_traced.add(source)
                nodes_per_layer[source[0]].add(source[1])
        elif source and target and change is not None:
            if source[0] != "input_source":
                nodes_per_layer[source[0]].add(source[1])
                total_sources_traced.add(source)
            nodes_per_layer[target[0]].add(target[1])
            pathway_edges.append((source, target, change))

    # 1. Sparsity Score: Average inverse layer width
    sparsity_score = 0.0
    num_active_layers = 0
    relevant_layers = [layer for layer in guideline_layers if layer in nodes_per_layer]

    if relevant_layers:
        for layer_name in relevant_layers:
            layer_width = len(nodes_per_layer[layer_name])
            if layer_width > 0:
                sparsity_score += 1.0 / layer_width
                num_active_layers += 1
        if num_active_layers > 0:
            sparsity_score /= num_active_layers  # Average

    # 2. Success Score: Fraction of traced sources that did *not* fail
    num_failed = len(failed_sources)
    num_traced = len(total_sources_traced)
    success_score = (num_traced - num_failed) / num_traced if num_traced > 0 else 1.0

    return {"sparsity": sparsity_score, "success": success_score}


def analyze_pathway_quality(
    model: nn.Module,
    input_dim: int,
    top_k: int,
    activation_threshold: float,
    guidelines: dict,
    output_dir: Optional[str] = None,
) -> dict[int, dict[str, float]]:
    """
    Calculates the quality score components for pathways originating from each input neuron.

    Args:
        model: The neural network model
        input_dim: Dimension of the input
        top_k: Number of top nodes to consider in pathway tracing
        activation_threshold: Threshold for activation changes
        guidelines: Model structure guidelines
        output_dir: Directory to save the plot (if None, plot won't be saved)

    Returns:
        Dictionary mapping input neuron indices to their quality score components
    """
    print("\n--- Analyzing Pathway Quality ---")
    all_quality_components = {}
    guideline_layers = guidelines.get("ordered_layer_names", [])

    for i in range(input_dim):
        print(f"  Analyzing quality for input neuron {i}...")
        results = get_pathways_for_input_neuron(
            model,
            input_dim,
            i,
            top_k,
            activation_threshold,
            guidelines,
        )
        if not results:
            print("    No pathways found.")
            quality_components = {"sparsity": 0.0, "success": 0.0}
        else:
            quality_components = calculate_pathway_quality(results, guideline_layers)
            print(
                f"    Calculated Sparsity: {quality_components['sparsity']:.4f}, Success: {quality_components['success']:.4f}"
            )
        all_quality_components[i] = quality_components

    plt.figure(figsize=(10, 5))

    neuron_indices = list(all_quality_components.keys())
    sparsity_scores = [qc["sparsity"] for qc in all_quality_components.values()]
    success_scores = [qc["success"] for qc in all_quality_components.values()]

    plt.bar(neuron_indices, sparsity_scores, label="Sparsity Score")
    plt.bar(
        neuron_indices, success_scores, bottom=sparsity_scores, label="Success Score"
    )

    plt.xlabel("Input Neuron Index")
    plt.ylabel("Pathway Quality Score Components")
    plt.title("Decomposed Pathway Quality per Input Neuron")
    plt.xticks(range(input_dim))
    plt.grid(axis="y", linestyle="--")
    plt.legend()

    if output_dir is not None:
        import os

        output_path = os.path.join(output_dir, "pathway_quality.png")
        plt.savefig(output_path)
        print(f"  Saved pathway quality plot to {output_path}")

    plt.tight_layout()
    plt.show()

    return all_quality_components


def multiplot_pathway_analyses(
    model: nn.Module,
    input_dim: int,
    top_k_values: list[int],
    activation_threshold: float,
    guidelines: dict,
    output_dir: Optional[str] = None,
    show_plots: bool = True,
):
    """
    Creates multiplots comparing pathway overlaps and quality metrics for different top_k values.

    Args:
        model: The neural network model
        input_dim: Dimension of the input
        top_k_values: List of top_k values to compare (e.g., [5, 10, 15, 20])
        activation_threshold: Threshold for activation changes
        guidelines: Model structure guidelines
        output_dir: Directory to save the plots (if None, plots won't be saved)
        show_plots: Whether to display the plots
    """
    if len(top_k_values) == 0:
        raise ValueError("top_k_values must contain at least one value")

    # --- First plot: Overlap Matrices ---
    plt.figure(figsize=(5 * len(top_k_values), 5))
    gs = gridspec.GridSpec(1, len(top_k_values))

    overlap_matrices = []

    print("\n--- Generating Pathway Overlap Multiplots ---")
    for i, top_k in enumerate(top_k_values):
        print(f"\nAnalyzing overlaps with top_k={top_k}...")
        overlap_matrix = analyze_input_overlaps(
            model=model,
            input_dim=input_dim,
            top_k=top_k,
            activation_threshold=activation_threshold,
            guidelines=guidelines,
            show_plot=False,
        )
        overlap_matrices.append(overlap_matrix)

        ax = plt.subplot(gs[i])
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=[str(j) for j in range(input_dim)],
            yticklabels=[str(j) for j in range(input_dim)],
            ax=ax,
        )
        ax.set_title(f"top_k = {top_k}")
        ax.set_xlabel("Input Neuron Index")
        ax.set_ylabel("Input Neuron Index")

    plt.suptitle("Pathway Overlap Comparison for Different top_k Values", fontsize=16)
    plt.tight_layout()

    if output_dir is not None:
        overlap_path = os.path.join(output_dir, "overlap_matrices_comparison.png")
        plt.savefig(overlap_path, dpi=300, bbox_inches="tight")
        print(f"  Saved overlap matrices comparison to {overlap_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # --- Second plot: Quality Metrics ---
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    all_quality_results = []

    print("\n--- Generating Pathway Quality Multiplots ---")
    for i, top_k in enumerate(top_k_values):
        print(f"\nAnalyzing quality with top_k={top_k}...")
        quality_results = {}
        guideline_layers = guidelines.get("ordered_layer_names", [])

        for j in range(input_dim):
            print(f"  Input neuron {j}...")
            results = get_pathways_for_input_neuron(
                model=model,
                input_dim=input_dim,
                neuron_idx=j,
                top_k=top_k,
                activation_threshold=activation_threshold,
                guidelines=guidelines,
            )
            if not results:
                quality_components = {"sparsity": 0.0, "success": 0.0}
            else:
                quality_components = calculate_pathway_quality(
                    results, guideline_layers
                )
            quality_results[j] = quality_components

        all_quality_results.append((top_k, quality_results))

    ax1 = plt.subplot(gs[0])
    for top_k, quality_results in all_quality_results:
        neuron_indices = list(quality_results.keys())
        sparsity_scores = [qc["sparsity"] for qc in quality_results.values()]
        ax1.plot(neuron_indices, sparsity_scores, marker="o", label=f"top_k = {top_k}")

    ax1.set_title("Sparsity Scores Comparison")
    ax1.set_xlabel("Input Neuron Index")
    ax1.set_ylabel("Sparsity Score")
    ax1.set_xticks(range(input_dim))
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2 = plt.subplot(gs[1])
    for top_k, quality_results in all_quality_results:
        neuron_indices = list(quality_results.keys())
        success_scores = [qc["success"] for qc in quality_results.values()]
        ax2.plot(neuron_indices, success_scores, marker="o", label=f"top_k = {top_k}")

    ax2.set_title("Success Scores Comparison")
    ax2.set_xlabel("Input Neuron Index")
    ax2.set_ylabel("Success Score")
    ax2.set_xticks(range(input_dim))
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.suptitle(
        "Pathway Quality Metrics Comparison for Different top_k Values", fontsize=16
    )
    plt.tight_layout()

    if output_dir is not None:
        quality_path = os.path.join(output_dir, "quality_metrics_comparison.png")
        plt.savefig(quality_path, dpi=300, bbox_inches="tight")
        print(f"  Saved quality metrics comparison to {quality_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    print("\nMultiplot analysis complete.")
