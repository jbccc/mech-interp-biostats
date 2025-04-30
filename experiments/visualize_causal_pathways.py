import sys
import traceback

sys.path.append(".")

import os
import sys
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

from src.data.no_effect_confounders_data import \
    generate_no_effect_confounders_data
from src.models.toy_tmle_no_effect_confounder import ToyTMLENoEffectConfounder
from src.utils.causal_pathways_utils import (analyze_input_overlaps,
                                             get_pathways_for_input_neuron)
from src.utils.toy_tmle_model import ToyTMLE

OUTPUT_DIR = "exp3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR_INDIVIDUAL = os.path.join(OUTPUT_DIR, "individual")
OUTPUT_DIR_COMBINED = os.path.join(OUTPUT_DIR, "combined")
os.makedirs(OUTPUT_DIR_INDIVIDUAL, exist_ok=True)
os.makedirs(OUTPUT_DIR_COMBINED, exist_ok=True)


def get_model_layer_info(
    model: ToyTMLE,
) -> tuple[list[str], dict[str, int], dict[str, Any]]:
    """
    Extracts ordered layer names, sizes, and guidelines from ToyTMLE.
    """
    if not isinstance(model, ToyTMLE):
        raise TypeError("Model must be an instance of ToyTMLE.")

    input_dim = model.input_dim
    ordered_module_names = []
    layer_sizes = {"input": input_dim}

    if hasattr(model, "shared") and isinstance(model.shared, nn.Sequential):
        current_dim = input_dim
        for i, layer in enumerate(model.shared):
            layer_name = f"shared.{i}"
            if isinstance(layer, nn.Linear):
                ordered_module_names.append(layer_name)
                layer_sizes[layer_name] = layer.out_features
                current_dim = layer.out_features
            elif isinstance(layer, nn.Dropout):
                ordered_module_names.append(layer_name)
                layer_sizes[layer_name] = current_dim
            elif isinstance(layer, nn.ReLU):
                ordered_module_names.append(layer_name)
                layer_sizes[layer_name] = current_dim
    else:
        print(
            "Warning: Model does not have a 'shared' nn.Sequential attribute. Cannot extract layer info."
        )

    ordered_layers_with_input = ["input"] + ordered_module_names

    guidelines = {
        "input_dim": input_dim,
        "ordered_layer_names": ordered_module_names,
    }

    return ordered_layers_with_input, layer_sizes, guidelines


def plot_causal_graph_from_trace(
    trace_results: list[
        tuple[
            Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool
        ]
    ],
    ordered_layers: list[str],
    layer_sizes: dict[str, int],
    output_filename: str = "causal_pathway_visualization",
    edge_color: str = "blue",
    min_edge_width: float = 0.2,
    max_edge_width: float = 2.5,
    title_suffix: str = "",
):
    """
    Generates a plot of the causal graph from specific trace_results,
    showing nodes based on model structure and edges from the trace.

    Args:
        trace_results: Output from trace_causal_pathways for a specific source.
        ordered_layers: List of layer names in order (including "input").
        layer_sizes: Dictionary mapping layer names to their sizes (must be provided).
        output_filename: Base filename for saving the plot.
        edge_color: The color to use for drawing edges from this trace.
        min_edge_width: Minimum width for edges.
        max_edge_width: Maximum width for edges (corresponding to max absolute change).
        title_suffix: Optional suffix to add to the plot title.
    """
    if not layer_sizes:
        raise ValueError("layer_sizes must be provided.")

    G = nx.DiGraph()
    layer_to_index = {layer: i for i, layer in enumerate(ordered_layers)}
    nodes_by_layer = defaultdict(list)
    positions = {}
    node_colors = {}
    node_sizes = {}
    node_labels = {}

    node_cmap = get_cmap("Blues")

    # --- Node Generation based on Layer Structure ---
    for layer_name in ordered_layers:
        if layer_name not in layer_sizes:
            continue
        size = layer_sizes[layer_name]
        if not isinstance(size, int) or size <= 0:
            continue

        current_layer_nodes = []
        for i in range(size):
            node = (layer_name, i)
            G.add_node(node)
            current_layer_nodes.append((node, i))

            layer_idx = layer_to_index.get(layer_name, 0)
            if layer_name == "input":
                node_colors[node] = "red"
                node_sizes[node] = 400
                node_labels[node] = f"In{i}"
            else:
                layer_pos_norm = layer_idx / max(1, len(ordered_layers) - 1)
                color_val = min(1.0, max(0.0, 0.3 + 0.7 * layer_pos_norm))
                node_colors[node] = node_cmap(color_val)
                node_sizes[node] = 250
                layer_label_short = layer_name.split(".")[-1]
                node_labels[node] = f"L{layer_label_short}N{i}"

        nodes_by_layer[layer_name] = sorted(current_layer_nodes, key=lambda x: x[1])

    # --- Optional: Mark Failed Sources from Trace Results ---
    failed_sources = set()
    processed_sources = set()
    for source, _target, _change, is_failed_source_marker in trace_results:
        if is_failed_source_marker and source and source not in processed_sources:
            if source[0] == "input_source":
                source = ("input", source[1])

            if source in G:
                failed_sources.add(source)
                node_colors[source] = "grey"
                node_sizes[source] = 150
                if source in node_labels:
                    node_labels[source] += " (Fail)"
                else:
                    node_labels[source] = f"{source[0]}{source[1]} (Fail)"
            processed_sources.add(source)

    # --- Calculate Node Positions ---
    max_nodes_in_any_layer = 1
    if layer_sizes:
        valid_layer_names = [lname for lname in nodes_by_layer if nodes_by_layer[lname]]
        if valid_layer_names:
            max_nodes_in_any_layer = max(
                layer_sizes.get(lname, 1) for lname in valid_layer_names
            )
        else:
            max_nodes_in_any_layer = 1
    else:
        max_nodes_in_any_layer = (
            max(len(nodes) for nodes in nodes_by_layer.values() if nodes)
            if nodes_by_layer
            else 1
        )

    vertical_spacing = 1.0 / max(1, max_nodes_in_any_layer)

    for layer_name, layer_nodes in nodes_by_layer.items():
        layer_idx = layer_to_index.get(layer_name, 0)
        num_nodes_in_layer = len(layer_nodes)
        if num_nodes_in_layer == 0:
            continue

        total_height = (num_nodes_in_layer - 1) * vertical_spacing
        start_y = 0.5 + total_height / 2.0

        for i, (node, _) in enumerate(layer_nodes):
            y_pos = start_y - i * vertical_spacing
            positions[node] = (layer_idx, y_pos)

    # --- Prepare lists for drawing nodes ---
    draw_nodes_list = list(G.nodes())
    draw_node_colors = [node_colors.get(n, "purple") for n in draw_nodes_list]
    draw_node_sizes = [node_sizes.get(n, 250) for n in draw_nodes_list]

    # --- Process Edges from Trace Results ---
    edges_to_draw = []
    edge_weights = []
    abs_changes = []

    for source, target, change, is_failed_source in trace_results:
        if not is_failed_source and source and target and change is not None:
            if source[0] == "input_source":
                source = ("input", source[1])

            if source in G and target in G:
                G.add_edge(source, target, weight=change)
                edges_to_draw.append((source, target))
                edge_weights.append(change)
                abs_changes.append(abs(change))

    # --- Normalize edge weights for thickness ---
    edge_widths = []
    if abs_changes:
        min_abs_chg = min(abs_changes)
        max_abs_chg = max(abs_changes)
        if max_abs_chg > min_abs_chg:
            norm_factor = max_abs_chg - min_abs_chg
            normalized_widths = [
                (chg - min_abs_chg) / norm_factor for chg in abs_changes
            ]
            edge_widths = [
                min_edge_width + w * (max_edge_width - min_edge_width)
                for w in normalized_widths
            ]
        elif max_abs_chg > 0:
            edge_widths = [(min_edge_width + max_edge_width) / 2.0] * len(abs_changes)
        else:
            edge_widths = [min_edge_width] * len(abs_changes)
    else:
        edge_widths = []

    # --- Drawing ---
    plt.figure(
        figsize=(
            max(10, len(ordered_layers) * 2.5),
            max(8, max_nodes_in_any_layer * 0.6),
        )
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=draw_nodes_list,
        node_size=draw_node_sizes,
        node_color=draw_node_colors,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )

    # Draw edges if any exist
    if edges_to_draw:
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=edges_to_draw,
            width=edge_widths,
            edge_color=edge_color,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
        )

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        positions,
        labels=node_labels,
        font_size=7,
        font_color="black",
        font_family="sans-serif",
    )

    # Add layer labels at the top
    for layer_name, layer_idx in layer_to_index.items():
        display_name = layer_name
        if layer_name.startswith("network."):
            display_name = f"L{layer_name.split('.')[-1]}"
        elif layer_name == "input":
            display_name = "Input"
        plt.text(
            layer_idx,
            plt.gca().get_ylim()[1] * 1.05,
            display_name,
            horizontalalignment="center",
            fontweight="bold",
        )

    # Add a legend for node types
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Input Neuron",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=node_cmap(0.7),
            markersize=8,
            label="Hidden Neuron",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markersize=6,
            label="Failed Source",
        ),
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=8,
    )

    plt.axis("off")
    title = f"Causal Pathway Visualization"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title, fontsize=14)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    # Save figure
    save_path_png = f"{output_filename}.png"
    try:
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        print(f"  -> Saved graph to {save_path_png}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()


def plot_combined_pathways(
    trace1: list[
        tuple[
            Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool
        ]
    ],
    trace2: list[
        tuple[
            Optional[tuple[str, int]], Optional[tuple[str, int]], Optional[float], bool
        ]
    ],
    color1: str,
    color2: str,
    ordered_layers: list[str],
    layer_sizes: dict[str, int],
    output_filename: str,
    min_edge_width: float = 0.2,
    max_edge_width: float = 2.5,
    title_suffix: str = "",
    input_neuron_index1: Optional[int] = None,
    input_neuron_index2: Optional[int] = None,
):
    """
    Plots two causal pathways on the same graph with different colors.
    """

    def get_edges_and_widths(trace, graph):
        edges_to_draw = []
        edge_weights = []
        abs_changes = []

        for source, target, change, is_failed_source in trace:
            if not is_failed_source and source and target and change is not None:
                if source[0] == "input_source":
                    source = ("input", source[1])

                if source in graph and target in graph:
                    graph.add_edge(source, target, weight=change)
                    edges_to_draw.append((source, target))
                    edge_weights.append(change)
                    abs_changes.append(abs(change))

        edge_widths = []
        if abs_changes:
            min_abs_chg = min(abs_changes)
            max_abs_chg = max(abs_changes)
            if max_abs_chg > min_abs_chg:
                norm_factor = max_abs_chg - min_abs_chg
                normalized_widths = [
                    (chg - min_abs_chg) / norm_factor for chg in abs_changes
                ]
                edge_widths = [
                    min_edge_width + w * (max_edge_width - min_edge_width)
                    for w in normalized_widths
                ]
            elif max_abs_chg > 0:
                edge_widths = [(min_edge_width + max_edge_width) / 2.0] * len(
                    abs_changes
                )
            else:
                edge_widths = [min_edge_width] * len(abs_changes)

        return edges_to_draw, edge_widths

    G = nx.DiGraph()
    layer_to_index = {layer: i for i, layer in enumerate(ordered_layers)}
    nodes_by_layer = defaultdict(list)
    positions = {}
    node_colors = {}
    node_sizes = {}
    node_labels = {}
    node_cmap = get_cmap("Blues")

    for layer_name in ordered_layers:
        if layer_name not in layer_sizes:
            continue
        size = layer_sizes[layer_name]
        if not isinstance(size, int) or size <= 0:
            continue
        current_layer_nodes = []
        for i in range(size):
            node = (layer_name, i)
            G.add_node(node)
            current_layer_nodes.append((node, i))
            layer_idx = layer_to_index.get(layer_name, 0)
            if layer_name == "input":
                node_colors[node] = "red"
                node_sizes[node] = 400
                node_labels[node] = f"In{i}"
            else:
                layer_pos_norm = layer_idx / max(1, len(ordered_layers) - 1)
                color_val = min(1.0, max(0.0, 0.3 + 0.7 * layer_pos_norm))
                node_colors[node] = node_cmap(color_val)
                node_sizes[node] = 250
                layer_label_short = layer_name.split(".")[-1]
                node_labels[node] = f"L{layer_label_short}N{i}"
        nodes_by_layer[layer_name] = sorted(current_layer_nodes, key=lambda x: x[1])

    # --- Identify Overlapping Nodes ---
    nodes_in_trace1 = set()
    for source, target, _change, is_failed in trace1:
        if not is_failed and target:
            if source and source[0] == "input_source":
                source = ("input", source[1])
            if source in G:
                nodes_in_trace1.add(source)
            if target in G:
                nodes_in_trace1.add(target)

    nodes_in_trace2 = set()
    for source, target, _change, is_failed in trace2:
        if not is_failed and target:
            if source and source[0] == "input_source":
                source = ("input", source[1])
            if source in G:
                nodes_in_trace2.add(source)
            if target in G:
                nodes_in_trace2.add(target)

    overlapping_nodes = nodes_in_trace1.intersection(nodes_in_trace2)

    for node in overlapping_nodes:
        node_colors[node] = "lightgreen"

    # --- Process Failed Sources ---
    failed_sources_combined = set()
    for source, _target, _change, is_failed in trace1 + trace2:
        if is_failed and source:
            if source[0] == "input_source":
                source = ("input", source[1])
            if source in G:
                failed_sources_combined.add(source)

    for node in failed_sources_combined:
        node_colors[node] = "grey"
        node_sizes[node] = 150
        if node in node_labels:
            node_labels[node] += " (Fail)"
        else:
            node_labels[node] = f"{node[0]}{node[1]} (Fail)"

    max_nodes_in_any_layer = max(
        (
            layer_sizes.get(lname, 1)
            for lname in nodes_by_layer
            if nodes_by_layer[lname]
        ),
        default=1,
    )
    vertical_spacing = 1.0 / max(1, max_nodes_in_any_layer)
    for layer_name, layer_nodes in nodes_by_layer.items():
        layer_idx = layer_to_index.get(layer_name, 0)
        num_nodes_in_layer = len(layer_nodes)
        if num_nodes_in_layer == 0:
            continue
        total_height = (num_nodes_in_layer - 1) * vertical_spacing
        start_y = 0.5 + total_height / 2.0
        for i, (node, _) in enumerate(layer_nodes):
            y_pos = start_y - i * vertical_spacing
            positions[node] = (layer_idx, y_pos)

    # --- Prepare for Drawing ---
    draw_nodes_list = list(G.nodes())
    draw_node_colors = [node_colors.get(n, "purple") for n in draw_nodes_list]
    draw_node_sizes = [node_sizes.get(n, 250) for n in draw_nodes_list]

    edges1, widths1 = get_edges_and_widths(trace1, G)
    edges2, widths2 = get_edges_and_widths(trace2, G)

    # --- Drawing ---
    plt.figure(
        figsize=(
            max(10, len(ordered_layers) * 2.5),
            max(8, max_nodes_in_any_layer * 0.6),
        )
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=draw_nodes_list,
        node_size=draw_node_sizes,
        node_color=draw_node_colors,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )

    # Draw edges
    if edges1:
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=edges1,
            width=widths1,
            edge_color=color1,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
        )

    if edges2:
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=edges2,
            width=widths2,
            edge_color=color2,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.05",
        )

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        positions,
        labels=node_labels,
        font_size=7,
        font_color="black",
        font_family="sans-serif",
    )

    # Add layer labels at the top
    for layer_name, layer_idx in layer_to_index.items():
        display_name = layer_name
        if layer_name.startswith("shared."):
            display_name = f"L{layer_name.split('.')[-1]}"
        elif layer_name == "input":
            display_name = "Input"
        plt.text(
            layer_idx,
            plt.gca().get_ylim()[1] * 1.05,
            display_name,
            horizontalalignment="center",
            fontweight="bold",
        )

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Input Neuron",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=node_cmap(0.7),
            markersize=8,
            label="Hidden Neuron",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgreen",
            markersize=8,
            label="Shared by Both Pathways",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markersize=6,
            label="Failed Source",
        ),
        Line2D(
            [0],
            [0],
            color=color1,
            lw=2,
            label=f"Pathway 1 {'(In' + str(input_neuron_index1) + ')' if input_neuron_index1 is not None else ''}",
        ),
        Line2D(
            [0],
            [0],
            color=color2,
            lw=2,
            label=f"Pathway 2 {'(In' + str(input_neuron_index2) + ')' if input_neuron_index2 is not None else ''}",
        ),
    ]

    plt.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=8,
    )

    plt.axis("off")
    title = f"Combined Causal Pathways"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title, fontsize=14)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    # Save figure
    try:
        plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches="tight")
        print(f"  -> Saved combined graph to {output_filename}.png")
    except Exception as e:
        print(f"Error saving combined plot {output_filename}: {e}")
    plt.close()


if __name__ == "__main__":
    TOP_K_TRACING = 10
    TOP_K_VALUES = [5, 10, 15, 20]
    SEED = 42
    ACTIVATION_THRESHOLD = 1e-6
    BASE_VALUE = 0.0
    SOURCE_VALUE = 1.0
    N_SAMPLES = 1000

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Load Data to get Input Dim ---
    print("Loading data (No Effect Confounders) to determine input dimension...")
    data, feature_cols = generate_no_effect_confounders_data(
        n_samples=N_SAMPLES, seed=SEED
    )
    INPUT_DIM = len(feature_cols)
    print(f"Input dimension determined: {INPUT_DIM}")

    # --- Create and Load Model ---
    print("Loading ToyTMLE model...")
    model_trainer = ToyTMLENoEffectConfounder()
    model = model_trainer.load_model()
    print(f"Model loaded successfully: {type(model).__name__}")

    # --- Get Model Layer Information ---
    print("Extracting model layer information...")
    ordered_layers, layer_sizes, guidelines = get_model_layer_info(model)
    print(f"Found {len(ordered_layers)} ordered layers:")
    for i, layer in enumerate(ordered_layers):
        size = layer_sizes.get(layer, "Unknown")
        print(f"  {i}. {layer} (size: {size})")

    print(f"Output directories setup:")
    print(f"  Main: {OUTPUT_DIR}")
    print(f"  Individual pathways: {OUTPUT_DIR_INDIVIDUAL}")
    print(f"  Combined pathways: {OUTPUT_DIR_COMBINED}")

    # --- 1. Generate and Visualize Individual Pathways ---
    print("-" * 30)
    print("Generating individual pathway visualizations...")
    all_input_traces = {}

    for i in range(INPUT_DIM):
        print(f"Processing Input Neuron {i}/{INPUT_DIM - 1}...")
        try:
            trace_results = get_pathways_for_input_neuron(
                model=model,
                input_dim=INPUT_DIM,
                neuron_idx=i,
                top_k=TOP_K_TRACING,
                activation_threshold=ACTIVATION_THRESHOLD,
                guidelines=guidelines,
                base_value=BASE_VALUE,
                source_value=SOURCE_VALUE,
            )
            all_input_traces[i] = trace_results
            print(f"  Found {len(trace_results)} pathway segments/markers.")

            if not trace_results:
                print("  No significant pathways found, skipping plot.")
                continue

            output_file_base = os.path.join(
                OUTPUT_DIR_INDIVIDUAL, f"pathways_input_{i}"
            )
            plot_causal_graph_from_trace(
                trace_results=trace_results,
                ordered_layers=ordered_layers,
                layer_sizes=layer_sizes,
                output_filename=output_file_base,
                edge_color="darkred",
                title_suffix=f"Input Neuron {i} (No Effect Confound Data)",
            )

        except Exception as e:
            print(f"ERROR processing input neuron {i}: {e}")
            import traceback

            traceback.print_exc()

    print("-" * 30)

    # --- 3. Find Best/Worst Overlap Pairs and Plot Combined ---
    overlap_matrix = None
    try:
        # Use TOP_K_TRACING value just for finding best/worst pairs
        overlap_matrix = analyze_input_overlaps(
            model=model,
            input_dim=INPUT_DIM,
            top_k=TOP_K_TRACING,
            activation_threshold=ACTIVATION_THRESHOLD,
            guidelines=guidelines,
            show_plot=False,
        )
    except Exception as e:
        print(f"Error in overlap analysis: {e}")
        print("Skipping combined pathway visualization")

    if overlap_matrix is not None and INPUT_DIM > 1:
        print("Finding best and worst overlapping input pairs...")
        max_overlap = -1.0
        min_overlap = 2.0
        best_pair = (-1, -1)
        worst_pair = (-1, -1)

        for i in range(INPUT_DIM):
            for j in range(i + 1, INPUT_DIM):
                overlap = overlap_matrix[i, j]
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_pair = (i, j)
                if overlap <= min_overlap:
                    min_overlap = overlap
                    worst_pair = (i, j)

        print(
            f"  Highest overlap ({max_overlap:.3f}) between Input {best_pair[0]} and {best_pair[1]}."
        )
        print(
            f"  Lowest overlap ({min_overlap:.3f}) between Input {worst_pair[0]} and {worst_pair[1]}."
        )

        # --- Plot Combined Pathways ---
        print("Plotting combined pathways for most overlapping pair...")
        if best_pair[0] >= 0 and best_pair[1] >= 0:
            if (
                best_pair[0] in all_input_traces
                and best_pair[1] in all_input_traces
                and all_input_traces[best_pair[0]]
                and all_input_traces[best_pair[1]]
            ):
                try:
                    best_output_file = os.path.join(
                        OUTPUT_DIR_COMBINED,
                        f"combined_high_overlap_{best_pair[0]}_{best_pair[1]}",
                    )
                    plot_combined_pathways(
                        trace1=all_input_traces[best_pair[0]],
                        trace2=all_input_traces[best_pair[1]],
                        color1="blue",
                        color2="red",
                        ordered_layers=ordered_layers,
                        layer_sizes=layer_sizes,
                        output_filename=best_output_file,
                        title_suffix=f"High Overlap ({max_overlap:.3f})",
                        input_neuron_index1=best_pair[0],
                        input_neuron_index2=best_pair[1],
                    )
                except Exception as e:
                    print(f"Error plotting best overlap pair: {e}")
            else:
                print(
                    f"  Missing trace data for best pair {best_pair}. Skipping combined plot."
                )

        print("Plotting combined pathways for least overlapping pair...")
        if worst_pair[0] >= 0 and worst_pair[1] >= 0:
            if (
                worst_pair[0] in all_input_traces
                and worst_pair[1] in all_input_traces
                and all_input_traces[worst_pair[0]]
                and all_input_traces[worst_pair[1]]
            ):
                try:
                    worst_output_file = os.path.join(
                        OUTPUT_DIR_COMBINED,
                        f"combined_low_overlap_{worst_pair[0]}_{worst_pair[1]}",
                    )
                    plot_combined_pathways(
                        trace1=all_input_traces[worst_pair[0]],
                        trace2=all_input_traces[worst_pair[1]],
                        color1="darkorange",
                        color2="purple",
                        ordered_layers=ordered_layers,
                        layer_sizes=layer_sizes,
                        output_filename=worst_output_file,
                        title_suffix=f"Low Overlap ({min_overlap:.3f})",
                        input_neuron_index1=worst_pair[0],
                        input_neuron_index2=worst_pair[1],
                    )
                except Exception as e:
                    print(f"Error plotting worst overlap pair: {e}")
            else:
                print(
                    f"  Missing trace data for worst pair {worst_pair}. Skipping combined plot."
                )

    print("-" * 30)
    print(f"Visualization tasks complete! All plots saved to directory: {OUTPUT_DIR}")
