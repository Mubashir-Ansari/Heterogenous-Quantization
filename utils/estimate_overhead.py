import os
import torch
import pandas as pd


# ============================================================
# A. Parameter & layer analysis utilities
# ============================================================

def get_nested_attr(obj, attr):
    """
    Resolve things like 'conv1[0]' or 'features[4]' or 'classifier[6]'.
    """
    try:
        parts = attr.split(".")
        for part in parts:
            if "[" in part and "]" in part:
                part, idx = part.split("[")
                idx = int(idx[:-1])
                obj = getattr(obj, part)[idx]
            else:
                obj = getattr(obj, part)
        return obj
    except AttributeError as e:
        print(f"Error: {e}")
        return None


def count_layer_weights(layer):
    """
    Count number of weights (parameters) in a layer.
    Works for Conv2d and Linear layers.
    """
    if hasattr(layer, "weight") and layer.weight is not None:
        return layer.weight.numel()
    return 0


def analyze_model_layers(model, layer_list, protected_layers):
    """
    Returns detailed statistics for a given model.
    """

    total_params = 0
    protected_params = 0
    conv_layers = 0
    fc_layers = 0

    for name in layer_list:
        layer = get_nested_attr(model, name)
        w = count_layer_weights(layer)
        total_params += w

        if isinstance(layer, torch.nn.Conv2d):
            conv_layers += 1
        elif isinstance(layer, torch.nn.Linear):
            fc_layers += 1

        if name in protected_layers:
            protected_params += w

    stats = {
        "Total layers": len(layer_list),
        "Conv layers": conv_layers,
        "FC layers": fc_layers,
        "Protected layers": len(protected_layers),
        "Total parameters": total_params,
        "Protected parameters": protected_params,
    }

    return stats


# ============================================================
# B. REQAP / DMR overhead estimation
# ============================================================

def estimate_reqap_overhead(
    total_params,
    protected_params,
    replication_factor_protected=2,
    replication_factor_unprotected=1,
    bits_per_weight=8,
):
    """
    Estimate memory and gate overhead for REQAP / DMR.

    Assumptions:
    - Protected layers use DMR => replication factor = 2
    - Unprotected layers stored normally => factor = 1
    - Weight bitwidth = bits_per_weight (e.g. 8-bit quantization)

    Memory Overhead Formula:
    ------------------------
    baseline_memory = total_params * bits_per_weight

    reqap_memory =
        (protected_params * bits_per_weight * replication_factor_protected)
      + ((total_params - protected_params) * bits_per_weight)

    overhead (%) =
        (reqap_memory - baseline_memory) / baseline_memory * 100

    Gate Overhead (Estimated):
    --------------------------
    For DMR:
      - Comparator per weight bit
      - Approx: bits_per_weight XOR + OR gates

    gate_overhead ≈ protected_params * bits_per_weight
    """

    baseline_memory = total_params * bits_per_weight

    reqap_memory = (
        protected_params * bits_per_weight * replication_factor_protected
        + (total_params - protected_params) * bits_per_weight
    )

    memory_overhead_pct = (
        (reqap_memory - baseline_memory) / baseline_memory
    ) * 100

    estimated_gate_overhead = protected_params * bits_per_weight

    return {
        "Baseline memory (bits)": baseline_memory,
        "REQAP memory (bits)": reqap_memory,
        "Memory overhead (%)": memory_overhead_pct,
        "Estimated gate overhead": estimated_gate_overhead,
    }


# ============================================================
# C. Create report table for a model
# ============================================================

def generate_model_report(
    model_name,
    model,
    layer_list,
    protected_layers,
    output_csv="model_reqap_report.csv",
):
    """
    Generate a single-row report for the given model
    and append it to a CSV file.
    """

    layer_stats = analyze_model_layers(
        model, layer_list, protected_layers
    )

    overhead_stats = estimate_reqap_overhead(
        total_params=layer_stats["Total parameters"],
        protected_params=layer_stats["Protected parameters"],
    )

    report_row = {
        "Model": model_name,
        **layer_stats,
        **overhead_stats,
    }

    df = pd.DataFrame([report_row])

    # Append if file exists, else create
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)

    print("\n===== MODEL REQAP REPORT =====")
    print(df.to_string(index=False))

    return df
