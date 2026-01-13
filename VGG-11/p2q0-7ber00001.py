import os
import io
import csv
import timeit
import random
import numpy as np
import torch
import pandas

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# If you need VGG definitions explicitly; not strictly required for torch.load()
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

from fi07 import FI
from fi072 import FI2
from dmr import DMR
from dmr2 import DMR2

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.estimate_overhead import generate_model_report

# ============================================================
# 0. Reproducibility
# ============================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 1. Config
# ============================================================
Sufficient_no_faults = 50       # number of FI iterations
BER = 0.0003                    # bit error rate
BASELINE_ACC = 89.8337          # baseline accuracy of golden quantized VGG-11 (from paper/code)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acc_dict = {"noFI": [], "Accuracy": []}
csv_acc = "pvgg11-0-7-fi50-ber0003.csv"
fault_csv_path = "vgg11_fault_list50_ber0003.csv"

# Per-image output logging
output_results_file = open("poutput_vgg11_0-7_ber0003", "w", newline="")
output_results = csv.DictWriter(
    output_results_file,
    [
        "fault_id",
        "img_id",
        "predicted",
        *[f"p{i}" for i in range(10)],
    ],
)
output_results.writeheader()
output_results_file.close()

# Re-open in append mode
output_results_file = open("poutput_vgg11_0-7_ber0003", "a", newline="")
output_results = csv.DictWriter(
    output_results_file,
    [
        "fault_id",
        "img_id",
        "predicted",
        *[f"p{i}" for i in range(10)],
    ],
)

# ============================================================
# 2. CIFAR-10 test dataloader (no augmentation)
# ============================================================
def val_dataloader(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2471, 0.2435, 0.2616),
):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(
        root="../datasets/cifar10_data",
        train=False,
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=128,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    return dataloader


data = val_dataloader()

# ============================================================
# 3. Load quantized VGG-11 (REQAP)
# ============================================================
model = torch.load("qvgg11-0-7.pth", map_location=DEVICE, weights_only=False)
model.to(DEVICE)

# ============================================================
# 4. Layer lists & accessor
# ============================================================
layer_list = [
    "features[0]",
    "features[4]",
    "features[8]",
    "features[11]",
    "features[15]",
    "features[18]",
    "features[22]",
    "features[25]",
    "classifier[0]",
    "classifier[3]",
    "classifier[6]",
]

first = [
    "features[0]",
    "features[4]",
    "features[8]",
    "features[11]",
    "features[15]",
    "features[18]",
    "features[22]",
    "features[25]",
]

second = [
    "classifier[0]",
    "classifier[3]",
    "classifier[6]",
]

# Protected layers according to heterogeneous (3,5) scheme for VGG-11
protected_layers = [
    "features[0]",
    "features[4]",
    "classifier[6]",
]


def get_nested_attr(obj, attr):
    """
    Resolve attributes like 'features[0]' or 'classifier[6]'.
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
        print(f"Error resolving attr {attr}: {e}")
        return None


def reqap_activation_packing(act, layer_name="unknown", inject_fault=True):
    """
    Simulates REQAP-style activation packing:
    - Quantize activations (8-bit)
    - Duplicate for redundancy (DMR)
    - Optionally inject random bit flips
    - Correct via simple majority voting
    """

    # Safety check (recommended)
    assert isinstance(act, torch.Tensor), f"{layer_name} received non-tensor!"

    # 1️⃣ Quantize activations
    act_q = torch.clamp((act * 127).round(), -128, 127).to(torch.int8)

    # 2️⃣ Duplicate for redundancy
    act_copy1 = act_q.clone()
    act_copy2 = act_q.clone()

    # 3️⃣ Optional: Inject random bit flips
    if inject_fault:
        ber = 1e-5
        fault_mask = (torch.rand_like(act_copy1.float()) < ber)
        act_copy1 = act_copy1 ^ fault_mask.to(torch.int8)

    # 4️⃣ Correction (DMR-style)
    diff_mask = (act_copy1 != act_copy2)
    act_recovered = torch.where(diff_mask, act_copy2, act_copy1)

    # 5️⃣ De-quantize back to float
    act_recovered = act_recovered.float() / 127.0

    return act_recovered

# ============================================================
# 5. Apply DMR/DMR2 "set" to protected layers (initial TMR)
# ============================================================
for l in protected_layers:
    print("Set DMR for layer:", l)
    weights = get_nested_attr(model, l).weight._data
    if l in first:
        dmr = DMR(weights)
    else:
        dmr = DMR2(weights)
    new_weights = dmr.set()
    get_nested_attr(model, l).weight._data = new_weights

# Store original TMR-set weights for all layers we care about
original_tmr_weights = []
for name in layer_list:
    w = get_nested_attr(model, name).weight._data.clone()
    original_tmr_weights.append(w)

# ============================================================
# 6. Count "no_faults" (N) – as in original implementation
# ============================================================
def no_faults():
    number = []
    for name in layer_list:
        weights = get_nested_attr(model, name).weight._data
        if name in first:
            fi = FI(weights)
        else:
            fi = FI2(weights)
        nn = fi.param(weights)
        number.append(nn)
    total = sum(number) * 5  # multiplier as used in the original code
    return total


# ============================================================
# 7. Generate fault list CSV (like original GitHub version)
# ============================================================
# ============================================================
# 7. Generate fault list CSV (optimized)
# ============================================================
def generate_fault_list(n, ber, num_iterations, csv_path):
    """
    Generate a fault list for VGG-11 and save as CSV.
    Columns: Iteration, Layer, Index, Bit

    Optimized: reuses FI/FI2 objects per layer instead of recreating them
    for every single fault.
    """
    faults_per_iteration = int(ber * n)
    if faults_per_iteration <= 0:
        raise ValueError(
            f"Faults per iteration is {faults_per_iteration}, "
            f"check BER ({ber}) or n ({n})"
        )

    print(f"Total N (no_faults) = {n}")
    print(f"Faults per iteration: {faults_per_iteration}")
    print(f"Total faults over all iterations: {faults_per_iteration * num_iterations}")

    # ---- NEW: cache FI/FI2 objects per layer ----
    fi_cache = {}
    for layer in layer_list:
        weights = get_nested_attr(model, layer).weight._data
        if layer in first:
            fi_cache[layer] = FI(weights)
        else:
            fi_cache[layer] = FI2(weights)

    fault_dict = {
        "Iteration": [],
        "Layer": [],
        "Index": [],
        "Bit": [],
    }

    for k in range(num_iterations):
        print(f"Generating faults for iteration {k+1}/{num_iterations}")
        for _ in range(faults_per_iteration):
            layer = random.choice(layer_list)

            # Reuse existing FI/FI2 object
            fi = fi_cache[layer]
            index, bit = fi.fault_position()

            fault_dict["Iteration"].append(k)
            fault_dict["Layer"].append(layer)
            fault_dict["Index"].append(index)
            fault_dict["Bit"].append(bit)

    df = pandas.DataFrame(fault_dict)
    df.to_csv(csv_path, index=False)
    print(f"Saved fault list to {csv_path}")

    return df, faults_per_iteration

# ============================================================
# 8. One FI test pass for iteration k (using precomputed fault list)
# ============================================================
def test_with_faults(iteration_k, fault_list_df, faults_per_iteration):
    # 1) Reset all layers to original TMR-set weights
    for name, orig in zip(layer_list, original_tmr_weights):
        get_nested_attr(model, name).weight._data = orig.clone()

    # 2) Select the subset of faults for this iteration k
    iter_faults = fault_list_df[fault_list_df["Iteration"] == iteration_k]

    # Optional sanity check:
    if len(iter_faults) != faults_per_iteration:
        print(
            f"[Warning] Iteration {iteration_k} has {len(iter_faults)} faults, "
            f"expected {faults_per_iteration}"
        )

    # 3) Inject faults for this iteration
    for _, row in iter_faults.iterrows():
        layer = row["Layer"]
        index = int(row["Index"])
        bit = int(row["Bit"])

        weights = get_nested_attr(model, layer).weight._data
        if layer in first:
            fi = FI(weights)
        else:
            fi = FI2(weights)

        new_weights = fi.inject(index, bit)
        get_nested_attr(model, layer).weight._data = new_weights

    # 4) Re-apply DMR/DMR2 protection on protected layers
    start_time1 = timeit.default_timer()
    for l in protected_layers:
        print("Protect layer:", l)
        weights = get_nested_attr(model, l).weight._data
        if l in first:
            dmr = DMR(weights)
        else:
            dmr = DMR2(weights)
        new_weights = dmr.protect()
        get_nested_attr(model, l).weight._data = new_weights

    # 5) Evaluate accuracy on CIFAR-10 test set
    correct = 0
    total = 0
    img_id = 0
    model.eval()
    start_time = timeit.default_timer()

    with torch.no_grad():
        for _, (images, labels) in tqdm(enumerate(data), total=len(data)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, reqap_fn=reqap_activation_packing, protected_layers=protected_layers)
            # outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log per-image outputs to CSV (same as original)
            for i in range(outputs.size(0)):
                image_output = outputs[i]
                prediction = predicted[i]
                probs = {
                    f"p{i}": "{:.2f}".format(float(image_output[i]))
                    for i in range(10)
                }
                csv_output = {
                    "fault_id": iteration_k,
                    "img_id": img_id,
                    "predicted": prediction.item(),
                }
                csv_output.update(probs)
                img_id += 1
                output_results.writerow(csv_output)

    eval_time = timeit.default_timer() - start_time
    total_time = timeit.default_timer() - start_time1
    accuracy = 100.0 * correct / total

    print(f"Eval time: {eval_time:.4f} s")
    print(f"Total time (protect+eval): {total_time:.4f} s")
    print(f"Accuracy (iteration {iteration_k}): {accuracy:.4f} %")

    return accuracy


# ============================================================
# 9. Main
# ============================================================
if __name__ == "__main__":
    # Compute N
    N = no_faults()
    print(f"N (no_faults total) = {N}")

    # Always (re)generate fault list for this run
    fault_list_df, faults_per_iteration = generate_fault_list(
        n=N,
        ber=BER,
        num_iterations=Sufficient_no_faults,
        csv_path=fault_csv_path,
    )

    # Run FI iterations
    for k in range(Sufficient_no_faults):
        print(f"\nIteration {k+1}/{Sufficient_no_faults} – injecting faults")
        acc = test_with_faults(k, fault_list_df, faults_per_iteration)
        acc_dict["Accuracy"].append(acc)
        acc_dict["noFI"].append(k)

    # Save accuracy CSV
    acc_df = pandas.DataFrame(acc_dict)
    acc_df.to_csv(csv_acc, index=False)

    avg_accuracy = sum(acc_dict["Accuracy"]) / len(acc_dict["Accuracy"])
    print(f"\nAverage Faulty Accuracy: {avg_accuracy:.4f} %")
    print(f"Accuracy Drop: {BASELINE_ACC - avg_accuracy:.4f} %")

    # Close per-image log file
    output_results_file.close()


    generate_model_report(
    model_name="VGG-11 (REQAP 0–7)",
    model=model,
    layer_list=layer_list,
    protected_layers=protected_layers,
    output_csv="reqap_summary_table.csv",
)