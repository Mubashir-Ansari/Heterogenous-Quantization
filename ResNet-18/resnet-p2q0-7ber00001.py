import os
import zipfile
import torch
import io

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn
import random
import timeit

from models.resnet import resnet18, resnet34, resnet50  # not strictly needed if you only load .pth

# import quanto
import matplotlib.pyplot as plt

from fi07 import FI
from fi072 import FI2
from dmr import DMR
from dmr2 import DMR2
import pandas
import csv
import numpy as np

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.estimate_overhead import generate_model_report

# -----------------------------
# 0. Reproducibility
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# 1. Config
# -----------------------------
Sufficient_no_faults = 25          # number of fault-injection iterations
BER = 0.00001                      # bit error rate
csv_acc = "p2resnet-0-7-fi50-ber00001.csv"

acc_dict = {"noFI": [], "Accuracy": []}

output_results_file = open("p2output_resnet_0-7_ber00001", "w")
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

output_results_file = open("p2output_resnet_0-7_ber00001", "a")
output_results = csv.DictWriter(
    output_results_file,
    ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. CIFAR-10 test dataloader
# -----------------------------
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

# -----------------------------
# 3. Load quantized ResNet-18
# -----------------------------
# This should be the heterogeneously quantized REQAP model (0–7 range)
model = torch.load("qresnet18-0-7.pth", map_location=DEVICE)
model.to(DEVICE)

# -----------------------------
# 4. Layer lists & accessor
# -----------------------------
layer_list = [
    "conv1",
    "layer1[0].conv1",
    "layer1[0].conv2",
    "layer1[1].conv1",
    "layer1[1].conv2",
    "layer2[0].conv1",
    "layer2[0].conv2",
    "layer2[1].conv1",
    "layer2[1].conv2",
    "layer3[0].conv1",
    "layer3[0].conv2",
    "layer3[1].conv1",
    "layer3[1].conv2",
    "layer4[0].conv1",
    "layer4[0].conv2",
    "layer4[1].conv1",
    "layer4[1].conv2",
    "fc",
]

first = [
    "conv1",
    "layer1[0].conv1",
    "layer1[0].conv2",
    "layer1[1].conv1",
    "layer1[1].conv2",
    "layer2[0].conv1",
    "layer2[0].conv2",
    "layer2[1].conv1",
    "layer2[1].conv2",
    "layer3[0].conv1",
    "layer3[0].conv2",
    "layer3[1].conv1",
    "layer3[1].conv2",
    "layer4[0].conv1",
    "layer4[0].conv2",
    "layer4[1].conv1",
    "layer4[1].conv2",
]

second = ["fc"]

# Protected layers (heterogeneous (3,5) scheme)
protected_layers = [
    "conv1",
    "fc",
    "layer1[1].conv1",
    "layer1[0].conv2",
    "layer1[1].conv2",
]


def get_nested_attr(obj, attr):
    """
    Resolve things like 'layer1[0].conv1' etc.
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


def reqap_activation_protect(act, layer_name="unknown", inject_fault=True, protected_layers=None):
    """
    Simulates REQAP-style activation protection:
    - Quantize activations (8-bit)
    - Duplicate for redundancy (DMR)
    - Optionally inject random bit flips
    - Correct via simple majority voting
    """

    # Always return tensor if not protected
    if protected_layers is not None and layer_name not in protected_layers:
        return act

    # 1️⃣ Quantize
    act_q = torch.clamp((act * 127).round(), -128, 127).to(torch.int8)

    # 2️⃣ Duplicate
    act_copy1 = act_q.clone()
    act_copy2 = act_q.clone()

    # 3️⃣ Optional fault injection
    if inject_fault:
        ber = 1e-5
        fault_mask = (torch.rand_like(act_copy1.float()) < ber)
        act_copy1 = act_copy1 ^ fault_mask.to(torch.int8)

    # 4️⃣ Correction (DMR-style)
    diff_mask = (act_copy1 != act_copy2)
    act_recovered = torch.where(diff_mask, act_copy2, act_copy1)

    # 5️⃣ De-quantize
    act_recovered = act_recovered.float() / 127.0

    return act_recovered

# -----------------------------
# 5. Apply DMR/TMR "set" to protected layers
# -----------------------------
for l in protected_layers:
    print("Set DMR for layer:", l)
    weights = get_nested_attr(model, l).weight._data
    if l in first:
        dmr = DMR(weights)
    else:
        dmr = DMR2(weights)
    new_weights = dmr.set()
    get_nested_attr(model, l).weight._data = new_weights

# Store original protected-weights state as baseline
original_tmr_weights = []
for name in layer_list:
    original_tmr_weights.append(get_nested_attr(model, name).weight._data.clone())


# -----------------------------
# 6. Count "no_faults" (N)
# -----------------------------
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
    total = sum(number) * 5  # as in original implementation
    return total


# -----------------------------
# 7. Inject faults for one iteration
# -----------------------------
def inject_faults_for_iteration(iteration_k, n, log_faults=False):
    """
    Inject int(BER * n) random faults into random layers of the model.
    """
    faults_per_iteration = max(1, int(BER * n))

    fault_dict_local = {
        "fault_id": [],
        "Layer": [],
        "Index": [],
        "Bit": [],
    }

    for f in range(faults_per_iteration):
        layer = random.choice(layer_list)
        weights = get_nested_attr(model, layer).weight._data
        if layer in first:
            fi = FI(weights)
        else:
            fi = FI2(weights)

        index, bit = fi.fault_position()
        new_weights = fi.inject(index, bit)
        get_nested_attr(model, layer).weight._data = new_weights

        if log_faults:
            fault_dict_local["fault_id"].append(iteration_k)
            fault_dict_local["Layer"].append(layer)
            fault_dict_local["Index"].append(index)
            fault_dict_local["Bit"].append(bit)

    if log_faults:
        df = pandas.DataFrame(fault_dict_local)
        df.to_csv(f"resnet18_faults_iter_{iteration_k}.csv", index=False)


# -----------------------------
# 8. One full test with faults for iteration k
# -----------------------------
def test_with_faults(n, iteration_k):
    # 1) Reset all layers to original DMR-set weights
    for name, orig in zip(layer_list, original_tmr_weights):
        get_nested_attr(model, name).weight._data = orig.clone()

    # 2) Inject random faults for THIS iteration
    inject_faults_for_iteration(iteration_k, n, log_faults=False)

    # 3) Re-apply DMR protection on protected layers
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

    # 4) Evaluate accuracy on test set
    correct = 0
    total = 0
    img_id = 0
    model.eval()
    start_time = timeit.default_timer()

    with torch.no_grad():
        for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, reqap_fn=lambda x, name: reqap_activation_protect(x, name, protected_layers=protected_layers))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # log per-image outputs to CSV (same format as original)
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


# -----------------------------
# 9. Main loop
# -----------------------------
if __name__ == "__main__":
    n = no_faults()
    print(f"N (no_faults total) = {n}")

    for k in range(Sufficient_no_faults):
        print(f"\nIteration {k+1}/{Sufficient_no_faults} – injecting faults")
        accuracy = test_with_faults(n, k)
        acc_dict["Accuracy"].append(accuracy)
        acc_dict["noFI"].append(k)

    data_df = pandas.DataFrame(acc_dict)
    data_df.to_csv(csv_acc, index=False)

    avg_accuracy = sum(acc_dict["Accuracy"]) / len(acc_dict["Accuracy"])
    print(f"Average Faulty Accuracy: {avg_accuracy:.4f} %")
    print(f"Accuracy Drop: {90.9454 - avg_accuracy:.4f} %")

    # --- NEW: REQAP REPORT TABLE ---
    generate_model_report(
        model_name="ResNet-18 (REQAP 0–7)",
        model=model,
        layer_list=layer_list,
        protected_layers=protected_layers,
        output_csv="reqap_summary_table.csv",
    )
