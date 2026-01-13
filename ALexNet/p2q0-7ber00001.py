import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import quanto
import csv

import random
from fi07 import FI
from fi072 import FI2
from dmr import DMR
from dmr2 import DMR2
import pandas
import csv
import timeit

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
Sufficient_no_faults = 25      # number of fault-injection iterations
BER = 0.00001                  # bit error rate
csv_acc = "p2alex-0-7-fi25-ber00001.csv"

acc_dict = {"noFI": [], "Accuracy": []}

# CSV for per-image outputs (optional, same as original)
output_results_file = open("p2output_alex_0-7_ber00001", "w")
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

output_results_file = open("p2output_alex_0-7_ber00001", "a")
output_results = csv.DictWriter(
    output_results_file,
    ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]],
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# -----------------------------
# 2. Data loading
# -----------------------------
train_csv = pd.read_csv("dataset/fashionmnist/fashion-mnist_train.csv")
test_csv = pd.read_csv("dataset/fashionmnist/fashion-mnist_test.csv")

# Customize training size here
inputSize = 8000
train_csv = train_csv[:inputSize]


class FashionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label, image = [], []
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])

        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype("float32")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]

        if self.transform is not None:
            pil_image = Image.fromarray(np.uint8(image))
            image = self.transform(pil_image)

        return image, label


AlexTransform = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_loader = DataLoader(
    FashionDataset(train_csv, transform=AlexTransform),
    batch_size=100,
    shuffle=False,
)

# IMPORTANT: use TEST SET here
test_loader = DataLoader(
    FashionDataset(test_csv, transform=AlexTransform),
    batch_size=100,
    shuffle=False,
)

# -----------------------------
# 3. Model definition (AlexNet)
# -----------------------------
class fasion_mnist_alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.log_file = open("activation_protection_log.csv", "w", newline="")
        self.csv_logger = csv.DictWriter(
            self.log_file,
            fieldnames=["layer", "total_values", "faults_injected", "mismatches_corrected"]
        )
        self.csv_logger.writeheader()
# ============================================================
    # REQAP Activation Protection Function
    # ============================================================
    def reqap_activation_protect(self, act, layer_name="unknown", inject_fault=True):
        """
        Simulates REQAP-style activation protection:
        - Quantize activations (8-bit)
        - Duplicate for redundancy (DMR)
        - Optionally inject random bit flips
        - Correct via simple majority voting
        """
        act_q = torch.clamp((act * 127).round(), -128, 127).to(torch.int8)
        total_vals = act_q.numel()

        # 2️⃣ Duplicate for redundancy
        act_copy1 = act_q.clone()
        act_copy2 = act_q.clone()

        faults_injected = 0
        mismatches_corrected = 0

        # 3️⃣ Optional: Inject random bit flips
        if inject_fault:
            ber = 1e-5  # Bit Error Rate
            fault_mask = (torch.rand_like(act_copy1.float()) < ber)
            faults_injected = fault_mask.sum().item()
            flipped = act_copy1 ^ fault_mask.to(torch.int8)
            act_copy1 = flipped

        # 4️⃣ Correction (simulate majority voting)
        diff_mask = (act_copy1 != act_copy2)
        mismatches_corrected = diff_mask.sum().item()
        act_recovered = torch.where(diff_mask, act_copy2, act_copy1)

        # 5️⃣ Convert back to float for next layer
        act_recovered = act_recovered.float() / 127.0

        # 6️⃣ Logging to CSV
        # self.csv_logger.writerow({
        #     "layer": layer_name,
        #     "total_values": total_vals,
        #     "faults_injected": faults_injected,
        #     "mismatches_corrected": mismatches_corrected,
        # })
        # self.log_file.flush()

        # # Optional: print live to console (comment out if verbose)
        # print(f"[REQAP] {layer_name:6s} | vals={total_vals:<8d} "
        #       f"faults={faults_injected:<5d} corrected={mismatches_corrected:<5d}")

        return act_recovered

    # ============================================================
    # Forward Pass with Activation-Level REQAP
    # ============================================================
    def forward(self, x):
        out = self.conv1(x)
        out = self.reqap_activation_protect(out, "conv1", inject_fault=True)

        out = self.conv2(out)
        out = self.reqap_activation_protect(out, "conv2", inject_fault=True)

        out = self.conv3(out)
        out = self.reqap_activation_protect(out, "conv3", inject_fault=True)

        out = self.conv4(out)
        out = self.reqap_activation_protect(out, "conv4", inject_fault=True)

        out = self.conv5(out)
        out = self.reqap_activation_protect(out, "conv5", inject_fault=True)

        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.reqap_activation_protect(out, "fc1", inject_fault=True)

        out = F.dropout(out, 0.5, training=self.training)
        out = F.relu(self.fc2(out))
        out = self.reqap_activation_protect(out, "fc2", inject_fault=True)

        out = F.dropout(out, 0.5, training=self.training)
        out = self.fc3(out)
        out = self.reqap_activation_protect(out, "fc3", inject_fault=True)

        out = F.log_softmax(out, dim=1)
        return out

# -----------------------------
# 4. Eval helper
# -----------------------------
def testt(model, device, test_loader):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100.0 * correct / len(test_loader.dataset)
    return acc


# -----------------------------
# 5. Load quantized model
# -----------------------------
# Make sure model is on DEVICE
model = torch.load("qalex-0-7.pth", map_location=DEVICE)
model.to(DEVICE)

# Reinitialize csv_logger since it's not saved in the model
model.log_file = open("activation_protection_log.csv", "w", newline="")
model.csv_logger = csv.DictWriter(
    model.log_file,
    fieldnames=["layer", "total_values", "faults_injected", "mismatches_corrected"]
)
model.csv_logger.writeheader()

# -----------------------------
# 6. Accessing layers & lists
# -----------------------------
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


layer_list = ["conv1[0]", "conv2[0]", "conv3[0]", "conv4[0]", "conv5[0]", "fc1", "fc2", "fc3"]
first = ["conv1[0]", "conv2[0]", "conv3[0]", "conv4[0]", "conv5[0]"]  # conv layers
second = ["fc1", "fc2", "fc3"]                                        # FC layers
protected_layers = ["conv1[0]", "fc3"]


# -----------------------------
# 7. Apply DMR "set" to protected layers (baseline)
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

# Save original (DMR-set) weights for reset
original_weights = []
for name in layer_list:
    original_weights.append(get_nested_attr(model, name).weight._data.clone())


# -----------------------------
# 8. Count "no_faults" (N)
# -----------------------------
def no_faults():
    numbers = []
    for name in layer_list:
        weights = get_nested_attr(model, name).weight._data
        if name in first:
            fi = FI(weights)
        else:
            fi = FI2(weights)
        nn = fi.param(weights)
        numbers.append(nn)
    total = sum(numbers) * 5  # as in original code
    return total


# -----------------------------
# 9. Fault injection per iteration
# -----------------------------
def inject_faults_for_iteration(iteration_k, n, log_faults=False):
    """
    Inject int(BER * n) random faults into random layers.
    Optionally log them to CSV if needed.
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
        df.to_csv(f"alex_faults_iter_{iteration_k}.csv", index=False)


# -----------------------------
# 10. One full test with faults for iteration k
# -----------------------------
def test_with_faults(n, iteration_k):
    # 1) Reset weights to original DMR-set weights
    for name, orig_w in zip(layer_list, original_weights):
        get_nested_attr(model, name).weight._data = orig_w.clone()

    # 2) Inject faults for THIS iteration
    inject_faults_for_iteration(iteration_k, n, log_faults=False)

    # 3) Apply DMR protection on protected layers
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

    # 4) Evaluate accuracy on test set (no dropout in eval)
    model.eval()
    start_time = timeit.default_timer()
    correct = 0
    total = 0
    img_id = 0

    with torch.no_grad():
        for _, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Optional: log per-image outputs (same as original structure)
            for i in range(outputs.size(0)):
                image_output = outputs[i]
                prediction = predicted[i]
                probs = {f"p{i}": "{:.2f}".format(float(image_output[i])) for i in range(10)}
                csv_output = {
                    "fault_id": iteration_k,
                    "img_id": img_id,
                    "predicted": prediction.item(),
                }
                csv_output.update(probs)
                img_id += 1
                output_results.writerow(csv_output)

    acc = 100.0 * correct / total
    print("Eval time:", timeit.default_timer() - start_time)
    print("Total time (protection+eval):", timeit.default_timer() - start_time1)
    print(f"Accuracy (iteration {iteration_k}): {acc:.4f} %")
    return acc

# ============================================================
# REQAP HARDWARE OVERHEAD ESTIMATION (PLATFORM-INDEPENDENT)
# ============================================================

# -----------------------------
# Cost model assumptions
# -----------------------------
# - Baseline quantization: 8-bit weights, no redundancy
# - REQAP uses DMR (2 copies) for protected layers
# - Per-bit protection logic:
#     XOR  -> 1 gate (bit comparison)
#     MUX  -> 3 gates (bit selection)
#   Total = 4 gates per protected bit

BITS_PER_WEIGHT = 8
GATES_PER_BIT = 4   # XOR + MUX (symbolic gate cost)


def count_layer_weights(model, layer_names):
    """
    Count number of weights (parameters) per layer.

    Returns:
        dict: {layer_name: number_of_weights}
    """
    weight_counts = {}
    for name in layer_names:
        layer = get_nested_attr(model, name)
        if layer is None:
            continue
        weight_counts[name] = layer.weight.numel()
    return weight_counts


def estimate_reqap_overhead(weight_counts, protected_layers):
    """
    Estimate REQAP hardware overhead compared to baseline quantization.

    Baseline:
        Memory = W × B
        Logic  = 0 extra gates

    REQAP (DMR):
        Memory = W × B × 2
        Extra memory = W × B
        Extra logic  = W × B × 4 gates

    Args:
        weight_counts (dict): number of weights per layer
        protected_layers (list): layers protected by REQAP

    Returns:
        total_extra_bits (int)
        total_extra_gates (int)
    """
    total_extra_bits = 0
    total_extra_gates = 0

    for layer, W in weight_counts.items():
        if layer in protected_layers:
            # Memory overhead: duplicate copy
            extra_bits = W * BITS_PER_WEIGHT

            # Logic overhead: XOR + MUX per bit
            extra_gates = W * BITS_PER_WEIGHT * GATES_PER_BIT

            total_extra_bits += extra_bits
            total_extra_gates += extra_gates

    return total_extra_bits, total_extra_gates


# -----------------------------
# 11. Main loop
# -----------------------------
if __name__ == "__main__":

    n = no_faults()
    print(f"Total 'no_faults' count (N): {n}")

    for k in range(Sufficient_no_faults):
        print(f"\nIteration {k+1}/{Sufficient_no_faults} – injecting faults")
        accuracy = test_with_faults(n, k)
        acc_dict["Accuracy"].append(accuracy)
        acc_dict["noFI"].append(k)

    data = pandas.DataFrame(acc_dict)
    data.to_csv(csv_acc, index=False)

    avg_accuracy = sum(acc_dict["Accuracy"]) / len(acc_dict["Accuracy"])
    print(f"Average Faulty Accuracy: {avg_accuracy:.4f} %")
    print(f"Accuracy Drop: {93.2625 - avg_accuracy:.4f} %")

    generate_model_report(
    model_name="AlexNet (REQAP 0–7)",
    model=model,
    layer_list=layer_list,
    protected_layers=protected_layers,
    output_csv="reqap_summary_table.csv",
)

    # Close the log file
    model.log_file.close()