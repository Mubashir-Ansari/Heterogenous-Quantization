# import os
# import zipfile
# import torch
# import io

# import requests
# from torch.utils.data import DataLoader
# from torchvision import transforms as T
# from torchvision.datasets import CIFAR10
# from tqdm import tqdm
# import torch.nn as nn
# import random
# import timeit

# #from models.resnet import resnet18, resnet34, resnet50
# from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# # import quanto
# import matplotlib.pyplot as plt

# from fi07 import FI
# from fi072 import FI2
# from dmr import DMR
# from dmr2 import DMR2
# import pandas
# import csv

# acc_dict = {"noFI": [], "Accuracy": []}
# csv_acc = "pvgg11-0-7-fi50-ber0003.csv"

# #fault_dict = {"Iteration": [], "Layer": [], "Index": [], "Bit": []}
# #csv_fault = "vgg11_fault_list50_ber00001.csv"

# output_results_file = open("poutput_vgg11_0-7_ber0003", "w")
# output_results = csv.DictWriter(output_results_file,
#                                         [
#                                             "fault_id",
#                                             "img_id",
#                                             "predicted",
#                                             *[f"p{i}" for i in range(10)],
#                                         ]
#                                         )
# output_results.writeheader()
# output_results_file.close()

# output_results_file = open("poutput_vgg11_0-7_ber0003", "a")
# output_results = csv.DictWriter(output_results_file, ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]])

# Sufficient_no_faults = 50
# BER = 0.0003

# # fault_list = pandas.read_csv('vgg11_fault_list50_ber0003.csv')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.load('qvgg11-0-7.pth').to(device)
# # model = resnet18(pretrained=True)

# layer_list = ['features[0]', 'features[4]', 'features[8]', 'features[11]', 'features[15]', 'features[18]', 'features[22]', 'features[25]', 'classifier[0]', 'classifier[3]', 'classifier[6]']
# first = ['features[0]', 'features[4]', 'features[8]', 'features[11]', 'features[15]', 'features[18]', 'features[22]', 'features[25]']
# second = ['classifier[0]', 'classifier[3]', 'classifier[6]']
# protected_layers = ['features[0]', 'features[4]', 'classifier[6]']

# def get_nested_attr(obj, attr):
#     try:
#         # Split the string by '.' to get individual attributes and indices
#         parts = attr.split('.')
#         for part in parts:
#             # Check if part is indexed (e.g., 'features[0]')
#             if '[' in part and ']' in part:
#                 # Split by '[' and extract the index
#                 part, idx = part.split('[')
#                 idx = int(idx[:-1])  # Convert '0]' to 0
#                 obj = getattr(obj, part)[idx]
#             else:
#                 obj = getattr(obj, part)
#         return obj
#     except AttributeError as e:
#         print(f"Error: {e}")
#         return None

# for l in protected_layers:
#     print(l)
#     weights = get_nested_attr(model, l).weight._data
#     if l in first : dmr = DMR(weights)
#     if l in second : dmr = DMR2(weights)
#     new_weights = dmr.set()
#     get_nested_attr(model, l).weight._data = new_weights
    
# original_tmr_weights = []
# for i in layer_list:
#     original_tmr_weights.append(get_nested_attr(model, i).weight._data)


# #model.eval()

# def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

#     transform = T.Compose(
#         [
#             T.ToTensor(),
#             T.Normalize(mean, std),
#         ]
#     )
#     dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
#     dataloader = DataLoader(
#         dataset,
#         shuffle= False,
#         batch_size=128,
#         num_workers=0,
#         drop_last=True,
#         pin_memory=False,
#     )
#     return dataloader

# transform = T.Compose(
#         [
#             T.RandomCrop(32, padding=4),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)),
#         ]
#     )
# dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

# evens = list(range(0, len(dataset), 10))
# trainset_1 = torch.utils.data.Subset(dataset, evens)

# data = val_dataloader()

# # print(model)
# # print(model.fc.weight._data)
# # print(model.features[25].weight._data)
# # 0, 4, 8, 11, 15, 18, 22, 25
# # print(model.classifier[0].weight)
# # 0, 3, 6

# # layer = 'features[0]'

# # Function to dynamically get the attribute


# # layer_list = ['conv1', 'layer1[0].conv1', 'layer1[0].conv2', 'layer1[1].conv1', 'layer1[1].conv2', 'layer2[0].conv1', 'layer2[0].conv2', 'layer2[1].conv1', 'layer2[1].conv2', 'layer3[0].conv1', 'layer3[0].conv2', 'layer3[1].conv1', 'layer3[1].conv2', 'layer4[0].conv1', 'layer4[0].conv2', 'layer4[1].conv1', 'layer4[1].conv2', 'fc']
# # layer = random.choice(layer_list)
# # print(layer)
# # layer_weights = get_nested_attr(model, layer).weight._data.numpy()

# # # # Plot the distribution of weights
# # plt.hist(layer_weights.flatten(), bins=50)
# # plt.xlabel('Weight Value')
# # plt.ylabel('Frequency')
# # plt.title('Distribution of Weights')
# # plt.show()

# # import timeit
# # correct = 0
# # total = 0

# # model.eval()
# # start_time = timeit.default_timer()
# # with torch.no_grad():
# #     for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
# #         images, labels = images.to("cpu"), labels.to("cpu")
# #         outputs = model(images)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum().item()
# # print(timeit.default_timer() - start_time)
# # print('Accuracy of the network on the 10000 test images: %.4f %%' % (
# #     100 * correct / total))

    
# # # print(model.conv1.weight)

# # quanto.quantize(model, weights=quanto.qint8, activations=None)
# # quanto.freeze(model)
# # torch.save(model, 'qresnet18-16-63.pth')

# # torch.load('qresnet-80-127.pth')

# #print(model.conv1.weight)

# # b = io.BytesIO()
# # torch.save(model.state_dict(), b)
# # b.seek(0)
# # state_dict2 = torch.load(b)


# # loaded_state_dict2 = torch.load('qresnet-80-127.pth')
# # model.load_state_dict(loaded_state_dict2)

# # import timeit
# # correct = 0
# # total = 0

# # model.eval()
# # start_time = timeit.default_timer()
# # with torch.no_grad():
# #     for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
# #         images, labels = images.to("cpu"), labels.to("cpu")
# #         outputs = model(images)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum().item()
# # print(timeit.default_timer() - start_time)
# # print('Accuracy of the golden quantized network on the 10000 test images: %.4f %%' % (
# #     100 * correct / total))



# def no_faults():
#     number =[]
#     for i in layer_list:
#         weights = get_nested_attr(model, i).weight._data
#         if i in first: fi = FI(weights)
#         elif i in second: fi = FI2(weights)
#         nn = fi.param(weights)
#         # print(nn)
#         number.append(nn)
#     total = sum(number) * 5
#     return(total)

# fault_dict = {"Iteration": [], "Layer": [], "Index": [], "Bit": []}

# def generate_fault_list(n):
#     no_faults_each_iteration = int(BER * n)
#     print("each iteration:", no_faults_each_iteration)
#     for j in range(no_faults_each_iteration):
#         layer = random.choice(layer_list)
#         # layer = 'features[15]'
#         # print(layer)
#         weights = get_nested_attr(model, layer).weight._data
#         if layer in first: fi = FI(weights)
#         elif layer in second: fi = FI2(weights)
#         index, bit = fi.fault_position()
#         fault_dict['Iteration'].append(j)
#         fault_dict['Layer'].append(layer)
#         fault_dict['Index'].append(index)
#         fault_dict['Bit'].append(bit)
    
#     fault_df = pandas.DataFrame(fault_dict)
#     fault_df.to_csv('vgg11_fault_list50_ber0003.csv', index=False)
#     return fault_df 

# def test(n, iteration_k):
#     """Test with faults for iteration k"""
#     faults_per_iteration = int(BER * n)
    
#     # Reset weights to original
#     layer_count = 0
#     for i in layer_list:
#         get_nested_attr(model, i).weight._data = original_tmr_weights[layer_count].clone()
#         layer_count += 1

#     # Calculate correct fault indices for this iteration
#     start_idx = iteration_k * faults_per_iteration
#     end_idx = start_idx + faults_per_iteration

#     # Inject faults for this iteration only
#     for t in range(faults_per_iteration):
#         fault_idx = start_idx + t
#         if fault_idx >= len(fault_list):
#             print(f"Warning: fault_idx {fault_idx} exceeds fault_list length {len(fault_list)}")
#             break
        
#         layer = fault_list['Layer'].iloc[fault_idx]
#         index = fault_list['Index'].iloc[fault_idx]
#         bit = fault_list['Bit'].iloc[fault_idx]
        
#         weights = get_nested_attr(model, layer).weight._data
#         if layer in first:
#             fi = FI(weights)
#         elif layer in second:
#             fi = FI2(weights)
#         new_weights = fi.inject(index, bit)
#         get_nested_attr(model, layer).weight._data = new_weights

#     # Apply DMR protection
#     start_time1 = timeit.default_timer()
#     for l in protected_layers:
#         print(l)
#         weights = get_nested_attr(model, l).weight._data
#         if l in first:
#             dmr = DMR(weights)
#         elif l in second:
#             dmr = DMR2(weights)
#         new_weights = dmr.protect()
#         get_nested_attr(model, l).weight._data = new_weights

#     # Test accuracy
#     correct = 0
#     total = 0
#     img_id = 0

#     model.eval()
#     start_time = timeit.default_timer()
#     with torch.no_grad():
#         for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             for i in range(outputs.size(0)):
#                 image_output = outputs[i]
#                 prediction = predicted[i]
#                 probs = {f"p{i}": "{:.2f}".format(float(image_output[i])) for i in range(10)}
#                 csv_output = {
#                     "fault_id": iteration_k, "img_id": img_id, "predicted": prediction.item(),
#                 }
#                 csv_output.update(probs)
#                 img_id += 1
#                 output_results.writerow(csv_output)
    
#     print(timeit.default_timer() - start_time)
#     print("Total time:", timeit.default_timer() - start_time1)
#     accuracy = 100 * correct / total
#     print('Accuracy: %.4f %%' % accuracy)
#     return accuracy

# # def test(n):
# #     #model = torch.load('qvgg11-0-7.pth')
# #     layer_count = 0
# #     for i in layer_list:
# #         get_nested_attr(model, i).weight._data = original_tmr_weights[layer_count] 
# #         layer_count += 1
# #     # loaded_state_dict2 = torch.load('qresnet-80-127.pth')

# #     #layer_list = ['features[0]', 'features[4]', 'features[8]', 'features[11]', 'features[15]', 'features[18]', 'features[22]', 'features[25]']


    


# # # # model2 = resnet50(pretrained=True)
# # # print(model.conv1.weight)
# #     # model.load_state_dict(loaded_state_dict2)
# #     # print(model.conv1.weight._data)
# #     p = 0
# #     for t in range(int(BER * n)):
# #         layer = fault_list['Layer'][k+t]
        
# #         #layer = random.choice(layer_list)
# #         #print(layer)
# #         weights = get_nested_attr(model, layer).weight._data
# #         if layer in first : fi = FI(weights)
# #         if layer in second : fi = FI2(weights)
# #         #index, bit = fi.fault_position()
# #         index = fault_list['Index'][k+t]
# #         bit = fault_list['Bit'][k+t]
# #         new_weights = fi.inject(index, bit)
# #         get_nested_attr(model, layer).weight._data = new_weights
# #         p += 1
# #         #print("which fault", k, p)
# #     start_time1 = timeit.default_timer()
# #     # layer_list = ['features[0]' , 'features[4]', 'features[8]', 'features[11]', 'features[15]']
# #     for l in protected_layers:
# #         print(l)
# #         weights = get_nested_attr(model, l).weight._data
# #         if l in first : dmr = DMR(weights)
# #         if l in second : dmr = DMR2(weights)
# #         new_weights = dmr.protect()
# #         get_nested_attr(model, l).weight._data = new_weights
# # #layer_weights = model.conv1.weight._data.numpy()

# # # Plot the distribution of weights
# # #plt.hist(layer_weights.flatten(), bins=50)
# # #plt.xlabel('Weight Value')
# # #plt.ylabel('Frequency')
# # #plt.title('Distribution of Weights')
# # #plt.show()


    
# #     correct = 0
# #     total = 0
# #     img_id = 0

# #     model.eval()
# #     start_time = timeit.default_timer()
# #     with torch.no_grad():
# #         for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
# #             images, labels = images.to(device), labels.to(device)
# #             outputs = model(images)
# #             _, predicted = torch.max(outputs.data, 1)
# #             total += labels.size(0)
# #             correct += (predicted == labels).sum().item()
# #             for i in range(outputs.size(0)):  # Iterate over the batch size
# #                 image_output = outputs[i]
# #                 prediction = predicted[i]
# #                 probs = {f"p{i}": "{:.2f}".format(float(image_output[i])) for i in range(10)}
# #                 csv_output = {
# #                     "fault_id": k, "img_id": img_id, "predicted": prediction.item(),
# #                 }
# #                 csv_output.update(probs)
# #                 img_id += 1
# #                 output_results.writerow(csv_output)
# #     print(timeit.default_timer() - start_time)
# #     print("Total time:", timeit.default_timer() - start_time1)
# #     print('Accuracy: %.4f %%' % (
# #     100 * correct / total))
# #     return(100. * correct / total)

# n = no_faults()
# print(n)
# fault_list = generate_fault_list(n)
# print("Fault list generated successfully!")

# for k in range(Sufficient_no_faults):
#     print(f"{Sufficient_no_faults - k} faults to inject")
#     accuracy = test(n,k)
#     acc_dict["Accuracy"].append(accuracy)
#     acc_dict["noFI"].append(k)
#     #generate_fault_list(n)

# #data = pandas.DataFrame(fault_dict)
# #data.to_csv(csv_fault)
# data = pandas.DataFrame(acc_dict)
# data.to_csv(csv_acc)
# avg_accuracy = sum(acc_dict["Accuracy"])/len(acc_dict["Accuracy"])
# print('Average Faulty Accuracy: %.4f %%' % (avg_accuracy))
# print('Accuracy Drop: %.4f %%' % (89.8337 - avg_accuracy))    

#--------------------------------------------------------------------------------------------------------------------#
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
        root="datasets/cifar10_data",
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
model = torch.load("qvgg11-0-7.pth", map_location=DEVICE)
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
def generate_fault_list(n, ber, num_iterations, csv_path):
    """
    Generate a fault list for VGG-11 and save as CSV.
    Columns: Iteration, Layer, Index, Bit
    """
    faults_per_iteration = int(ber * n)
    if faults_per_iteration <= 0:
        raise ValueError(
            f"Faults per iteration is {faults_per_iteration}, "
            f"check BER ({ber}) or n ({n})"
        )

    print(f"Total N (no_faults) = {n}")
    print(f"Faults per iteration: {faults_per_iteration}")

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
            weights = get_nested_attr(model, layer).weight._data

            if layer in first:
                fi = FI(weights)
            else:
                fi = FI2(weights)

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
            outputs = model(images)
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
