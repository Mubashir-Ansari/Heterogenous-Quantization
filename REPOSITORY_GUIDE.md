# Repository Guide: Reliability-Aware Performance Optimization through Heterogeneous Quantization

## Overview

This repository implements a **Reliability-Aware Performance Optimization** framework for Deep Neural Network (DNN) hardware accelerators using **Heterogeneous Quantization**. The framework evaluates the resilience of quantized DNN models (AlexNet, VGG-11, ResNet-18) to bit-flip faults and applies **Dual Modular Redundancy (DMR)** protection to critical layers.

## Repository Structure

```
Heterogenous-Quantization/
├── ALexNet/          # AlexNet implementation (Fashion-MNIST dataset)
├── VGG-11/           # VGG-11 implementation (CIFAR-10 dataset)
├── ResNet-18/        # ResNet-18 implementation (CIFAR-10 dataset)
├── fi07.py           # Fault Injection for convolutional layers
├── fi072.py          # Fault Injection for fully-connected layers
├── dmr.py            # Dual Modular Redundancy for conv layers
├── dmr2.py           # Dual Modular Redundancy for FC layers
└── README.md
```

## Algorithm Workflow

### Phase 1: Quantization (Pre-requisite)
- Models are quantized using `quanto` library (int8 quantization)
- Quantized models are saved as `.pth` files (e.g., `qalex-0-7.pth`)
- **Note**: The quantization scripts are not included in this repo, but commented code shows:
  ```python
  # quanto.quantize(model, weights=quanto.qint8, activations=None)
  # quanto.freeze(model)
  # torch.save(model, 'qalex-0-7.pth')
  ```

### Phase 2: Fault Injection Testing
The main algorithm (`p2q0-7ber00001.py`) performs:

1. **Load Quantized Model**: Loads pre-quantized model
2. **DMR Initialization**: Applies DMR protection to critical layers (sets redundant bits)
3. **Fault Injection**: Injects bit-flip faults at specified BER (Bit Error Rate)
4. **DMR Protection**: Applies DMR correction to protected layers
5. **Accuracy Evaluation**: Tests model accuracy after fault injection
6. **Results Collection**: Records accuracy for different fault injection iterations

### Phase 3: Genetic Algorithm (Optional - `*-qgen.py`)
- Uses DEAP library to find critical weights through genetic algorithms
- Evaluates weight sensitivity by perturbing weights and measuring accuracy drop

## Key Components

### 1. Fault Injection (FI)
- **`fi07.py`**: Handles convolutional layers (4D tensors)
- **`fi072.py`**: Handles fully-connected layers (2D tensors)
- **Functions**:
  - `fault_position()`: Randomly selects weight and bit position
  - `inject(index, bit)`: Injects bit-flip fault at specified location
  - `flip_random_bit()`: Flips a bit at position 0-7 (special handling for bit 7 = sign bit)

### 2. Dual Modular Redundancy (DMR)
- **`dmr.py`**: DMR for convolutional layers
- **`dmr2.py`**: DMR for fully-connected layers
- **Functions**:
  - `set()`: Initializes DMR by setting redundant bits (bits 3,4 copy bit 5)
  - `protect()`: Corrects faults using majority voting on redundant bits
  - `correct()`: Error correction logic

### 3. DMR Protection Strategy
- **Protected Layers**: Critical layers are protected (e.g., `conv1[0]`, `fc3` for AlexNet)
- **Redundancy**: Uses bits 3, 4, 5 as triple redundancy for error correction
- **Correction**: Majority voting to correct single-bit errors

## How to Run the Algorithm

### Prerequisites

1. **Required Python Packages**:
   ```bash
   pip install torch torchvision
   pip install quanto
   pip install pandas numpy
   pip install deap  # For genetic algorithm scripts
   pip install tqdm sklearn matplotlib
   pip install PIL
   ```

2. **Required Files**:
   - Quantized model file: `qalex-0-7.pth` (for AlexNet)
   - Fault list CSV: `alex-0-7_fault_list50_ber00001.csv`
   - Dataset files:
     - AlexNet: Fashion-MNIST CSV files in `./kaggle/input/fashionmnist/`
     - VGG-11/ResNet-18: CIFAR-10 (automatically downloaded)

### Running AlexNet Algorithm

1. **Navigate to AlexNet folder**:
   ```bash
   cd ALexNet
   ```

2. **Run the main fault injection script**:
   ```bash
   python p2q0-7ber00001.py
   ```

3. **What the script does**:
   - Loads quantized model `qalex-0-7.pth`
   - Reads fault list from CSV
   - Applies DMR protection to `conv1[0]` and `fc3` layers
   - Injects faults iteratively (25 iterations by default)
   - Tests model accuracy after each fault injection
   - Saves results to `p2alex-0-7-fi25-ber00001.csv`

### Configuration Parameters

In `p2q0-7ber00001.py`:
- `Sufficient_no_faults = 25`: Number of fault injection iterations
- `BER = 0.00001`: Bit Error Rate (faults per total parameters)
- `protected_layers = ['conv1[0]', 'fc3']`: Layers to protect with DMR
- `inputSize = 8000`: Training dataset size

### Running Other Models

**VGG-11**:
```bash
cd VGG-11
python p2q0-7ber00001.py
```

**ResNet-18**:
```bash
cd ResNet-18
python resnet-p2q0-7ber00001.py
```

### Running Genetic Algorithm (Optional)

To find critical weights using genetic algorithms:
```bash
cd ALexNet
python alex-qgen.py
```

This will:
- Run genetic algorithm for 40 generations
- Population size: 100
- Find weights that cause maximum accuracy drop when perturbed

## Expected Output

### Main Script Output
- Accuracy after each fault injection iteration
- Average faulty accuracy
- Accuracy drop compared to baseline
- CSV file with accuracy results

### Output Files
- `p2alex-0-7-fi25-ber00001.csv`: Accuracy results
- `p2output_alex_0-7_ber00001`: Detailed prediction results

## Algorithm Details

### Bit Error Rate (BER) Calculation
- BER = 0.00001 means 1 fault per 100,000 parameters
- Total faults per iteration = `BER * total_parameters`

### DMR Error Correction
1. **Initialization (`set()`)**:
   - Copies bit 5 to bits 3 and 4 (triple redundancy)
   - Format: `[bit0][bit1][bit2][bit3=bit5][bit4=bit5][bit5][bit6][bit7]`

2. **Protection (`protect()`)**:
   - Detects errors by comparing bits 3, 4, 5
   - Uses majority voting to correct single-bit errors
   - Clears sign bits (0-2) if set

### Fault Injection
- Randomly selects layer, weight index, and bit position (0-4)
- Flips the selected bit
- Special handling for bit 7 (sign bit): negates value

## Troubleshooting

### Missing Files
1. **Quantized Model**: You need to create quantized models first (not included)
2. **Fault List CSV**: Should contain columns: `Layer`, `Index`, `Bit`
3. **Dataset**: 
   - AlexNet: Requires Fashion-MNIST CSV files
   - VGG/ResNet: CIFAR-10 downloads automatically

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing CSV**: Generate fault list or reduce `Sufficient_no_faults`
3. **Model Loading Error**: Ensure quantized model file exists

## File Descriptions

| File | Purpose |
|------|---------|
| `p2q0-7ber00001.py` | Main fault injection testing script |
| `*-qgen.py` | Genetic algorithm for critical weight finding |
| `fi07.py` | Fault injection for conv layers |
| `fi072.py` | Fault injection for FC layers |
| `dmr.py` | DMR protection for conv layers |
| `dmr2.py` | DMR protection for FC layers |

## Research Context

This implementation evaluates:
- **Reliability**: Model resilience to hardware faults (bit-flips)
- **Performance**: Accuracy degradation under fault conditions
- **Protection**: Effectiveness of DMR in critical layers
- **Heterogeneous Quantization**: Different quantization strategies per layer

The framework helps optimize DNN accelerators by identifying critical layers and applying targeted protection mechanisms.


