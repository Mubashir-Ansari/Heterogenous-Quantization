# Generating Fault List CSV

## Problem
The script `p2q0-7ber00001.py` requires a fault list CSV file: `alex-0-7_fault_list50_ber00001.csv`

## Solution

### Option 1: Auto-generation (if model exists)
If you have the quantized model `qalex-0-7.pth`, the main script will attempt to auto-generate the fault list.

### Option 2: Manual generation
Run the generator script manually:

```bash
cd ALexNet
python generate_fault_list.py
```

This will create the required CSV file.

## Requirements
- Quantized model file: `qalex-0-7.pth` must exist in the `ALexNet` directory
- All dependencies must be installed (torch, pandas, etc.)

## What the CSV contains
The fault list CSV has three columns:
- `Layer`: The layer name where the fault will be injected (e.g., 'conv1[0]', 'fc3')
- `Index`: The weight index within that layer
- `Bit`: The bit position to flip (0-4, or 7 for sign bit)

## File structure
The CSV file contains enough fault entries to cover all fault injection iterations. The number of faults depends on:
- `Sufficient_no_faults = 25` (number of iterations)
- `BER = 0.00001` (bit error rate)
- Total model parameters

## Troubleshooting
If you get errors:
1. Ensure `qalex-0-7.pth` exists
2. Check that all dependencies are installed
3. Verify you're in the `ALexNet` directory when running the script


