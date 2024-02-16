# Analysis

This directory contains code for analyzing different metrics over the SRAM data.

## Usage

Create a virtual environment and activate it (optional, recommended):
```bash
python3 -m venv .venv
# For windows:
venv\Scripts\activate
# For macOS/Linux. May change depending on your shell
source venv/bin/activate 
```

Install the required libraries
```bash
python3 -m pip install -r requirements.txt
```

Run the entry file
```bash
python3 main.py
```

Then follow the instruction to choose which chips
you want to evaluate and what metrics to run.

## Implemented metrics

* Bit error rate
* Autocorrelation
* Fractional Hamming weight
* Cell stability
* Inter-chip Hamming distance
* Inter-chip minimum entropy
* Intra-chip minimum entropy
