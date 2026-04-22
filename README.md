# Code for "Vertical inheritance, environmental stochasticity and population-level buffering shape cellular microbiota"

## Contents
- Introduction
- System Requirement and dependency
- Usage
- Dataset
- Citation

## Introduction
This repository contains the Python scripts and input data used to generate the analyses and figures in the paper "Vertical inheritance, environmental stochasticity and population-level buffering shape cellular microbiota".

## System Requirement and dependency

### Hardware Requirements
A standard computer with 2 GB RAM is sufficient. For optimal performance: 8+ GB RAM, 4+ cores CPU.

### Software Requirements
- Python 3.x
- Packages: numpy, pandas, scipy, matplotlib, openpyxl

## Dataset
Analysis_for_exp1&exp2/Exp1-Exp2-data.csv
Analysis_for_exp3/Exp3-data.csv
Script-for-FigS3/FigS3-data.xlsx

## Usage
Run the Python script inside each folder with its corresponding input file.

```bash
cd Analysis_for_exp1\&exp2
python Exp1-Exp2-script.py
cd Analysis_for_exp3
python Exp3-script.py
cd Script-for-FigS3
python FigS3-script.py
