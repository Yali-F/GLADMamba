# GLADMamba: Unsupervised Graph-Level Anomaly Detection Powered by Selective State Space Model

## Requirements
This code requires the following dependencies:
- Python == 3.9.19
- PyTorch == 1.11.0
- PyTorch Geometric == 2.0.4
- NumPy == 1.26.4
- Scikit-learn == 1.0.2
- OGB == 1.3.3
- NetworkX == 2.7.1
- Einops == 0.8.0 

## Hardware Infrastructures
The hardware configuration used for the implementation on the Linux server includes the following:
- **CPU**: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- **GPU**: NVIDIA L40 (48GB), NVIDIA A40 (48GB)

## Datasets
Download and process automatically from the TUDataset [https://chrsmrrs.github.io/datasets/docs/datasets/](https://chrsmrrs.github.io/datasets/docs/datasets/)

## Quick Start
run the script corresponding to the dataset you want. For instance:
```bash
bash script/ad_BZR.sh
