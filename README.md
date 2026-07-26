# ZTF real/bogus CNN

This project implements and evaluates various computer vision models to classify astronomical events from the Zwicky Transient Facility (ZTF) survey. The main objective is to perform real-bogus classification using 63×63 pixel triplet images (science, reference, and difference). Additionally, it extends to multiclass classification (transient, periodic, stochastic) and includes transfer learning experiments using MobileNetV2 and Braai.

## Overview

## Repository structure

```
.
├── Notebooks/
│   ├── ztf_real_bogus_cnn_I.ipynb    
|   ├── ztf_real_bogus_cnn_II.ipynb    
│   └── ztf_real_bogus_cnn_III.ipynb    
├── data/                                                    
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```


