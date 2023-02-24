# Tabular Results

## Overview

This repository contains the results of Table 2 and the decision rules that are given in Figure 2.

## Results

The following datasets have been evaluated:

- Adult
- Breast Cancer



|       | Accuracy (top 1) | Total Logic Gates with HK (OP)  | Total Logic Gates without HK (OP) | 
|-------|:----------------:|:-------------------------------:|:---------------------------------:|
| Adult |      85.35%      |              1474               |               1954                |
| Breast Cancer |      77.58%      |               71                |                123                |



## Usage

### Configuration
This project uses Python 3.10. To install the required packages, run the following command:

```
pip3 install -r requirements.txt
```

### Running Inference

To run the inference, use the following command:

```
python3 main.py
```

### Changing the Dataset

To change the dataset, modify the dataset field in the `config.yaml` file.

## Interpreting the Adult Dataset

The rules for the Adult dataset can be found in the Excel format, in the file `adult_testset_fold_3_RulesUseCase.xlsx`.
