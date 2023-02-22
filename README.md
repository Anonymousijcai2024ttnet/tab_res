# Tabular Results

## Overview

This repository contains the results of Table XXX and the decision rules that are given in Figure XXX.

## Results

The following datasets have been evaluated:

- Adult
- Breast Cancer

### Adult
The accuracy for the Adult dataset is 85.35%.
71 logic gates (AND/OR) with human kwnoledge injection, 123 logic gates (AND/OR) without human kwnoledge injection.

### Breast Cancer
The accuracy for the Breast Cancer dataset is 77.58%.
1474 logic gates (AND/OR) with human kwnoledge injection, 1954 logic gates (AND/OR) without human kwnoledge injection.

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