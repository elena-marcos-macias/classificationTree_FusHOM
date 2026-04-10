# Decision Tree Classifier with Repeated K-Fold Cross-Validation

A MATLAB pipeline for training and evaluating CART classification trees using repeated stratified k-fold cross-validation. Designed for binary or multi-class classification tasks with an emphasis on reproducible performance estimation and interpretable outputs.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Configuration: the JSON file](#configuration-the-json-file)
- [Input Data Format](#input-data-format)
- [Outputs](#outputs)
- [Key Design Decisions](#key-design-decisions)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This pipeline:

1. Trains a **full CART decision tree** on all available data for visualisation and global predictor importance.
2. Evaluates generalisation performance via **repeated k-fold cross-validation**, producing error rates and AUC distributions across runs.
3. Optionally evaluates an **independent external test set**.
4. Exports all results (metrics, predictor importance, ROC curves, histograms, confusion matrix) to Excel and JPG figures.

> **Important:** The full model is for visualisation only. Its in-sample error is optimistically biased and should **not** be reported as the model's performance. Use the cross-validation metrics instead.

---

## Requirements

- **MATLAB** R2018b or later (R2020b+ recommended)
- **Statistics and Machine Learning Toolbox**
- The custom utility function `selectColumns()` (included in `./utils/`)

To verify you have the required toolbox installed, run in the MATLAB command window:

```matlab
ver('stats')
```

---

## Repository Structure

```
project-root/
├── data/
│   ├── instructionsDecisionTree.json   ← Configuration file (edit this)
│   ├── your_training_data.xlsx         ← Training dataset
│   └── your_test_data.xlsx             ← Independent test dataset
├── utils/
│   └── selectColumns.m                 ← Column selection helper
├── requirements/
│   └── ...                             ← Any additional dependencies
├── results/                            ← Created automatically on first run
│   ├── DecisionTree.jpg
│   ├── Histograms.jpg
│   ├── ROCcurves.jpg
│   ├── Confusion_TestSet.jpg
│   ├── ROC_TestSet.jpg
│   ├── your_cv_metrics.xlsx
│   └── your_test_results.xlsx
└── DecisionTree_CV.m                   ← Main script
```

---

## Quick Start

1. Place your training and test Excel files in the `./data/` folder.
2. Edit `./data/instructionsDecisionTree.json` to point to your files and configure the analysis (see [Configuration](#configuration-the-json-file) below).
3. Open MATLAB, set the working directory to the project root, and run:

```matlab
run('DecisionTree_CV.m')
```

4. All outputs are saved automatically to `./results/`.

---

## Configuration: the JSON file

All analysis parameters are controlled through a single JSON file located at `./data/instructionsDecisionTree.json`. No changes to the MATLAB script itself are needed for a standard analysis.

### Full example

```json
{
   "inputDataSelection": {
        "trainingDataFile": "FDGHom_forfitctree_20260329.xlsx",
        "testDataFile":     "FDGHom_forfitctree_testDiet.xlsx",
        "columnCriteria": {
            "target_columns": [1, 22],
            "ignore_columns": [2, 3]
        },
        "resultsVariable": "Death",
        "positiveClass":   "SUD",
        "catVariable":     "Genotype"
    },
    "crossValidationMethods": {
        "nFolds": [3],
        "nRuns":  [1000]
    },
    "outputFileNames": {
        "excelFileName": "FemFusHOMcTree_CVmetrics.xlsx",
        "testExcelFile": "FusHOMDiet_testDataset.xlsx"
    }
}
```

### Field reference

#### `inputDataSelection`

| Field | Type | Description |
|-------|------|-------------|
| `trainingDataFile` | string | Filename of the training Excel file (must be in `./data/`). |
| `testDataFile` | string | Filename of the independent test Excel file (must be in `./data/`). |
| `columnCriteria.target_columns` | array of 2 integers | **[first, last]** column indices (1-based) to include as the predictor range. Example: `[1, 22]` selects columns 1 through 22. |
| `columnCriteria.ignore_columns` | array of integers | Column indices to **exclude** from the predictor range. Example: `[2, 3]` drops columns 2 and 3 even though they fall within `target_columns`. |
| `resultsVariable` | string | Exact name of the column in the Excel file that contains the class labels (the response/target variable). Case-sensitive. |
| `positiveClass` | string | Exact label of the class of interest. Determines the orientation of the ROC curve and AUC — i.e., TPR/FPR are computed for this class as the "positive". Case-sensitive. |
| `catVariable` | string | Exact name of the single categorical predictor column. Must fall within the selected predictor range. Case-sensitive. |

#### `crossValidationMethods`

| Field | Type | Description |
|-------|------|-------------|
| `nFolds` | integer (wrapped in array) | Number of folds for k-fold cross-validation. Must be ≥ 2 and ≤ number of observations. Typical values: 3, 5, 10. |
| `nRuns` | integer (wrapped in array) | Number of times the full k-fold procedure is repeated (each run uses a different random partition). Higher values give more stable estimates — 100–1000 is typical. |

> **Note on the array syntax:** The JSON values for `nFolds` and `nRuns` must be wrapped in square brackets (e.g., `[10]`) due to how MATLAB's `readstruct` parses scalar values from JSON arrays. Do not write them without brackets.

#### `outputFileNames`

| Field | Type | Description |
|-------|------|-------------|
| `excelFileName` | string | Name of the output Excel file containing CV metrics and predictor importance (saved to `./results/`). |
| `testExcelFile` | string | Name of the output Excel file containing external test set results (saved to `./results/`). |

---

## Input Data Format

Both the training and test files must be **Excel files (`.xlsx`)** with the following structure:

- **First row:** column headers (variable names).
- **Subsequent rows:** one observation per row.
- The response variable column (specified in `resultsVariable`) must contain class labels as text strings.
- The categorical predictor column (specified in `catVariable`) must contain text strings or integer codes.
- All other selected predictor columns should contain numeric values.
- Rows with any missing values are removed automatically (listwise deletion). A warning is printed for each removed row.

### Example layout

| Genotype | Var2 | Var3 | ... | Var22 | Death |
|----------|------|------|-----|-------|-------|
| WT       | 0.5  | 1.2  | ... | 3.1   | SUD   |
| KO       | 0.8  | 0.9  | ... | 2.7   | Control |

With `target_columns: [1, 22]`, `ignore_columns: [2, 3]`, `catVariable: "Genotype"`, and `resultsVariable: "Death"`, the script would use columns 1 and 4–22 as predictors, with column 22 excluded from predictors because it is the response, and column 1 treated as categorical.

---

## Outputs

All files are saved to `./results/` (created automatically).

### Figures (JPG, 300 dpi)

| File | Description |
|------|-------------|
| `DecisionTree.jpg` | Full-model tree diagram. For visualisation and interpretation only. |
| `Histograms.jpg` | Distribution of error rate and AUC across all CV runs. Includes a red vertical line at the mean. |
| `ROCcurves.jpg` | ROC curves for the best, median-closest, mean, and worst CV runs by AUC. |
| `Confusion_TestSet.jpg` | Confusion matrix for the external test set predictions. |
| `ROC_TestSet.jpg` | ROC curve for the external test set. |

### Excel files

**`excelFileName.xlsx`** — cross-validation results:

| Sheet | Contents |
|-------|----------|
| `Errors` | Mean, SD, best, worst, and median-closest error rate and AUC across all CV runs. |
| `PredictorImportance` | Gini importance for each predictor from the full model. |
| `CVTreeImportance` | Importance scores from every individual CV training tree (one row per fold per run). |
| `CVTreeSummary` | Mean, SD, median, usage rate, and value range of importance scores across all CV trees. |

**`testExcelFile.xlsx`** — external test set results:

| Sheet | Contents |
|-------|----------|
| `ExternalTest` | Error rate and AUC on the independent test set. |

---

## Key Design Decisions

**Why two separate models (full model vs CV models)?**
The full model is trained on 100% of the data. It is intentionally biased (overfitting) and exists solely for visualisation and understanding which predictors drive splits. The CV models are what produce honest, generalisation-relevant performance estimates.

**Why is `positiveClass` declared explicitly in the JSON?**
Automatic inference of the positive class (e.g., taking the minority class) is fragile and varies with class balance. Declaring it explicitly ensures that the ROC curve always faces the same direction across datasets, runs, and collaborators.

**Why repeated k-fold instead of a single run?**
A single k-fold run produces a performance estimate that depends on one particular random partition. Repeating the procedure many times (e.g. 1000 runs) averages out this variance and provides a distribution of error rates and AUCs, giving a more reliable picture of model performance.

---

## Troubleshooting

**`selectColumns() not found`**
Make sure the `./utils/` folder exists and contains `selectColumns.m`. The `addpath(genpath('./utils'))` call at the top of the script should handle this automatically when run from the project root.

**`Positive class "X" not found`**
The `positiveClass` field in the JSON is case-sensitive and must exactly match one of the values in your response column. Check for trailing spaces or encoding differences.

**`nFolds cannot exceed the number of observations`**
Your dataset is too small for the number of folds requested. Reduce `nFolds` (e.g., use 3 instead of 10) or provide more data.

**The Excel file is not saved / throws an error**
Close the file in Excel before re-running the script. MATLAB cannot write to an Excel file that is currently open.

**`Required toolbox not found: Statistics and Machine Learning Toolbox`**
Install it via **Home → Add-Ons → Get Add-Ons** in MATLAB, or contact your institution's MATLAB administrator.

**Class imbalance warning**
If one class represents less than 15% of the training data, a warning is issued. The tree may be biased towards the majority class. Consider resampling strategies or adjusting class weights (`fitctree` supports a `'Cost'` parameter).

---

## Citation

If you use this code in your research, please cite the repository:

```
Marcos-Macías, E. (2026). Decision Tree Classifier with Repeated K-Fold Cross-Validation [MATLAB].
GitHub. https://github.com/elena-marcos-macias/classificationTree_FusHOM.git
```

---

*README last updated: April 2026*
