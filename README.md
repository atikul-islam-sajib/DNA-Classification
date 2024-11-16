# DNA Classification Task 

**DNA classification** is a computational and biological process used to categorize DNA sequences into predefined classes or groups based on specific characteristics, such as their genetic composition, structure, or origin. It can involve analyzing nucleotide sequences (A, T, C, G) and determining their relationship to known organisms, functional elements, or evolutionary traits.

Classification can be done at different levels, such as:
- **Taxonomic classification**: Assigning DNA sequences to species, genera, or higher taxonomic levels.
- **Functional classification**: Identifying functional elements like genes, promoters, or regulatory sequences.
- **Medical classification**: Grouping sequences based on their association with diseases or therapeutic targets.

![DNA](https://assets.zilliz.com/1_a7469e9eac.png)

---

This project provides a tool for training and evaluating machine learning models for DNA classification tasks. It is a modular and configurable tool that allows users to preprocess datasets, extract features, perform feature selection, tune hyperparameters, and train models with K-Fold cross-validation.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [CLI Arguments](#cli-arguments)
5. [Configuration](#configuration)
6. [Examples](#examples)
7. [File Structure](#file-structure)
8. [License](#license)

---

## Features

- **Dataset Loading and Splitting**: Load datasets and split them into training and testing subsets.
- **Feature Engineering**: Option to perform feature extraction and selection.
- **Hyperparameter Tuning**: Automated hyperparameter tuning for supported models.
- **K-Fold Cross-Validation**: Configurable K-Fold cross-validation for robust evaluation.
- **Multiple Models Support**: Train and evaluate using models such as Random Forest (RF), Decision Tree (DT), Logistic Regression (LR), XGBoost (XGB), and Naive Bayes (NB).
- **Configurable via YAML**: Easy configuration of dataset paths, preprocessing methods, and training parameters through a `config.yml` file.

## File Structure

```
.
├── README.md                # Documentation
├── requirements.txt         # Python dependencies
├── config.yml               # Configuration file
├── src/
│   ├── cli.py               # Main CLI script
│   ├── utils/
│   │   └── config.py        # Configuration handling
│   ├── feature_generator.py # Feature generation module
│   ├── dataloader.py        # Data loading and preprocessing module
│   └── trainer.py           # Training module
├── data/                    # Dataset files (to be added by the user)
│   ├── raw/                 # Raw datasets
│   └── processed/           # Processed datasets
└── artifacts/               # Model artifacts and results
```
---

## Installation

1. Clone the repository:

   ```bash
   https://github.com/atikul-islam-sajib/DNA-Classification.git
   cd DNA-Classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `src/` folder is in your Python path:

   ```bash
   export PYTHONPATH=$PYTHONPATH:./src/
   ```

---

## Usage

Run the CLI tool using the following command:

```bash
python cli.py [OPTIONS]
```

---

## CLI Arguments

| Argument         | Type    | Default                      | Choices                  | Description                                                                 |
|-------------------|---------|------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `--dataset`       | `str`   | From config file            | N/A                      | Path to the dataset file.                                                  |
| `--approaches`    | `list`  | From config file            | ["single", "di", "tri", "tetra", "gc_content"]                     | List of approaches to preprocess the dataset.                              |
| `--split_size`    | `float` | From config file            | N/A                      | Ratio of train/test split.                                                 |
| `--FE`            | `bool`  | From config file            | `True`, `False`          | Perform feature extraction.                                                |
| `--FS`            | `bool`  | From config file            | `True`, `False`          | Perform feature selection.                                                 |
| `--HP`            | `bool`  | From config file            | `True`, `False`          | Perform hyperparameter tuning.                                             |
| `--model`         | `str`   | `"RF"`                     | `RF`, `DT`, `LR`, `XGB`, `NB` | Model to be trained.                                                       |
| `--KFold`         | `int`   | From config file            | `1` to `10`              | Number of folds for cross-validation.                                      |

---

## Configuration

The tool uses a `config.yml` file to define default paths and parameters. Below is an example of the configuration file:

```yaml
path:
  raw_path: "./data/raw/"                         # Directory path for raw data files
  processed_path: "./data/processed/"             # Directory path for processed data files
  files_path: "./artifacts/files/"                # Directory path for saved artifacts and files

dataloader:
  dataset: "./data/raw/DNA-Classification.csv"    # Path to the main dataset file
  split_size: 0.20                                # Test split size for train-test split
  approaches: ["single", "di", "gc_content"]      # Feature extraction approaches: single, di, gc_content

trainer:
  model_name: "LR"                                # Model choice: "RF", "DT", "NB", "LR", or "XGB"
  features_extraction: True                       # Enable feature extraction in training
  features_selection: False                       # Disable feature selection by default
  hyperparameter_tuning: False                    # Disable hyperparameter tuning when using KFold
  KFold: 2                                        # Number of folds for KFold cross-validation
```

To use this configuration, ensure that the `config.yml` file is located in the root directory of your project.

---

## Examples

### Train a Random Forest model with feature extraction and K-Fold cross-validation:

```bash
python cli.py --dataset data/dna.csv --approaches single di tri tetra gc_content --split_size 0.8 --FE True --FS False --HP True --model RF --KFold 5 --train
```

### Train a Logistic Regression model without feature selection:

```bash
python cli.py --test
```

## License

This project is licensed under the [MIT License](LICENSE).

---

## Notes

For additional help, feel free to raise an issue on the project's repository or consult the source code for detailed implementation.
