# ELM-Gene-Expression Prediction

This repository contains the implementation of an Lap-ELM model.  The model is trained and evaluated using gene expression data, and the results are presented in terms of MSE (Mean Squared Error) and R² (coefficient of determination).

---

## 📂 Repository Structure

- **`elm_gene_prediction.py`** – Python script for training and testing the ELM model.
- **`data/`** – Folder where the gene expression data files (CSV) should be placed (see below).
- **`results/`** – Folder where the model's output (e.g., `elm_results_by_gene_10000.xlsx`) is saved. (The saved file is actually the results of running the code with 10,000 neurons in the hidden layer). 
- **`requirements.txt`** – File containing the required Python packages and dependencies.

## 📥 Dataset Instructions

Before running the code, you need to download the gene expression dataset and place it in the `data/` folder.

### Dataset Download:
You can download the dataset from the following link:

🔗 [Download Dataset from Google Drive](https://drive.google.com/file/d/1zVGeWnQtlE7yOcY-YEmd3du4Pm3-to2m/view?usp=sharing)

After downloading and extracting the dataset, place the following CSV files in the `data/` folder:

- `new_X_train.csv` – The training feature data.
- `new_y_train.csv` – The training target data (gene expression values).
- `new_X_test.csv` – The testing feature data.
- `new_y_test.csv` – The testing target data (gene expression values).
