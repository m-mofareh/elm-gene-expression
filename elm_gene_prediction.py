import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import solve_sylvester
# ==== File Paths ====
data_dir = "data"  # User should put data files here
required_files = ["new_X_train.csv", "new_y_train.csv", "new_X_test.csv", "new_y_test.csv"]

# Check if all required files are present
missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
if missing:
    raise FileNotFoundError(f"Missing data files in '{data_dir}': {missing}\n"
                            "Please upload the required CSV files to the 'data/' directory.")

# ==== Load Data ====
Xtrain = pd.read_csv(os.path.join(data_dir, "new_X_train.csv")).iloc[:, 1:]
Ytrain = pd.read_csv(os.path.join(data_dir, "new_y_train.csv")).iloc[:, 1:]
Xtest = pd.read_csv(os.path.join(data_dir, "new_X_test.csv")).iloc[:, 1:]
Ytest = pd.read_csv(os.path.join(data_dir, "new_y_test.csv")).iloc[:, 1:]

genes_of_interest = [
    "TMEM2", "IARS2", "PRSS23", "PMAIP1", "PARP1", "TRIB1", "RRAGA", "CCNB2", "ARHGEF2", "DUSP4",
    "TXNRD1", "CDK7", "P4HA2", "STX1A", "ARHGAP1", "CSNK2A2", "FOSL1", "PSME1", "NIPSNAP1", "ZFP36",
    "HADH", "ATP2C1", "MCM3", "GLOD4", "GABPB1", "PGM1", "HSPA1A", "ATP1B1", "DNTTIP2", "STMN1",
    "CDK4", "GADD45A", "NPC1", "CDK1", "LIPA", "MYCBP", "BIRC5", "S100A13", "CCNA2", "YKT6", "B4GAT1",
    "HSPB1", "GFPT1", "S100A4", "PSMD2", "ELOVL6", "GALE", "UBE2C", "DNAJB1", "TOP2A"
]
Ytrain = Ytrain[genes_of_interest]
Ytest = Ytest[genes_of_interest]

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x / 8))

def compute_similarity_matrix(Y, method="correlation"):
    return (np.corrcoef(Y.T) + 1) / 2 if method == "correlation" else cosine_similarity(Y.T)

def generate_weights(X, n_components=1000, variance_scale=0.00001):
    range_val = np.sqrt(3 / (variance_scale * n_components))
    return np.random.uniform(-range_val, range_val, (X.shape[1], n_components)).T

def train_elm(X, Y, neurons, lambda_reg=None, S=None, scaling=0.01):
    W = generate_weights(X, n_components=neurons)
    H = sigmoid(np.dot(X, W.T))
    N = X.shape[0]  # Number of training samples

    # MSE-style scaled matrix products
    HtH = (1 / N) * H.T @ H
    HtY = (1 / N) * H.T @ Y

    if S is None and lambda_reg is None:
        # No regularization (MSE-style loss, but solution unchanged)
        beta = np.linalg.pinv(H) @ Y

    elif S is None:
        # L2 regularization (Ridge) using MSE formulation
        regularization_term = HtH + lambda_reg * np.eye(H.shape[1])
        beta = np.linalg.inv(regularization_term).dot(HtY)

    else:
        # Laplacian regularization with MSE-style scaling
        D = np.diag(S.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(S.sum(axis=1) + 1e-8))
        L = D - S
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Solve Sylvester equation: (HtH + λI)β + (scaling * L)β = HtY
        beta = solve_sylvester(HtH + lambda_reg * np.eye(H.shape[1]), scaling * L_norm, HtY)

    return W, beta


def predict_elm(X, W, beta):
    return sigmoid(X @ W.T) @ beta

def evaluate(Y_true, Y_pred):
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.DataFrame(Y_pred, columns=Y_true.columns)
    mse = [mean_squared_error(Y_true.iloc[:, i], Y_pred.iloc[:, i]) for i in range(Y_true.shape[1])]
    r2 = [r2_score(Y_true.iloc[:, i], Y_pred.iloc[:, i]) for i in range(Y_true.shape[1])]
    return np.array(mse), np.array(r2)

# Setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)
neurons = 2000 
lambda_reg = 0.001 #λ1
scaling_param = 0.01 #λ2
S = compute_similarity_matrix(Ytrain)

# Initialize storage for results
mse_val_results = {'NoReg': [], 'L2': [], 'Lap': []}
r2_val_results = {'NoReg': [], 'L2': [], 'Lap': []}
mse_test_results = {'NoReg': [], 'L2': [], 'Lap': []}
r2_test_results = {'NoReg': [], 'L2': [], 'Lap': []}

# Training loop
for fold_num, (train_idx, val_idx) in enumerate(kf.split(Xtrain), 1):
    print(f"Training on fold {fold_num}")
    X_train, Y_train_fold = Xtrain.iloc[train_idx], Ytrain.iloc[train_idx]
    X_val, Y_val = Xtrain.iloc[val_idx], Ytrain.iloc[val_idx]
    
    for method in ['NoReg', 'L2', 'Lap']:
        if method == 'NoReg':
            W, beta = train_elm(X_train, Y_train_fold, neurons)
        elif method == 'L2':
            W, beta = train_elm(X_train, Y_train_fold, neurons, lambda_reg=lambda_reg)
        else:
            W, beta = train_elm(X_train, Y_train_fold, neurons, lambda_reg=lambda_reg, S=S, scaling=scaling_param)

        # Predict
        Y_val_pred = predict_elm(X_val, W, beta)
        Y_test_pred = predict_elm(Xtest, W, beta)

        # Evaluate
        mse_val, r2_val = evaluate(Y_val, Y_val_pred)
        mse_test, r2_test = evaluate(Ytest, Y_test_pred)

        # Save results
        mse_val_results[method].append(mse_val)
        r2_val_results[method].append(r2_val)
        mse_test_results[method].append(mse_test)
        r2_test_results[method].append(r2_test)

# Convert the results into DataFrames per gene
df_mse = pd.DataFrame({'Gene': genes_of_interest})
df_r2 = pd.DataFrame({'Gene': genes_of_interest})

for method in ['NoReg', 'L2', 'Lap']:
    # Validation
    df_mse[f'Val_{method}'] = np.mean(mse_val_results[method], axis=0)
    df_r2[f'Val_{method}'] = np.mean(r2_val_results[method], axis=0)
    # Test
    df_mse[f'Test_{method}'] = np.mean(mse_test_results[method], axis=0)
    df_r2[f'Test_{method}'] = np.mean(r2_test_results[method], axis=0)

# Save to Excel
with pd.ExcelWriter('elm_results_by_gene.xlsx') as writer:
    df_mse.to_excel(writer, sheet_name='MSE', index=False)
    df_r2.to_excel(writer, sheet_name='R2', index=False)

# Print overall averages
print("\n--- AVERAGE RESULTS ACROSS GENES ---")
for metric, df in zip(['MSE', 'R2'], [df_mse, df_r2]):
    print(f"\n{metric} Averages:")
    for col in df.columns[1:]:
        print(f"{col}: {df[col].mean():.4f}")
