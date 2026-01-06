# Housing Dataset Regression Project

This project serves as a machine learning training exercise. The primary objective is to utilize the **Housing dataset** to implement and compare **7 different regression models**, as referenced in [GeeksforGeeks: Regression in Machine Learning](https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/).

## 🚀 Models Implemented

The following regression techniques have been implemented as individual Python scripts:

1.  **Simple Linear Regression** (`Simple_linear.py`)
    - Predicts `price` using only `lotsize`.
2.  **Multiple Linear Regression** (`Multiple_linear.py`)
    - Predicts `price` using multiple features: `lotsize`, `bedrooms`, `bathrms`, `stories`, `garagepl`.
3.  **Polynomial Regression** (`Polynomial_regression.py`)
    - Uses polynomial features (Degree 2) of `lotsize` to capture non-linear relationships.
4.  **Support Vector Regression (SVR)** (`SVR.py`)
    - Uses an RBF kernel with feature scaling to predict `price`.
5.  **Decision Tree Regression** (`Decision_tree.py`)
    - Uses a Decision Tree model to capture complex, non-linear patterns.
6.  **Random Forest Regression** (`Random_forest.py`)
    - Uses an ensemble of 100 decision trees for robust prediction.
7.  **Ridge & Lasso Regression** (`Ridge_lasso.py`)
    - Demonstrates L2 (Ridge) and L1 (Lasso) regularization techniques.

## 📂 Structure

- **`Regression_Housing/`**: Main study folder.
  - `Housing.csv`: The dataset containing housing prices and features.
  - `*.py`: Individual scripts for each regression model.
  - **`export/`**: Directory where all generated plots are saved.

## 🛠️ Installation & Setup

1.  **Create and activate a virtual environment**:

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

You can run any of the regression scripts directly from the root or the `Regression_Housing` directory. The scripts will automatically separate training and testing data and save the resulting visualization to the `export/` folder.

**Example:**

```bash
python Regression_Housing/Simple_linear.py
```

or

```bash
python Regression_Housing/Random_forest.py
```

### 📊 Outputs

Each script generates a visualization plot in the `export/` directory.
If a file already exists, the script will append a number (e.g., `plot_1.png`) to avoid overwriting previous results.
