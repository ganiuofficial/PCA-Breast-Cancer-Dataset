# Milestone Assignment 2: Principal Component Analysis (PCA)

## Dataset

The project uses the **Breast Cancer Wisconsin dataset** from scikit-learn.

* Source: sklearn.datasets.load_breast_cancer
* Records: 569
* Features: 30 numeric variables
* Target: Binary cancer classification (malignant or benign)

The dataset is built into scikit-learn.
No external file or download is required.

## Tools and Libraries

* Python 3
* NumPy
* Pandas
* Matplotlib
* scikit-learn

## Steps Performed

### 1. Data Loading

The dataset is loaded directly from sklearn.datasets.

### 2. Standardization

All features are scaled using StandardScaler.
PCA requires normalized data to prevent feature dominance.

### 3. PCA Implementation

* PCA is applied with n_components = 2.
* Dimensionality is reduced from 30 features to 2 components.

### 4. Explained Variance

The variance captured by each component is printed.

**Results**

* PC1: 44.27%
* PC2: 18.97%
* Total variance explained: **63.24%**

### 5. Visualization

A 2D scatter plot displays the two PCA components.
Points are colored by cancer class.

## Bonus: Logistic Regression

Logistic regression is applied to the PCA-reduced data.

Purpose:

* Evaluate predictive power after dimensionality reduction.
* Compare classification performance using only two components.

## How to Run the Project

1. Open a terminal.
2. Navigate to the project folder.
3. Run: python pca_analysis.py

Outputs:

* Explained variance printed in the terminal.
* Screenshot of output as **result_output.png**.
* PCA scatter plot saved as **pca_scatter_plot.png**.

## Key Insights

* Two principal components retain over 60% of the dataset variance.
* Feature reduction improves interpretability.
* PCA supports efficient downstream modeling.
* Logistic regression performs well despite reduced dimensionality.

## Conclusion

PCA successfully identifies essential variables from the cancer dataset.
The reduced feature space preserves meaningful structure while simplifying analysis.
This approach supports data-driven decision making for donor funding evaluation.