# NSDC-SBA-Credit-Risk-Loan-Default-Project
## Project Overview
Developed a robust machine learning classification system to predict small business loan default risk (**Charged-Off** vs. **Paid-in-Full**) using historical National Small Business Administration (SBA) data. This project aims to enhance risk assessment, minimize financial losses, and enable data-driven lending decisions for financial institutions.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
* **Techniques:** Classification, Hyperparameter Tuning (RandomizedSearchCV), SHAP Interpretability, Feature Engineering

## Key Features & Workflow

### 1. Data Engineering & Preprocessing
* **Leakage Prevention:** Identified and removed features not available at the time of loan application to ensure model validity.
* **Cleaning:** Transformed monetary artifacts, handled missing values through custom imputation, and standardized geographic/temporal data.
* **Transformation:** Applied One-Hot Encoding for categorical features and standardized numerical variables for gradient-based model compatibility.

### 2. Exploratory Data Analysis (EDA)
* Analyzed class distributions and identified key risk drivers: **Loan Term, Loan Amount, NAICS Industry Sector, and Approval Year.**
* Identified temporal trends and specific industries exhibiting higher baseline default rates.

### 3. Model Development & Benchmarking
Tested multiple supervised learning algorithms to optimize predictive performance:
* **Models:** Logistic Regression (Baseline), Decision Trees, Random Forest, K-Nearest Neighbors, and **XGBoost**.
* **Optimization:** Implemented `RandomizedSearchCV` with stratified sampling to handle class imbalance and prevent overfitting.
* **Selection:** Chose **XGBoost** as the final model for its superior F1-score and ability to capture non-linear interactions.

### 4. Model Interpretability (SHAP)
* Leveraged **SHAP (SHapeley Additive exPlanations)** to provide transparency.
* Used **Summary Plots** to rank global feature importance and **Dependence Plots** to visualize how individual attributes (like loan term or number of employees) influence the probability of default.

## Summary of Findings
* **Primary Drivers:** Loan structure (term/amount) and economic timing (approval year) are the strongest predictors of default.
* **Industry Risk:** Specific business sectors and characteristics significantly influence baseline credit risk levels.
* **Model Performance:** Ensemble methods significantly outperformed linear models, proving that default risk is driven by complex, multi-factor relationships.

## Repository Content
* `CreditRisk_NSDCProject.ipynb`: Full Python implementation and model training.
* `NSDC Credit Risk Presentation`: Slide deck summarizing business insights.
* `WinterQ NSDC Full Project Timeline`: Detailed documentation of project phases.

---
*Developed as part of the National Student Data Corps (NSDC) Project.*
