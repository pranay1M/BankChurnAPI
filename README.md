# Bank Customer Churn Prediction (Jupyter)

End-to-end notebook that:
- loads the Bank Customer Churn CSV,
- explores data & class imbalance (~20% churn),
- engineers features (one-hot encode `country`, `gender` with `drop_first=True`),
- trains baseline Logistic Regression and Random Forest,
- evaluates with ROC-AUC (primary), PR-AUC, confusion matrix, and report,
- performs Stratified 5-fold cross-validation,
- tunes Random Forest with GridSearchCV,
- inspects feature importances.

## Files
- `BankCustomerChurnProj.ipynb` — main notebook
- `requirements.txt` — Python dependencies
- `.gitignore` — repo ignores

## Data schema (columns used)
`customer_id` (dropped), `credit_score`, `country`, `gender`, `age`, `tenure`,
`balance`, `products_number`, `credit_card`, `active_member`, `estimated_salary`,
`churn` (target: 1 = exited)

## DATA
The dataset can be downloaded from:
https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

## Reproduce locally
```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
Go to http://127.0.0.1:8000/docs
#python app.py