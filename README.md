# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer will churn, deployed as an interactive web app using Streamlit.

🔗 **Live App:** [adithyan-churn-prediction.streamlit.app](https://adithyan-churn-prediction.streamlit.app)

---

## Problem Statement

Telecom companies lose millions in revenue every year due to customer churn. This project builds a model that predicts which customers are likely to leave, allowing the company to take proactive retention measures.

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No)

## Key Findings from EDA

- **26.5%** of customers churned — imbalanced dataset
- Month-to-month contract customers churn at a significantly higher rate than one or two year contract customers
- Fiber optic internet service users have the highest churn rate
- New customers (low tenure) are far more likely to churn than long-term customers
- Churned customers pay higher monthly charges on average (~$80 vs ~$63)

## Model Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 82.11% | 0.639 |
| XGBoost | 79.13% | 0.570 |
| Random Forest | 78.71% | 0.525 |

**Logistic Regression** performed best and was selected as the final model.

## Top Features Influencing Churn

1. Contract Type (Two year)
2. Tenure
3. Internet Service (Fiber optic)

## Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Deployment:** Streamlit

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── Customer-churn-prediction.ipynb
├── app.py
├── model.pkl
├── scaler.pkl
└── requirements.txt
```

## How to Run Locally

```bash
git clone https://github.com/Adithyan-7/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## How the App Works

1. User enters customer details (contract type, tenure, monthly charges etc.)
2. Input is preprocessed and scaled using the saved scaler
3. Logistic Regression model predicts churn probability
4. Result is displayed with a probability percentage
