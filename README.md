# RetainIQ — Customer Churn Prediction Dashboard

An interactive, explainable machine learning dashboard that predicts telecom customer churn, explains *why* a customer is at risk using SHAP, segments customers with K-Means clustering, and surfaces retention strategies — all in a single Streamlit app.

Built end-to-end: data cleaning → model comparison → explainability → clustering → deployment.

---

## Features

- **Single-customer churn prediction** — fill in a customer's details and get an instant churn probability with a Low / Medium / High risk tier.
- **SHAP explainability** — a per-prediction bar chart showing exactly which features pushed that customer's risk up (red) or down (blue), not just a black-box score.
- **Customer segmentation** — K-Means clustering (k=3) groups customers by tenure and monthly charges into behavior segments, each paired with a suggested retention strategy.
- **Bulk CSV scoring** — upload a CSV of many customers and get churn predictions for all of them at once, downloadable as a results file.
- **Business Intelligence tab** — historical churn analytics: churn by contract type, tenure distribution, and monthly charges distribution.

---

## Tech Stack

| Layer | Tools |
|---|---|
| UI / App | [Streamlit](https://streamlit.io/) |
| Data | pandas, NumPy |
| ML | scikit-learn (Logistic Regression, K-Means, `ColumnTransformer`), compared against Random Forest & XGBoost |
| Explainability | SHAP (`LinearExplainer`) |
| Visualization | Plotly |
| Model persistence | joblib |
| Deployment | Render.com |

---

## Project Structure

```
Customer-Churn-Dashboard/
├── dashboard/
│   ├── app.py           # Main Streamlit app — UI, layout, prediction & segmentation flow
│   ├── prediction.py     # Model/preprocessor loading + single/bulk prediction logic
│   └── analytics.py      # Business Intelligence tab — historical churn charts
├── data/
│   └── telco_churn.csv   # Telco Customer Churn dataset
├── models/
│   ├── churn_model.pkl       # Trained Logistic Regression classifier
│   ├── preprocessor.pkl      # Fitted ColumnTransformer (scaling + one-hot encoding)
│   ├── kmeans_model.pkl      # Fitted K-Means clustering model
│   └── cluster_scaler.pkl    # StandardScaler used for clustering features
├── notebooks/
│   └── EDA.ipynb          # Data cleaning, EDA, model comparison, training pipeline
├── .streamlit/
│   └── config.toml        # Custom Streamlit theme
├── render.yaml             # Render deployment config
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Data cleaning & EDA** (`notebooks/EDA.ipynb`) — the Telco Customer Churn dataset (~7,043 customers) is cleaned (fixing the `TotalCharges` column, dropping `customerID`), explored, and split into train/test sets (stratified, to preserve the ~26.5% churn ratio).
2. **Preprocessing** — a `ColumnTransformer` applies `StandardScaler` to numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) and `OneHotEncoder` to categorical columns, all fit on the training set only.
3. **Model comparison** — Logistic Regression, Random Forest, and XGBoost are trained and compared by ROC-AUC (chosen over raw accuracy due to class imbalance). **Logistic Regression** is selected for production, balancing strong performance with interpretability.
4. **Clustering** — K-Means (k=3) groups customers by scaled tenure and monthly charges into distinct segments, each manually mapped to a business-friendly name and retention strategy.
5. **Explainability** — SHAP's `LinearExplainer` computes per-customer feature attributions for the deployed Logistic Regression model.
6. **Serving** — the trained model, preprocessor, and clustering artifacts are saved with `joblib` and loaded directly by the Streamlit app for real-time and bulk inference — no retraining happens in the app itself.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/Customer-Churn-Dashboard.git
cd Customer-Churn-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run dashboard/app.py
```

The app will open at `http://localhost:8501`.

---

## Deployment

Deployed on [Render](https://render.com/) as a Python web service, configured via `render.yaml`:

- **Build:** `pip install -r requirements.txt`
- **Start:** `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
- **Auto-deploy:** enabled — every push to the connected branch triggers a redeploy.

---

## Dataset

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — customer demographics, account information, subscribed services, billing details, and churn outcome for ~7,043 telecom customers.

---

## Known Limitations / Next Steps

- Model and preprocessor are reloaded from disk on every prediction request rather than cached (`@st.cache_resource`), which is inefficient at scale.
- Risk tier thresholds (0.40 / 0.70) are manually chosen, not statistically tuned against business cost/benefit.
- K-Means cluster IDs are mapped to segment names manually after inspection — this mapping would need to be revisited if the model is retrained.
- No database — data and models are file-based (CSV + `.pkl`), which is fine at this scale but wouldn't support live-updating customer records.

---

## License

This project is for educational/portfolio purposes.
