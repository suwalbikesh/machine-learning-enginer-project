# 🧠 End-to-End Machine Learning Pipeline

## Customer Value Prediction using Online Retail Data

![Python](https://img.shields.io/badge/Python-3.9-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Complete-success)

---

## 📌 Project Overview

This project demonstrates a **complete, end-to-end machine learning workflow**, from raw transactional data to a **production-ready machine learning pipeline**. The goal is to predict whether a customer is a **high-value customer** based on historical purchasing behavior.

The project is designed as a **portfolio project** for the **Machine Learning Engineer Career Path** and showcases skills in:

* Data preprocessing & ETL
* Exploratory Data Analysis (EDA)
* Feature engineering
* Supervised & unsupervised learning
* Model evaluation & hyperparameter tuning
* ML pipeline construction
* Business-focused insights

All analysis was conducted in **Jupyter Notebooks**, and version control was managed using **Git**.

---

## 🎯 Project Objectives

* Build an end-to-end machine learning workflow
* Apply supervised and unsupervised ML techniques
* Convert an ML workflow into a reusable pipeline
* Communicate insights clearly to technical and non-technical stakeholders
* Create a project suitable for ML Engineer & Data Scientist roles

---

## 📂 Dataset

**Online Retail Dataset**

* Transactional data from a UK-based online retailer 
* Time period: 2010–2011
* Records include invoices, products, quantities, prices, and customers

🔗 Dataset Source

The dataset is publicly available from the UCI Machine Learning Repository:
👉 https://archive.ics.uci.edu/dataset/352/online+retail

### Key Columns:

* `InvoiceNo`
* `StockCode`
* `Description`
* `Quantity`
* `InvoiceDate`
* `UnitPrice`
* `CustomerID`
* `Country`

---

## 🧩 Problem Definition

### Machine Learning Task

**Binary Classification**

> Predict whether a customer is a **high-value customer** based on historical purchasing behavior.

### Target Variable

* `HighValueCustomer`

  * `1` → High-value customer (top 25% spenders)
  * `0` → Low-value customer

---

## 🔄 ETL & Exploratory Data Analysis (EDA)

### Data Cleaning

* Removed canceled invoices
* Dropped rows with missing `CustomerID`
* Removed negative quantities and prices
* Created `TotalPrice = Quantity × UnitPrice`

### EDA Perspectives

* **Regression**: Distribution of revenue and spending
* **Classification**: Differences between high- and low-value customers
* **Unsupervised Learning**: Customer segmentation using RFM analysis

---

## 🧪 Unsupervised Learning: Customer Segmentation

To understand customer behavior patterns, **RFM features** were created:

* **Recency** – Days since last purchase
* **Frequency** – Number of unique purchases
* **Monetary** – Total spend

### Techniques Used

* Feature scaling (StandardScaler)
* KMeans clustering
* PCA for dimensionality reduction
* Visual inspection of customer segments

### Key Outcome

Four distinct customer segments were identified, including:

* High-value frequent buyers
* New or low-engagement customers
* Loyal moderate spenders

---

## 🛠 Feature Engineering

* Aggregated transactional data to customer-level features
* Created RFM metrics
* Log-transformed skewed variables
* Standardized numerical features
* Split data into:

  * Training set
  * Validation set
  * Test set

---

## 🤖 Machine Learning Models

Multiple models were trained and evaluated:

| Model               | Purpose                  |
| ------------------- | ------------------------ |
| Logistic Regression | Baseline model           |
| Random Forest       | Non-linear relationships |
| Gradient Boosting   | Final selected model     |

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Best Model

**Gradient Boosting Classifier**
Chosen for its strong performance and generalization ability.

---

## 🔧 Hyperparameter Tuning

* GridSearchCV used for:

  * Number of estimators
  * Learning rate
  * Tree depth
* Cross-validation ensured robust performance

---

## 🚀 Machine Learning Pipeline

A **production-ready pipeline** was created using `scikit-learn`:

### Pipeline Steps

1. Feature scaling
2. Model training
3. Prediction

### Benefits

* Reproducible
* Modular
* Easy to deploy and retrain
* Industry-standard ML workflow

---

## 📊 Results & Key Findings

* Monetary value is the strongest predictor of customer value
* Frequency significantly impacts customer classification
* Feature engineering had a larger impact than model selection
* Ensemble models outperformed linear models

---

## 🧠 What I Learned

* Real-world data requires extensive cleaning
* Feature engineering is critical for model performance
* Pipelines are essential for production ML systems
* Visual inspection is crucial in unsupervised learning

---

## 🔮 Future Improvements

* Predict Customer Lifetime Value (CLV)
* Add time-series purchasing features
* Address class imbalance with advanced techniques
* Deploy the model using FastAPI and Docker
* Automate retraining with new incoming data
* Integrate cloud services (AWS/GCP)

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
├── src/
│   ├── pipeline.py
│   ├── train.py
├── README.md
├── requirements.txt
```

---

## 🧑‍💻 Tools & Technologies

* Python
* pandas, NumPy
* scikit-learn
* matplotlib
* Jupyter Notebook
* Git & GitHub

---

## 📬 Contact

If you’d like to discuss this project or collaborate, feel free to connect:

* **GitHub:** [bikesh suwal](https://github.com/suwalbikesh)
* **LinkedIn:** [bikesh suwal](https://www.linkedin.com/in/bikesh-suwal/)
* **Email:** suwalbikezh@gmail.com

---

⭐ *If you found this project helpful, consider starring the repository!*

---
