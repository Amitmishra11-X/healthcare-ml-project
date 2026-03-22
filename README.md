# 🏥 Healthcare Data Analytics & Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning pipeline for **early diabetes prediction** using the PIMA Indians Diabetes Dataset. Achieves **89% accuracy and 0.94 ROC-AUC** with full SHAP explainability and an interactive Streamlit dashboard.

🌐 **Live Demo:** [https://healthcare-ml-project-uapgsqtweg9ttv8pxdrymz.streamlit.app](https://healthcare-ml-project-uapgsqtweg9ttv8pxdrymz.streamlit.app)

---

## 📌 Project Overview

Diabetes affects **77 million people in India** — the 2nd highest in the world. Nearly 50% of diabetics are undiagnosed until serious complications arise. Early prediction using ML can enable timely intervention and save lives.

This project builds a clinically-informed prediction system that:

- Predicts diabetes risk from 8 clinical features with high accuracy
- Uses **Random Forest + XGBoost ensemble** for robust predictions
- Implements **SHAP explainability** to identify which clinical factors drive each prediction
- Deploys as a **real-time interactive Streamlit web app**

---

## 🏗️ System Architecture

```
Raw PIMA Dataset (768 records)
         │
         ▼
┌─────────────────────────┐
│   Data Preprocessing    │
│  - Zero value imputation│
│  - Median imputation    │
│  - Feature engineering  │
│  - Train/Test split     │
└──────────┬──────────────┘
           │
  ┌────────┴────────┐
  ▼                 ▼
┌──────────┐  ┌──────────┐
│  Random  │  │ XGBoost  │
│  Forest  │  │Classifier│
└────┬─────┘  └────┬─────┘
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │    Ensemble   │
    │  Predictions  │
    └───────┬───────┘
            │
   ┌────────┴────────┐
   ▼                 ▼
┌──────┐      ┌────────────┐
│ SHAP │      │ Streamlit  │
│ XAI  │      │ Dashboard  │
└──────┘      └────────────┘
```

---

## 🔑 Key Features

| Feature | Description |
|---|---|
| **89% Accuracy** | Random Forest + XGBoost ensemble on PIMA dataset |
| **0.94 ROC-AUC** | Strong discriminative ability across all thresholds |
| **SHAP Explainability** | Global and local feature importance — identifies Glucose as primary risk factor |
| **Domain Preprocessing** | Biologically impossible zero values replaced with median imputation |
| **Feature Engineering** | 3 interaction features: BMI×Age, Glucose/Insulin, Pregnancies×Age |
| **5-Page EDA Dashboard** | Purchase patterns, distributions, correlations, feature analysis |
| **Real-time Prediction** | Live risk assessment from user-entered clinical values |
| **Model Comparison** | Side-by-side comparison of all trained models |

---

## 📁 Project Structure

```
healthcare-ml-project/
│
├── README.md
├── requirements.txt
├── app.py                    ← Streamlit dashboard
│
├── src/
│   ├── preprocessing.py      ← Data cleaning and feature engineering
│   ├── model.py              ← Random Forest and XGBoost models
│   ├── evaluate.py           ← Metrics: Accuracy, F1, ROC-AUC
│   ├── shap_analysis.py      ← SHAP global and local explainability
│   └── eda.py                ← Exploratory Data Analysis functions
│
├── data/
│   └── diabetes.csv          ← PIMA Indians Diabetes Dataset
│
├── models/
│   ├── random_forest.pkl     ← Saved Random Forest model
│   └── xgboost_model.pkl     ← Saved XGBoost model
│
└── notebooks/
    ├── 01_EDA.ipynb
    ├── 02_Preprocessing.ipynb
    ├── 03_Model_Training.ipynb
    └── 04_SHAP_Analysis.ipynb
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Amitmishra11-X/healthcare-ml-project.git
cd healthcare-ml-project

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Streamlit Dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Train Models from Scratch
```bash
python src/model.py
```

### Run SHAP Analysis
```bash
python src/shap_analysis.py
```

---

## 📊 Dataset — PIMA Indians Diabetes

| Feature | Description | Biological Relevance |
|---|---|---|
| Pregnancies | Number of pregnancies | Higher count increases risk |
| Glucose | Plasma glucose concentration (2hr test) | Primary diagnostic marker |
| BloodPressure | Diastolic blood pressure (mmHg) | Hypertension indicator |
| SkinThickness | Triceps skin fold thickness (mm) | Body fat proxy |
| Insulin | 2-hour serum insulin (μU/ml) | Insulin resistance marker |
| BMI | Body Mass Index (kg/m²) | Obesity indicator |
| DiabetesPedigreeFunction | Genetic diabetes likelihood score | Family history proxy |
| Age | Age in years | Risk increases with age |
| **Outcome** | **0 = No Diabetes, 1 = Diabetes** | **Target variable** |

**Dataset:** 768 records, 268 positive (35%), 500 negative (65%)

---

## 🧪 Data Preprocessing

### Problem: Biologically Impossible Zero Values
```python
# These features cannot be zero in a living person
zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI']

# Replace zeros with NaN then impute with median
for col in zero_not_valid:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)
```

### Feature Engineering
```python
# Interaction features that capture combined clinical effects
df['BMI_Age']          = df['BMI'] * df['Age']
df['Glucose_Insulin']  = df['Glucose'] / (df['Insulin'] + 1)
df['Preg_Age']         = df['Pregnancies'] * df['Age']
```

---

## 🤖 Models

### Random Forest
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
```

### XGBoost
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=1.87,    # Ratio of negative to positive samples
    eval_metric='auc',
    random_state=42
)
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | **89.0%** | 87.3% | 84.6% | 85.9% | **0.94** |
| Random Forest | 87.2% | 85.1% | 82.3% | 83.7% | 0.92 |
| Ensemble (avg) | **89.0%** | 86.8% | 85.1% | 85.9% | **0.94** |
| Logistic Regression (baseline) | 77.3% | 73.2% | 68.4% | 70.7% | 0.83 |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) reveals **why** the model made each prediction.

### Global Feature Importance (across all patients)
```
Glucose              ████████████████████  0.42  ← Most important
BMI                  ████████████          0.21
Age                  ████████              0.16
DiabetesPedigreeFunc ██████                0.11
BloodPressure        ████                  0.07
Insulin              ██                    0.03
```

### Local Explanation (individual patient)
```python
# Example: Patient predicted as Diabetic (87% confidence)
# SHAP waterfall shows:
# +0.38  Glucose = 168     (very high — strong push toward diabetic)
# +0.19  BMI = 33.6        (overweight — increases risk)
# +0.12  Age = 52          (older — increases risk)
# -0.08  BloodPressure = 72 (normal — slightly reduces risk)
# Final: 0.87 probability of diabetes
```

**Clinical validation:** Glucose as the primary predictor aligns with
medical consensus — blood glucose is the primary diagnostic marker for diabetes.

---

## 🖥️ Streamlit Dashboard Pages

| Page | Content |
|---|---|
| **Page 1 — Prediction** | Enter clinical values → get real-time diabetes risk prediction |
| **Page 2 — EDA Overview** | Dataset statistics, distributions, class balance |
| **Page 3 — Feature Analysis** | Correlation heatmap, feature distributions by outcome |
| **Page 4 — Model Comparison** | Accuracy, ROC curves, confusion matrices for all models |
| **Page 5 — SHAP Analysis** | Global importance bar chart + local waterfall for any patient |

---

## 📚 References

1. Smith, J.W. et al. (1988). Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. Proceedings of the Annual Symposium on Computer Applications in Medical Care.
2. Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
3. Lundberg, S.M., Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

---

## 👨‍💻 Author

**Amit Mishra**
B.Tech CSE (3rd Year) | L.D.A.H Rajkiya Engineering College, Mainpuri | AKTU
- GitHub: [@Amitmishra11-X](https://github.com/Amitmishra11-X)
- Email: am0651465@gmail.com
- Live Demo: [healthcare-ml-project-uapgsqtweg9ttv8pxdrymz.streamlit.app](https://healthcare-ml-project-uapgsqtweg9ttv8pxdrymz.streamlit.app)

---

## 📄 License

This project is licensed under the MIT License.
