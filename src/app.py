"""
Healthcare Data Analytics & Disease Prediction
Streamlit Web Application - Final Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Find Root Directory ───────────────────────────────────────────
# Works on local machine AND Streamlit Cloud
THIS_FILE = os.path.abspath(__file__)
SRC_DIR   = os.path.dirname(THIS_FILE)
ROOT      = os.path.dirname(SRC_DIR)

MODELS_DIR  = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA_DIR    = os.path.join(ROOT, 'data')

# ── Load Models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf  = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
        xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        sc  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        imp = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
        with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
            feat = json.load(f)
        return rf, xgb, sc, imp, feat, True
    except Exception as e:
         
         return None, None, None, None, None, False

# ── Load Data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))
    except:
        return None

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Navigation")
    page = st.radio("Go to", [
        "🏠 Home & Prediction",
        "📊 EDA Dashboard",
        "📈 Model Performance",
        "ℹ️ About"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** PIMA Indians Diabetes")
    st.markdown("**Records:** 768 patients")
    st.markdown("**Features:** 8 clinical features")
    st.markdown("---")
    st.markdown("**Made by:** Amit Mishra")
    st.markdown("**B.Tech CSE**")
    st.markdown("**LDAH Rajkiya Engg. College**")

# ══════════════════════════════════════════════════════
# PAGE 1 — HOME & PREDICTION
# ══════════════════════════════════════════════════════
if page == "🏠 Home & Prediction":

    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("*AI-powered clinical decision support using Random Forest & XGBoost with SHAP Explainability*")
    st.markdown("---")

    rf_model, xgb_model, scaler, imputer, feature_names, loaded = load_models()

    if not loaded:
        st.error("⚠️ Models not found. Please check models folder.")
    else:
        st.success("✅ Models loaded successfully!")
        st.subheader("Patient Clinical Data Input")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🩸 Blood Measurements**")
            glucose        = st.slider("Glucose (mg/dL)", 44, 200, 120)
            insulin        = st.slider("Insulin (μU/mL)", 0, 846, 80)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 24, 122, 72)

        with col2:
            st.markdown("**📏 Physical Measurements**")
            bmi            = st.slider("BMI (kg/m²)", 18.0, 67.0, 28.0, step=0.1)
            skin_thickness = st.slider("Skin Thickness (mm)", 7, 99, 23)
            pregnancies    = st.slider("Pregnancies", 0, 17, 2)

        with col3:
            st.markdown("**🧬 Genetic & Age**")
            dpf = st.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47, step=0.01)
            age = st.slider("Age (years)", 21, 81, 35)

        st.markdown("")

        if st.button("🔮 Predict Diabetes Risk", use_container_width=True):

            base_features = [pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]
            cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            base_df = pd.DataFrame([base_features], columns=cols)

            for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
                base_df[c] = base_df[c].replace(0, np.nan)

            imputed = pd.DataFrame(imputer.transform(base_df), columns=cols)
            imputed['BMI_Age']              = imputed['BMI'] * imputed['Age']
            imputed['Glucose_Insulin']      = imputed['Glucose'] / (imputed['Insulin'] + 1)
            imputed['DiabetesPedigree_Age'] = imputed['DiabetesPedigreeFunction'] * imputed['Age']

            scaled = scaler.transform(imputed)

            rf_prob  = rf_model.predict_proba(scaled)[0][1]
            xgb_prob = xgb_model.predict_proba(scaled)[0][1]
            avg_prob = (rf_prob + xgb_prob) / 2
            pred     = 1 if avg_prob >= 0.5 else 0

            st.markdown("---")
            st.subheader("📋 Prediction Results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Random Forest",    f"{rf_prob*100:.1f}%",  "Risk")
            c2.metric("XGBoost",          f"{xgb_prob*100:.1f}%", "Risk")
            c3.metric("Ensemble Average", f"{avg_prob*100:.1f}%", "Risk")
            c4.metric("Result", "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK")

            if pred == 1:
                st.error(f"⚠️ HIGH RISK OF DIABETES — {avg_prob*100:.1f}% — Please consult a doctor.")
            else:
                st.success(f"✅ LOW RISK OF DIABETES — {avg_prob*100:.1f}% — Continue regular health monitoring.")

            # Feature bar chart
            st.markdown("---")
            st.subheader("📊 Patient Feature Values")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(cols, base_features, color='#2980B9')
            ax.set_xlabel('Value')
            ax.set_title('Patient Clinical Values', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ══════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    df = load_data()

    if df is None:
        st.error("Dataset not found!")
    else:
        tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Correlations"])

        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", len(df))
            c2.metric("Features",      len(df.columns) - 1)
            c3.metric("Diabetic",      int(df['Outcome'].sum()))
            c4.metric("Non-Diabetic",  int((df['Outcome'] == 0).sum()))

            st.markdown("**Sample Data**")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("**Statistical Summary**")
            st.dataframe(df.describe().round(2), use_container_width=True)

        with tab2:
            feature = st.selectbox("Select Feature", df.columns[:-1])
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(df[df['Outcome']==0][feature], alpha=0.6,
                        label='No Diabetes', color='#27ae60', bins=25)
            axes[0].hist(df[df['Outcome']==1][feature], alpha=0.6,
                        label='Diabetes',    color='#e74c3c', bins=25)
            axes[0].set_title(f'{feature} Distribution', fontweight='bold')
            axes[0].legend()
            df.boxplot(column=feature, by='Outcome', ax=axes[1])
            axes[1].set_title(f'{feature} by Outcome', fontweight='bold')
            axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab3:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt='.2f',
                       cmap='coolwarm', ax=ax, linewidths=0.5)
            ax.set_title('Feature Correlation Heatmap', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.title("📈 Model Performance")
    st.markdown("---")

    st.subheader("Model Comparison Table")
    results = pd.DataFrame({
        'Model':    ['Random Forest', 'XGBoost'],
        'Accuracy': ['74.68%', '72.73%'],
        'F1 Score': ['0.642',  '0.631'],
        'ROC-AUC':  ['0.821',  '0.813'],
        'CV AUC':   ['0.829',  '0.806'],
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Result Charts")

    images = {
        "Confusion Matrices":  "05_confusion_matrices.png",
        "ROC Curves":          "06_roc_curves.png",
        "Feature Importance":  "07_feature_importance.png",
        "SHAP Summary":        "08_shap_summary.png",
        "SHAP Bar":            "09_shap_bar.png",
    }

    for title, fname in images.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_column_width=True)
        else:
            st.info(f"{title} — chart not found")

# ══════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ## 🏥 Healthcare Data Analytics & Disease Prediction

    End-to-end ML pipeline for diabetes risk prediction with SHAP explainability.

    ### 🔧 Tech Stack
    | Component | Technology |
    |---|---|
    | Language | Python 3.11 |
    | Data Processing | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn |
    | ML Models | Scikit-learn (Random Forest), XGBoost |
    | Explainability | SHAP |
    | Web App | Streamlit |
    | Dataset | PIMA Indians Diabetes — UCI ML Repository |

    ### 📊 Results
    | Metric | Score |
    |---|---|
    | Best Model | Random Forest |
    | Accuracy | 74.68% |
    | ROC-AUC | 0.821 |
    | CV AUC | 0.829 |

    ### 👨‍💻 Author
    **Amit Mishra**
    B.Tech CSE — LDAH Rajkiya Engineering College, Mainpuri
    GitHub: [Amitmishra11-X](https://github.com/Amitmishra11-X)
    """)
