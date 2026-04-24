# 🧠 Explainable AI for Diabetes Prediction using SHAP, LIME & Counterfactual Analysis

## 🚀 Overview
This project builds an **interpretable machine learning system** for predicting diabetes risk using clinical data.

Unlike traditional "black-box" models, this system focuses on **transparency, interpretability, and actionable insights** by combining prediction with human-understandable explanations.

The system integrates:
- High-performance ML model (**XGBoost**)
- Explainability (**SHAP & LIME**)
- Actionable insights (**Counterfactual Explanations**)
- Interactive simulations (**What-if Analysis via Streamlit**)

---

## 🎯 Problem Statement
Early detection of diabetes is critical. However, most machine learning models:
- Lack interpretability  
- Provide no actionable insights  
- Are difficult to trust in healthcare settings  

This project addresses these challenges by building a **transparent and interactive prediction system**.

---

## 📊 Dataset
- **Pima Indians Diabetes Dataset**

### Features:
- Glucose  
- BMI  
- Insulin  
- Blood Pressure  
- Age  
- Diabetes Pedigree Function  

### 🧹 Data Preprocessing
- Replaced invalid zero values using **median imputation**
- Performed **Stratified 80:20 train-test split**

---

## 🤖 Models Used

| Model | Accuracy |
|------|--------|
| Logistic Regression | 70% |
| Random Forest | 75% |
| **XGBoost (Final Model)** | **76.6%** |

### 📈 Final Model Performance
- Accuracy: **76.6%**
- ROC-AUC: **0.822**
- Improved Precision, Recall, and F1-score

---

## 🔍 Explainable AI Techniques

### 1. SHAP (Global + Local Interpretability)
- Identifies feature importance across the dataset  
- Explains individual predictions  

**Key Insight:**  
- Glucose and BMI are dominant risk factors  

---

### 2. LIME (Local Interpretability)
- Explains predictions for individual instances  
- Shows feature contribution (positive/negative impact)

---

### 3. Counterfactual Explanations
- Suggests minimal changes to alter prediction outcomes  

**Example:**
- Reduce Glucose levels  
- Reduce BMI  

➡️ Provides **actionable insights**, not just predictions

---

## 🔄 Additional Simulations

### 🧪 What-if Analysis
- Modify input features dynamically  
- Observe real-time changes in prediction probability  

---

### 💡 Health Recommendation System
- Rule-based personalized suggestions  

**Examples:**
- High Glucose → Reduce sugar intake  
- High BMI → Lifestyle modifications  

---

## 🖥️ Interactive Application
Built using **Streamlit**

### Features:
- Real-time prediction  
- SHAP visualizations  
- LIME explanations  
- What-if simulations  
- Counterfactual insights  
- Health recommendations  

---

## 🛠️ Tech Stack
- Python  
- XGBoost  
- Scikit-learn  
- SHAP  
- LIME  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  

---

## 📌 Key Contributions
- Integrated **prediction + explainability + actionability**
- Built an **interactive simulation system**
- Bridged gap between:
  - Model accuracy  
  - Interpretability  
  - Real-world usability  

---

## 🔮 Future Scope
- Improve model sensitivity (reduce false negatives)  
- Integrate real-time healthcare datasets  
- Extend to multi-class diabetes risk levels  
- Deploy as a clinical decision support tool  

---

## 📷 Results & Screenshots
(Add your outputs here)
- SHAP plots  
- LIME explanations  
- Streamlit UI  
- Simulation results  

---

## 📎 How to Run

```bash
git clone <your-repo-link>
cd diabetes-xai
pip install -r requirements.txt
streamlit run app.py
