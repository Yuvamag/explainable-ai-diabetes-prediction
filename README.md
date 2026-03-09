# Explainable AI for Diabetes Risk Prediction

## Project Overview
This project builds a **Machine Learning model to predict diabetes risk** and explains the predictions using **Explainable AI (XAI) techniques**.

The goal is not only to classify whether a patient is at **high or low risk of diabetes**, but also to provide **interpretability** so users understand **why the model made a specific prediction**.

The system also includes a **Streamlit interface** where users can input patient data and view predictions along with explanations.

---

## Features

- Diabetes risk prediction using **XGBoost**
- Model evaluation using accuracy and confusion matrix
- Explainable AI techniques
- Feature Importance visualization
- SHAP explanations
- Counterfactual explanations
- Interactive **Streamlit web interface**

---

## Dataset

The dataset contains medical attributes commonly used in diabetes prediction.

Example features:

- Glucose
- Blood Pressure
- BMI
- Insulin
- Age
- Diabetes Pedigree Function

Dataset included in this repository:

```
setdiab.csv
```

---

## Project Structure

```
diabetes-xai-prediction
│
├── diabetes_prediction_xai.ipynb
├── diabetes_prediction_xai.py
├── setdiab.csv
├── requirements.txt
├── README.md
│
└── images
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── shap_plot.png
    └── streamlit_interface.png
```

---

## Machine Learning Workflow

1. Load and preprocess the dataset
2. Split data into training and testing sets
3. Train the **XGBoost classifier**
4. Evaluate model performance
5. Generate explainability outputs
6. Deploy prediction interface using Streamlit

---

## Explainable AI Techniques Used

### Feature Importance
Shows which features contribute most to the prediction.

### SHAP (SHapley Additive Explanations)
Explains how each feature influences the prediction output.

### Counterfactual Explanation
Shows how **small changes in patient attributes** could change the prediction outcome.

Example:

```
If BMI decreases from 34 → 28
and Glucose decreases from 180 → 140

Prediction could change from High Risk → Low Risk
```

---

## Installation

Install the required Python libraries using:

```
pip install -r requirements.txt
```

---

## Running the Streamlit App

Run the application using:

```
streamlit run diabetes_prediction_xai.py
```

This will open the **Streamlit interface in your browser**, where users can:

- Enter patient medical parameters
- Get diabetes risk prediction
- View explanation plots

---

## Example Outputs

The project generates visual outputs including:

- Confusion Matrix
- Feature Importance Plot
- SHAP Explanation Plot
- Streamlit Prediction Interface

Screenshots of these outputs are available in the **images/** folder.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- SHAP
- Streamlit

---

## Future Improvements

- Improve counterfactual explanation generation
- Add additional medical datasets
- Compare multiple ML models
- Deploy the application online

---

## Author

Student Project – Machine Learning & Explainable AI
