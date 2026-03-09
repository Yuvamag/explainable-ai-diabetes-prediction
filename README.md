# Explainable AI for Diabetes Risk Prediction

## Project Overview
This project builds a **Machine Learning model to predict diabetes risk** and explains the predictions using **Explainable AI (XAI) techniques**.

The objective is not only to classify whether a person is at **high or low risk of diabetes**, but also to provide **interpretability** so users understand **why the model made a specific prediction**.

The project also includes a **Streamlit web interface** for interactive predictions.

---

## Features

- Diabetes risk prediction using Machine Learning  
- Model evaluation using accuracy and confusion matrix  
- Explainable AI techniques for transparency  
- SHAP explanations for feature impact  
- Counterfactual explanations  
- Interactive Streamlit interface  

---

## Dataset

The dataset contains medical attributes related to diabetes risk.

Example features include:

- Glucose Level  
- Blood Pressure  
- BMI  
- Insulin  
- Age  
- Diabetes Pedigree Function  

Dataset file included in this repository:

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

1. Load and preprocess dataset  
2. Train machine learning model  
3. Evaluate model performance  
4. Generate explainability insights  
5. Deploy Streamlit interface  

---

## Explainable AI Techniques Used

### Feature Importance
Shows which features contribute the most to diabetes prediction.

### SHAP (SHapley Additive Explanations)
Explains how each feature influences the prediction output.

### Counterfactual Explanation
Shows the **minimum changes required in input features** to change the prediction outcome.

Example:

```
If BMI decreases from 34 → 28
and Glucose decreases from 180 → 140

Prediction could change from High Risk → Low Risk
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/diabetes-xai-prediction.git
```

Move into the project folder:

```
cd diabetes-xai-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Streamlit App

Run the following command:

```
streamlit run diabetes_prediction_xai.py
```

The application will open in your browser.

You can:

- Enter medical parameters  
- Get diabetes risk prediction  
- View explanation plots  

---

## Example Outputs

The project generates several visual outputs:

- Confusion Matrix  
- Feature Importance Plot  
- SHAP Explanation Plot  
- Streamlit Interface  

(See the **images/** folder for screenshots)

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- SHAP  
- Streamlit  

---

## Future Improvements

- Improve counterfactual explanation generation  
- Add advanced models like Random Forest or XGBoost  
- Deploy the application online  
- Integrate additional medical datasets  

---

## Author

Student Project – Machine Learning and Explainable AI
