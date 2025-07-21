# Employee Salary Prediction

## Overview

This project is an end-to-end machine learning solution for predicting whether an employee earns more than 50K or not, based on demographic and work-related features. The solution includes data preprocessing, model training, evaluation, and a user-friendly web application for predictions.

## Features

- **Data Cleaning & Preprocessing:**  
  Handles missing values, encodes categorical variables, removes outliers, and drops redundant features.

- **Model Training & Evaluation:**  
  Multiple classification algorithms are compared (Logistic Regression, Random Forest, KNN, SVM, Gradient Boosting). The best model is selected based on accuracy and saved for deployment.

- **Web Application:**  
  A Streamlit app allows users to input employee details and predict salary class. Batch prediction via CSV upload is also supported.

- **Deployment:**  
  The app can be run locally or exposed to the internet using ngrok.

## Dataset

The project uses the [Adult Income Dataset](https://www.kaggle.com/datasets/sohaibanwaar1203/adultscsv) (also known as "Census Income" or "Census Salary" dataset).  
**Features include:**  
- Age  
- Workclass  
- Education & Educational Number  
- Marital Status  
- Occupation  
- Relationship  
- Race  
- Gender  
- Capital Gain/Loss  
- Hours per Week  
- Native Country  
- Income (target variable: `<=50K` or `>50K`)

## Project Structure

```
.
├── adults.csv
├── employee salary prediction.ipynb
├── app.py
├── best_model.pkl
└── README.md
```

## How to Run

1. **Install dependencies:**
    ```bash
    pip install pandas scikit-learn streamlit matplotlib joblib pyngrok
    ```

2. **Train the model:**  
   Run the Jupyter notebook `employee salary prediction.ipynb` to preprocess data and train the model. This will generate `best_model.pkl`.

3. **Start the app:**
    ```bash
    streamlit run app.py
    ```

4. **(Optional) Expose app with ngrok:**
    ```bash
    python -m pyngrok http 8080
    ```

## Usage

- **Single Prediction:**  
  Enter employee details in the sidebar and click "Predict Salary Class" to see the prediction.

- **Batch Prediction:**  
  Upload a CSV file with employee data to get predictions for multiple records at once.

## Model Performance

The notebook compares several models and selects the best based on test accuracy.  
A bar chart visualizes the performance of each model.

## Author

- Developed by Harsh Lagwal
- For educational and demonstration purposes.

---
