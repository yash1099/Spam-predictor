# Spam Predictor Web App

## Overview

This web application aims to predict whether a given text message is spam or not. The prediction is based on a machine learning model trained using supervised classification learning. The app provides a user-friendly interface where users can input text and receive a prediction along with probability scores.

## Features

- **Text Input:** Users can enter a text message into the provided input box.
- **Prediction:** The app will predict whether the entered text is spam or not.
- **Probability Score:** Along with the prediction, users will see a probability score indicating the model's confidence in the prediction.

## Technologies Used

- **Streamlit:** The app is built using Streamlit, a Python library for creating web applications with minimal code.
- **Scikit-learn:** The machine learning model is trained using Scikit-learn, a popular machine learning library in Python.
- **Pandas:** Used for data manipulation and handling the dataset.
- **Joblib:** The trained model is saved and loaded using Joblib for efficient model deployment.

## How to Run the App

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
