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

## Dataset
The model was trained on a dataset containing labeled examples of spam and non-spam text messages. The dataset used for training is not included in this repository due to its size, but you can use a similar dataset for training your model. You can find the dataset in Spam.csv file.

## Model Training
The model is trained using a supervised classification learning approach. The training script and model details can be found in the webapp.py file.

## Acknowledgments
The project structure and web app template are inspired by the [Streamlit Official Documentation](https://docs.streamlit.io/) .
The machine learning model is trained using resources from the [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html).

Feel free to explore and enhance the project as needed. If you encounter any issues or have suggestions for improvement, please open an issue.
