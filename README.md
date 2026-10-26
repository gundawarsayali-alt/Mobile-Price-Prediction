# Mobile Price Prediction Using Machine Learning

## Project Overview
This project predicts the price range of a mobile phone using machine learning based on its specifications. It also provides a Streamlit web interface for user interaction.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Machine Learning

## Project Files
- `app.py` – Streamlit GUI application
- `best_model.pkl` – Trained classification model
- `scaler.pkl` – Feature scaler for classification
- `best_reg_model.pkl` – Trained regression model
- `reg_scaler.pkl` – Feature scaler for regression
- `requirements.txt` – Required Python libraries
- Dataset CSV file – Data used for model development
- Colab notebook (`.ipynb`) – Data preprocessing, model training and evaluation

## How to Run the Project

1. Install Python.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the Streamlit application:

   `python -m streamlit run app.py`

4. Open the local URL shown in the terminal.

## Project Objective
To use mobile phone specifications to predict the price range and provide price predictions through a simple web application.
