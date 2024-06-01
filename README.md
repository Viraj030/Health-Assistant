# Health-Assistant

## About
Health-Assistant is a project aimed at assisting in the monitoring and management of various health conditions using data analysis and machine learning. The project includes models and scripts for predicting blood pressure, cholesterol levels, obesity, diabetes, and malaria detection using a convolutional neural network (CNN).

## Features
- **Blood Pressure Prediction**: Analyze blood pressure data and predict future readings.
- **Cholesterol Level Analysis**: Monitor and predict cholesterol levels.
- **Obesity Prediction**: Assess the risk of obesity based on various health metrics.
- **Diabetes Prediction**: Predict the likelihood of diabetes from patient data.
- **Malaria Detection**: Use CNN for detecting malaria from blood smear images.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Viraj030/Health-Assistant.git
    cd Health-Assistant
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Running the Application
1. Ensure you have all dependencies installed.
2. Run the application using:
    ```sh
    streamlit run app.py
    ```

### Jupyter Notebooks
- Use Jupyter notebooks for detailed analysis and visualization:
    - `BloodPressure.ipynb`
    - `Cholestrol.ipynb`
    - `diabetes.ipynb`
    - `obesity.ipynb`
    - `malaria_CNN.ipynb`

For the malaria prediction, the notebook will provide a GitHub link to download the dataset required for running the model.

### Data Files
- The project includes CSV files with sample data:
    - `BloodPressure.csv`
    - `Cholestrol.csv`
    - `diabetes.csv`
    - `Obesity.csv`

