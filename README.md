# Obesity Prediction

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A simple web app to predict the obesity level of a person based on multiple health and lifestyle features.

---

## Overview
Obesity depends on several factors, not just weight. In this project, machine learning models are trained on a dataset to predict whether a person is obese based on various features.

This project includes:
- Data exploration (EDA)
- Model training (Logistic Regression, Decision Tree, Random Forest)
- A Flask-based web app
- Docker containerization for easy deployment

---

## Dataset
- Source: [Kaggle - Obesity Prediction Dataset](https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction)
- Number of samples: 2111
- Number of features: 16 + 1 (Target Column)

**Note:** Place the downloaded CSV file in a `data/` folder inside the project.

---

## Exploratory Data Analysis (EDA)
EDA helps understand the dataset and identify important patterns before model training.
- EDA is performed in [notebooks/01-eda.ipynb](notebooks/01-eda.ipynb)

---

## Models
The following machine learning models were trained to predict obesity level:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## Web App (Flask)
A Flask web app allows you to input features and get a prediction for obesity level.

**Running the app:**
```bash
python src/predict.py
```

After running, visit `http://localhost:9696` in your browser.

**Testing the app:**
```bash
python src/predict-test.py
```

---

## Environment Setup
Python 3.10+ is recommended.

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate   # Linux/Mac
.\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Docker Deployment
To build and run the app locally using Docker:

```bash
# Build Docker image
docker build -t obesity-test .

# Run container
docker run -it --rm -p 9696:9696 obesity-test
```

Visit `http://localhost:9696` to use the app.

---

## Features
| Feature | Description |
|---------|-------------|
| Gender | Gender |
| Age | Age |
| Height | in metres |
| Weight | in kgs |
| family_history | Has a family member suffered or suffers from overweight? |
| FAVC | Do you eat high caloric food frequently? |
| FCVC | Do you usually eat vegetables in your meals? |
| NCP | Number of main meals per day |
| CAEC | Do you eat any food between meals? |
| SMOKE | Do you smoke? |
| CH2O | Daily water intake |
| SCC | Do you monitor daily calorie intake? |
| FAF | Frequency of physical activity |
| TUE | Time using technological devices daily |
| CALC | Alcohol consumption frequency |
| MTRANS | Usual mode of transportation |
| Obesity_level (Target) | Obesity level |

---

## TODO
- Remove weight column to test prediction using habits only
- Perform correlation and causation analysis on habits vs obesity
- Deploy the app to cloud

---

## License
This project is licensed under the MIT License.

