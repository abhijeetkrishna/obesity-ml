# Obesity Prediction

It is important to monitor the obesity level of a person. Whether you are obese or not depends on several factors, not just the weight. In this project, I used a dataset to train machine learning models to predict whether a person is obese or not. 
This is a simple web app that predicts the obesity level of a person based on the given features.

## Dataset 
A simulated dataset taken from Kaggle. Download [here](https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction)
Number of samples : 2111
Number of features : 16 + 1 (Target Column)

## Data exploration (EDA)
Exploratory data analysis is crucial before training machine learning models. I have done EDA in this [notebook](notebooks/01-eda.ipynb).

## Models
I have trained 3 machine learning models to predict the obesity level of a person. The models are:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Web App (Flask)
I have made an app using Flask. One can query this app by entering the features and the app will predict the obesity level of the person. One can run this [script](src/predict-test.py) to test this app.

## Environment management
To run this project, use the [requirements](requirements.txt) file to create a virtual environment and install the required packages.
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Containerization and Deployment
To deploy the app, run the following command:
```bash
docker build -t obesity-test .
docker run -it --rm -p 9696:9696 obesity-test
```

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
| NCP | How many main meals do you have daily? |
| CAEC | Do you eat any food between meals? |
| SMOKE | Do you smoke? |
| CH2O | How much water do you drink daily? |
| SCC | Do you monitor the calories you eat daily? |
| FAF | How often do you have physical activity? |
| TUE | How much time do you use technological devices such as cell phone, videogames, television, computer and others? |
| CALC | How often do you drink alcohol? |
| MTRANS | Which transportation do you usually use? |
| Obesity_level (Target Column) | Obesity level |

## TODO
- Remove weight column and see if one can predict obesity based on just habits 
- Do a correlation/causation analysis on such habits and obesity
- Deploy app to cloud