# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import numpy as np
import os
import pickle

#path to save model

# Save the model

# in the same folder as this script
C = 1.0
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, f'model_C={C}.bin')


df = pd.read_csv(os.path.join(os.path.dirname(script_dir), 'data', 'raw', 'Obesity prediction.csv'))

obesity_level = df.groupby("Obesity")["Weight"].median().sort_values().reset_index().reset_index(drop = False).rename(columns = {"index":"Obesity_level"})
df = df.merge(obesity_level[["Obesity_level", "Obesity"]], on = "Obesity", how = "left")
df[["Obesity", "Obesity_level"]].drop_duplicates().to_dict("records")
df["Obesity_binary"] = (df["Obesity_level"] > 3).astype(int)

# Remember to drop columns that can leak into training data. We are not dropping the main target variable yet. 

df.drop(columns = ["Obesity", "Obesity_level"], inplace = True)

# ## Setting up validation framework

from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# now we take the target variable out of the training data
y_train = df_train["Obesity_binary"].values
y_val = df_val["Obesity_binary"].values
y_test = df_test["Obesity_binary"].values
# and drop it from the training data
del df_train["Obesity_binary"]
del df_val["Obesity_binary"]
del df_test["Obesity_binary"]

# ## Model 1. Logistic Regression

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def train_LR(df_train, y_train, C=1.0):

    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict_LR(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# ### Parameter tuning
# 
# The parameter we want to tune is C. We will use cross-validation to find the best value of C.

n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.Obesity_binary.values
        y_val = df_val.Obesity_binary.values

        dv, model = train_LR(df_train, y_train, C=C)
        y_pred = predict_LR(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# Based on this we choose C = 0.1

dv, model = train_LR(df_full_train, df_full_train.Obesity_binary.values, C=0.1)
y_pred = predict_LR(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print("Final auc", auc)



# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')

# We have good score on the test dataset. Even though auc is perfect on training dataset, it is not bad on test dataset. Implying model is not overfitting.