from flask import Flask
from flask import request
from flask import jsonify

import os
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, f'model_C=1.0.bin')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('obesityPredict')

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    result = {
        'obesity_probability': float(y_pred),
        'obesity': bool(y_pred >= 0.5)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)