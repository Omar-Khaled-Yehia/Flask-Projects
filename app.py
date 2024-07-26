from flask import Flask, request, render_template, jsonify
from joblib import load
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model
model = load('Maids.cc_Model(1).pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[
        data['battery_power'],
        data['clock_speed'],
        data['int_memory'],
        # Include all features used in the model
    ]])
    prediction = model.predict(features)
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
