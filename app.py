from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('predict_form.html')


def preprocess_input(data):
    columns = [
        'Company', 'Product', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu',
        'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight'
    ]
    df = pd.DataFrame([data], columns=columns)

    # Feature Engineering
    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "").astype(float)
    df.Weight = df.Weight.astype(str).str.replace("kg", "").astype(float)
    df['touch_screen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS_display'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    new = df['ScreenResolution'].str.split('x', n=1, expand=True)
    df['X_res'] = new[0]
    df['y_res'] = new[1]
    df['X_res'] = df['X_res'].apply(lambda x: re.sub(r'\D', '', x)).astype(float)
    df['y_res'] = df['y_res'].astype(float)
    df['PPI'] = (((df['X_res'] ** 2) + (df['y_res'] ** 2)) ** 0.5 / df['Inches']).astype(float)
    df['PPI'] = df['PPI'].round(2)
    df['CPU_name'] = df['Cpu'].apply(lambda x: "".join(x.split()[:3]))

    def processor(x):
        if x in ['IntelCorei5', 'IntelCorei7', 'IntelCorei3']:
            return x
        if x.split()[0] == 'AMD':
            return 'AMD Processor'
        if x.split()[0] == 'Intel':
            return 'Other Intel Processor'
        return 'Samsung Processor'

    df['CPU'] = df['CPU_name'].apply(processor)
    df['Memory'] = df['Memory'].astype(str).replace('\\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '').str.replace('TB', '000')
    df['SSD'] = df['Memory'].apply(
        lambda x: re.findall(r'(\d+) SSD', x)[0] if re.findall(r'(\d+) SSD', x) else 0).astype(int)
    df['HDD'] = df['Memory'].apply(
        lambda x: re.findall(r'(\d+) HDD', x)[0] if re.findall(r'(\d+) HDD', x) else 0).astype(int)
    df['Flash_Storage'] = df['Memory'].apply(
        lambda x: re.findall(r'(\d+) Flash Storage', x)[0] if re.findall(r'(\d+) Flash Storage', x) else 0).astype(int)
    df.drop(columns=['CPU_name', 'Memory', 'Cpu', 'ScreenResolution', 'X_res', 'y_res', 'Product'], inplace=True)

    def GPU(x):
        if x.split()[0] == 'Intel':
            return 'Intel'
        if x.split()[0] == 'AMD':
            return 'AMD'
        if x.split()[0] == 'Nvidia':
            return "Nvidia"
        return "Other"

    df['GPU_brand'] = df['Gpu'].apply(GPU)
    df.drop(columns=['Gpu', 'Inches'], inplace=True)
    df = df[df.GPU_brand != 'ARM']
    df['OpSys'].replace({'Mac OS X': 'Mac', 'macOS': 'Mac'}, inplace=True)

    def OS(x):
        if x.split()[0] == 'Windows':
            return 'Windows'
        if x.split()[0] == 'Mac':
            return 'Mac'
        if x.split()[0] == 'Linux':
            return 'Linux'
        return 'Other OS'

    df['OS'] = df['OpSys'].apply(OS)
    df.drop(columns=['OpSys'], inplace=True)

    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all features from the form
        features = [
            request.form['Company'],
            request.form['Product'],
            request.form['TypeName'],
            float(request.form['Inches']),
            request.form['ScreenResolution'],
            request.form['Cpu'],
            request.form['Ram'],
            request.form['Memory'],
            request.form['Gpu'],
            request.form['OpSys'],
            request.form['Weight']
        ]

        # Preprocess the input data
        df_features = preprocess_input(features)

        # Predict using the model
        log_prediction = model.predict(df_features)

        # Apply the inverse of the log transformation to get the actual price
        prediction = np.exp(log_prediction)

        # Return the prediction to the form
        output = prediction[0]
        return render_template('predict_form.html', prediction_text='Estimated Laptop Price: ${:.2f}'.format(output))
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
