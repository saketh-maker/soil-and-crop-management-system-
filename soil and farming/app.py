# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    soil_type = data['soil_type']
    region = data['region']

    # Convert inputs to DataFrame
    input_df = pd.DataFrame([[soil_type, region]], columns=['soil_type', 'region'])
    input_df = pd.get_dummies(input_df)
    
    # Ensure all columns are present
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    prediction = model.predict(input_df)
    return jsonify({'crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
