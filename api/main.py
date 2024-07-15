from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Cargar el scaler
scaler = joblib.load('api/models/scaler.pkl')
CORS(app)

# Cargar el modelo de regresión logística
log_reg = joblib.load('api/models/logistic_regression_model.pkl')

# # Cargar el modelo de red neuronal
model_sigmoid = load_model('api/models/model_sigmoid.h5')
# model_relu = load_model('api/models/model_relu.h5')

@app.route('/', methods=['GET'])
def index():
    return 'API de predicción de Diabetes'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    new_data = pd.DataFrame([data['input']], columns=columns)

    # Estandarizar los nuevos datos utilizando el scaler ajustado
    new_data_scaled = scaler.transform(new_data)

#     # Predicción con el modelo de regresión logística
    log_reg_prediction = log_reg.predict(new_data_scaled)

#     # Predicción con el modelo de red neuronal (sigmoide)
    model_sigmoid_prediction = model_sigmoid.predict(new_data_scaled)

#     # Predicción con el modelo de red neuronal (relu)
    # model_relu_prediction = model_relu.predict(new_data_scaled)

    return jsonify({
        'log_reg_prediction': log_reg_prediction.tolist()[0],
        'model_sigmoid_prediction': model_sigmoid_prediction.tolist()[0][0],
        # 'model_relu_prediction': model_relu_prediction.tolist()[0][0],
        'coeficientes_regresion_log': log_reg.coef_.tolist()[0]

    })

if __name__ == '__main__':
    app.run(debug=True)
