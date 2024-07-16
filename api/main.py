from flask import Flask, request, jsonify, render_template
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

# Cargar el modelo de red neuronal
model_sigmoid = load_model('api/models/model_sigmoid.h5')
# model_relu = load_model('api/models/model_relu.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)['input']
        data = {key: float(value) for key, value in data.items()}

        columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        new_data = pd.DataFrame([data], columns=columns)

        # Estandarizar los nuevos datos utilizando el scaler ajustado
        new_data_scaled = scaler.transform(new_data)

        # Predicción con el modelo de regresión logística
        log_reg_prediction = log_reg.predict(new_data_scaled)

        # Predicción con el modelo de red neuronal (sigmoide)
        model_sigmoid_prediction = model_sigmoid.predict(new_data_scaled)

        # Predicción con el modelo de red neuronal (relu)
        # model_relu_prediction = model_relu.predict(new_data_scaled)

        return jsonify({
            'log_reg_prediction': int(log_reg_prediction[0]),
            'model_sigmoid_prediction': model_sigmoid_prediction.tolist()[0][0],
            # 'model_relu_prediction': model_relu_prediction.tolist()[0][0],
            'coeficientes_regresion_log': log_reg.coef_.tolist()[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
