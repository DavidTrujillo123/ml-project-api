from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar el modelo de regresión logística
log_reg = joblib.load('api/models/logistic_regression_model.pkl')

# Cargar el modelo de red neuronal
model_sigmoid = load_model('api/models/model_sigmoid.h5')
model_relu = load_model('api/models/model_relu.h5')

@app.route('/', methods=['GET'])
def index():
    return 'API de predicción de Diabetes'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)  # Asegúrate de que los datos estén en el formato correcto

    # Predicción con el modelo de regresión logística
    log_reg_prediction = log_reg.predict(input_data)

    # Predicción con el modelo de red neuronal (sigmoide)
    model_sigmoid_prediction = model_sigmoid.predict(input_data)

    # Predicción con el modelo de red neuronal (relu)
    model_relu_prediction = model_relu.predict(input_data)

    return jsonify({
        'log_reg_prediction': log_reg_prediction.tolist(),
        'model_sigmoid_prediction': model_sigmoid_prediction.tolist(),
        'model_relu_prediction': model_relu_prediction.tolist()
    })

# if __name__ == '__main__':
#     app.run(debug=False)
