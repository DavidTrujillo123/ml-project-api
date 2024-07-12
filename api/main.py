from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return 'API de predicción de Diabetes'

if __name__ == '__main__':
    app.run(debug=True)
