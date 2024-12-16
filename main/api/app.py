from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from producer import send_to_kafka
from inference import get_predictions
import subprocess

app = Flask(__name__, static_folder='../web')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('../web', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'close' not in data:
            return jsonify({"error": "Invalid input, 'close' data is required"}), 400

        closes = [float(price) for price in data['close']]
        dates = data.get('date', [None] * len(closes))

        # Kirim data ke Kafka
        send_to_kafka(dates, closes)

        # Dapatkan prediksi dari model inference
        predictions = get_predictions(closes)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Trigger training script
        result = subprocess.run(["python3", "train.py"], cwd="./api", capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return jsonify({"message": "Model training completed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
