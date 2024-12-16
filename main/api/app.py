from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from producer import send_to_kafka
from inference import get_predictions
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from minio import Minio
import joblib
import os
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MinIO Configuration
MINIO_CLIENT = Minio(
    "localhost:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)
BUCKET_NAME = "samsung-model"
MODEL_NAME = "samsung_lstm_model.h5"
SCALER_NAME = "scaler.gz"

app = Flask(__name__, static_folder='../web')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('../web', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.json
        if not data or 'close' not in data:
            return jsonify({"error": "Invalid input, 'close' data is required"}), 400

        closes = [float(price) for price in data['close']]
        dates = data.get('date', [None] * len(closes))

        # Kirim data ke Kafka
        send_to_kafka(dates, closes)

        # Dapatkan prediksi dari model inference
        predictions = get_predictions(closes)

        # Pastikan prediksi berupa list Python
        predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

        # Kembalikan hasil prediksi sebagai JSON
        response = {"predictions": predictions}

        # Jalankan proses training setelah prediksi
        train(closes, predictions)  # Menggunakan data prediksi dan input pengguna untuk training

        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def train(user_closes, previous_predictions):
    try:
        logging.info("Starting the training process...")

        # Pastikan bucket MinIO tersedia
        if not MINIO_CLIENT.bucket_exists(BUCKET_NAME):
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
            logging.info(f"Bucket '{BUCKET_NAME}' created in MinIO.")
        else:
            logging.info(f"Bucket '{BUCKET_NAME}' already exists.")

        # Normalisasi data
        scaler = MinMaxScaler()

        # Gabungkan data asli (user_closes) dan hasil prediksi sebelumnya (previous_predictions)
        combined_data = np.array(user_closes + previous_predictions).reshape(-1, 1)
        data_scaled = scaler.fit_transform(combined_data)

        # Persiapkan data untuk LSTM
        sequence_length = 5
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length])
            y.append(data_scaled[i + sequence_length])
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)

        # Bangun model LSTM
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Latih model
        logging.info("Training the model...")
        model.fit(X, y, epochs=20, batch_size=16, verbose=2)
        logging.info("Model training complete!")

        # Simpan model ke MinIO
        local_model_path = MODEL_NAME
        model.save(local_model_path)
        MINIO_CLIENT.fput_object(BUCKET_NAME, MODEL_NAME, local_model_path)
        logging.info(f"Model saved and uploaded to MinIO as {MODEL_NAME}")

        # Simpan scaler ke MinIO
        local_scaler_path = SCALER_NAME
        joblib.dump(scaler, local_scaler_path)
        MINIO_CLIENT.fput_object(BUCKET_NAME, SCALER_NAME, local_scaler_path)
        logging.info(f"Scaler saved and uploaded to MinIO as {SCALER_NAME}")

        # Hapus file lokal setelah diupload
        os.remove(local_model_path)
        os.remove(local_scaler_path)
        logging.info("Local temporary files cleaned up.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
