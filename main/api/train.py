import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from minio import Minio
import joblib
import os
from io import BytesIO
import json

# MinIO Configuration
MINIO_CLIENT = Minio(
    "localhost:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)
BUCKET_NAME = "stock-data"
MODEL_BUCKET = "samsung-model"
MODEL_NAME = "samsung_lstm_model.h5"
SCALER_NAME = "scaler.gz"

# Function: Load dataset from MinIO
def load_latest_data():
    objects = MINIO_CLIENT.list_objects(BUCKET_NAME, recursive=True)
    latest_file = max(objects, key=lambda obj: obj.last_modified)
    response = MINIO_CLIENT.get_object(BUCKET_NAME, latest_file.object_name)
    data = json.loads(response.read().decode('utf-8'))
    closes = np.array(data['closes']).reshape(-1, 1)
    return closes

# Function: Prepare LSTM training data
def prepare_data(data, sequence_length=5):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Function: Define LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main script
if __name__ == "__main__":
    try:
        # Load latest dataset from MinIO
        print("Loading dataset from MinIO...")
        data = load_latest_data()
        
        # Normalize the dataset
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Prepare LSTM training data
        print("Preparing training data...")
        sequence_length = 5
        X, y = prepare_data(data_scaled, sequence_length)

        # Build and train the model
        print("Building the LSTM model...")
        model = build_model(input_shape=(sequence_length, 1))
        print("Training the model...")
        model.fit(X, y, epochs=20, batch_size=16, verbose=2)
        print("Model training complete!")

        # Save the trained model locally
        local_model_path = MODEL_NAME
        model.save(local_model_path)
        print(f"Model saved locally as {local_model_path}")

        # Upload the model to MinIO
        print("Uploading model to MinIO...")
        MINIO_CLIENT.fput_object(MODEL_BUCKET, MODEL_NAME, local_model_path)

        # Save and upload the scaler
        local_scaler_path = SCALER_NAME
        joblib.dump(scaler, local_scaler_path)
        print(f"Scaler saved locally as {local_scaler_path}")

        print("Uploading scaler to MinIO...")
        MINIO_CLIENT.fput_object(MODEL_BUCKET, SCALER_NAME, local_scaler_path)

        # Cleanup local files
        os.remove(local_model_path)
        os.remove(local_scaler_path)
        print("Local temporary files cleaned up.")

        print("Training complete and files uploaded successfully to MinIO!")

    except Exception as e:
        print(f"An error occurred: {e}")
