import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from minio import Minio
from io import BytesIO
import joblib
import os

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
DATASET_PATH = "samsung_stock.csv"

# Function: Load and validate dataset
def load_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found.")
    df = pd.read_csv(filepath)
    if 'Close' not in df.columns:
        raise ValueError("Dataset does not contain the 'Close' column.")
    return df['Close'].values.reshape(-1, 1)

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

# Function: Save file to MinIO
def save_to_minio(client, bucket_name, object_name, local_path):
    try:
        client.fput_object(bucket_name, object_name, local_path)
        print(f"Uploaded to MinIO: {object_name}")
    except Exception as e:
        print(f"Error uploading {object_name} to MinIO: {e}")

# Main script
if __name__ == "__main__":
    try:
        # Ensure MinIO bucket exists
        if not MINIO_CLIENT.bucket_exists(BUCKET_NAME):
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
            print(f"Bucket '{BUCKET_NAME}' created in MinIO.")

        # Load and normalize dataset
        print("Loading dataset...")
        data = load_dataset(DATASET_PATH)
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

        # Save the trained model locally in .h5 format
        local_model_path = "samsung_lstm_model.h5"
        print("Saving model locally...")
        model.save(local_model_path)
        print(f"Model saved locally as {local_model_path}")

        # Upload the model to MinIO
        print("Uploading model to MinIO...")
        save_to_minio(MINIO_CLIENT, BUCKET_NAME, MODEL_NAME, local_model_path)

        # Save and upload the scaler
        print("Saving scaler locally...")
        local_scaler_path = "scaler.gz"
        joblib.dump(scaler, local_scaler_path)
        print(f"Scaler saved locally as {local_scaler_path}")

        print("Uploading scaler to MinIO...")
        save_to_minio(MINIO_CLIENT, BUCKET_NAME, SCALER_NAME, local_scaler_path)

        # Cleanup local files
        os.remove(local_model_path)
        os.remove(local_scaler_path)
        print("Local temporary files cleaned up.")

        print("All files uploaded successfully to MinIO!")

    except Exception as e:
        print(f"An error occurred: {e}")
