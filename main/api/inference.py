import numpy as np
from tensorflow.keras.models import load_model
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
MODEL_BUCKET = "samsung-model"
MODEL_NAME = "samsung_lstm_model.h5"
SCALER_NAME = "scaler.gz"

# Cache model and scaler
MODEL = None
SCALER = None

# Function: Load model and scaler from MinIO
def load_model_and_scaler():
    global MODEL, SCALER

    if MODEL is None or SCALER is None:
        # Download model
        model_path = "temp_model.h5"
        MINIO_CLIENT.fget_object(MODEL_BUCKET, MODEL_NAME, model_path)
        MODEL = load_model(model_path)

        # Download scaler
        scaler_path = "temp_scaler.gz"
        MINIO_CLIENT.fget_object(MODEL_BUCKET, SCALER_NAME, scaler_path)
        SCALER = joblib.load(scaler_path)

        # Cleanup local files
        os.remove(model_path)
        os.remove(scaler_path)

    return MODEL, SCALER

# Function: Perform prediction
def get_predictions(input_data):
    model, scaler = load_model_and_scaler()

    # Reshape input data for prediction
    input_data = np.array(input_data).reshape(-1, 1)
    input_data_scaled = scaler.transform(input_data)
    sequence_length = 5

    X = []
    X.append(input_data_scaled[-sequence_length:])
    X = np.array(X)

    # Predict next 5 days
    predictions = []
    for _ in range(5):
        pred = model.predict(X)
        predictions.append(pred[0][0])
        # Append prediction to sequence for next prediction
        X = np.append(X[:, 1:, :], [[pred]], axis=1)

    # Inverse transform predictions to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions
