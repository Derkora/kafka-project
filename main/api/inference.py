import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.utils import get_custom_objects
from minio import Minio
import joblib
import os
from sklearn.metrics import mean_squared_error
from keras.saving import register_keras_serializable

# Registrasi custom loss function (jika diperlukan oleh model)
@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Pastikan custom objects dikenali saat model dimuat
get_custom_objects().update({"mse": mse})

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
        print("Loading model and scaler from MinIO...")

        # Download model dari MinIO
        model_path = "temp_model.h5"
        MINIO_CLIENT.fget_object(MODEL_BUCKET, MODEL_NAME, model_path)
        MODEL = load_model(model_path, custom_objects={"mse": mse})
        os.remove(model_path)

        # Download scaler dari MinIO
        scaler_path = "temp_scaler.gz"
        MINIO_CLIENT.fget_object(MODEL_BUCKET, SCALER_NAME, scaler_path)
        SCALER = joblib.load(scaler_path)
        os.remove(scaler_path)

        print("Model and scaler loaded successfully!")

    return MODEL, SCALER

# Function: Perform prediction
def get_predictions(input_data):
    model, scaler = load_model_and_scaler()

    # Reshape dan transform data input
    input_data = np.array(input_data).reshape(-1, 1)
    input_data_scaled = scaler.transform(input_data)
    sequence_length = 5

    # Membentuk sequence data
    X = input_data_scaled[-sequence_length:].reshape(1, sequence_length, 1)

    # Predict next 5 days
    predictions = []
    for _ in range(5):
        pred = model.predict(X, verbose=0)
        predictions.append(pred[0][0])

        new_input = np.array([[pred[0][0]]])
        X = np.append(X[:, 1:, :], new_input.reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

# Function: Evaluate predictions
def evaluate_predictions(actuals, predictions):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse
