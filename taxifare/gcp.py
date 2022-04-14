from google.cloud import storage
from taxifare.params import BUCKET_NAME

def save_model_to_gcp():

    storage_location = "models/recap/random_forest_model.joblib"
    local_model_filename = "model.joblib"

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(local_model_filename)