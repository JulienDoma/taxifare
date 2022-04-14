import pandas as pd
from sklearn.model_selection import train_test_split
from taxifare.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH

def get_data(nb_line):
    url = "s3://wagon-public-datasets/taxi-fare-train.csv"
    df = pd.read_csv(url, nrows=nb_line)
    return df

def clean_df(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 1]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def holdout(df):
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
    
    return X_train, X_test, y_train, y_test

from google.cloud import storage


def get_data_using_pandas(line_count):

    # get data from aws s3
    # url = "s3://wagon-public-datasets/taxi-fare-train.csv"
    # df = pd.read_csv(url, nrows=100)

    # load n lines from my csv
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=line_count)
    return df


def get_data_using_blob(line_count):

    # get data from aws s3
    # url = "s3://wagon-public-datasets/taxi-fare-train.csv"

    # get data from my google storage bucket

    data_file = "train_1k.csv"

    client = storage.Client()  # verifies $GOOGLE_APPLICATION_CREDENTIALS

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)

    blob.download_to_filename(data_file)

    # load downloaded data to dataframe
    df = pd.read_csv(data_file, nrows=line_count)

    return df