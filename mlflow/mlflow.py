import os
import warnings
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from urllib.parse import urlparse
import wget
import mlflow
import mlflow.sklearn
import logging
from zipfile import ZipFile

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def preprocessing(df):
    X = df.drop(['instant', 'dteday', 'atemp', 'casual', 'registered', 'cnt'], axis=1).values
    y = df['cnt'].values
    return train_test_split(X, y, test_size=0.2)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    zip_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    )
    wget.download(zip_url)
    try:
        with ZipFile('Bike-Sharing-Dataset.zip', "r") as z:
            with z.open("hour.csv") as f:
                df = pd.read_csv(f, delimiter=',')
                print(df.head())

    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    X_train, X_test, y_train, y_test = preprocessing(df)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_features = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    with mlflow.start_run():
        rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)
        model = rf.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)
        print("Random Forest model (n_estimators=%d, max_features=%d, max_depth=%d):" % (
        n_estimators, max_features, max_depth))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(rf, "model", registered_model_name="RandomForestModel")
        else:
            mlflow.sklearn.log_model(rf, "model")