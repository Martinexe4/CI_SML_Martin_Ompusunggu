import json
import pandas as pd
import os
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

#load dataset
df = pd.read_csv("diabetes_processing.csv")

#Splitting data
X = df.drop(columns=['isDiabetes'])
y = df['isDiabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#Set URI
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Nama proyek
mlflow.set_experiment("Diabetes_Classification")

with mlflow.start_run():
    #Inisialisasi dan latih model
    n_estimators = 100
    random_state = 42
    mlflow.autolog()
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    # Prediksi
    y_pred = clf.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log MLFlow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    input_example = X_train.iloc[[0]]

    # Buat folder temp untuk artefak manual
    artifact_dir = "model"
    os.makedirs(artifact_dir, exist_ok=True)

    # Simpan model
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=input_example
    )
    # Log metrics
    accuracy = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)


    print(f"Model trained and logged to MLflow. Accuracy: {acc:.4f}")
