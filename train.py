import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import pandas as pd
raise Exception("intentional failure for testing")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("assignment5-pipeline")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Log to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy}")

    # Export the Run ID to a file
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)