import mlflow
import sys
import os

THRESHOLD = 0.85

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0.0)

print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAIL: Accuracy {accuracy} is below threshold {THRESHOLD}")
    sys.exit(1)
else:
    print(f"PASS: Accuracy {accuracy} meets threshold {THRESHOLD}")
    sys.exit(0)
