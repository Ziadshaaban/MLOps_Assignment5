FROM python:3.10-slim

ARG RUN_ID

RUN pip install mlflow scikit-learn

# Simulate downloading the model using the Run ID
RUN echo "Downloading model for Run ID: ${RUN_ID}"

# In a real scenario, you would do something like:
# RUN python -c "import mlflow; mlflow.artifacts.download_artifacts(run_id='${RUN_ID}', dst_path='/app/model')"

WORKDIR /app
COPY . /app

CMD ["echo", "Model server running"]