import mlflow
from sklearn.linear_model import LogisticRegression

# Déclare un "projet" MLflow
mlflow.set_experiment("tag_suggester")

# Démarre une expérimentation
with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("f1_score", 0.81)
    mlflow.sklearn.log_model(LogisticRegression(), name="model")
