import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

def log_metrics_and_artifacts(model, model_name, report, matrix_plot_path, X_sample, y_sample):
    with mlflow.start_run(run_name=model_name):
        # Log more detailed dataset info
        dataset_info = {
            "dataset_name": "Churn_Modelling_dataset",
            "dataset_version": "1.0",
            "dataset_rows": X_sample.shape[0],  
            "dataset_columns": X_sample.shape[1]
        }
        
        for key, value in dataset_info.items():
            mlflow.log_param(key, value)
            
        # Log metrics
        for key, value in report["1"].items():
            mlflow.log_metric(f"{key}", value)

        # Log accuracy separately
        if "accuracy" in report:
            mlflow.log_metric("accuracy", report["accuracy"])

        # Log confusion matrix plot
        mlflow.log_artifact(matrix_plot_path)

        # Infer model signature
        signature = infer_signature(X_sample, model.predict(X_sample))

        # Log the model with signature
        if model_name.lower() == "xgboost":
            mlflow.xgboost.log_model(
                model, 
                f"{model_name}_model",
                signature=signature,
                input_example=None,
                model_format="json",
                pip_requirements=["xgboost==2.0.3"]  
            )
        else:
            mlflow.sklearn.log_model(
                model, 
                f"{model_name}_model",
                signature=signature,
                input_example=None
            )

