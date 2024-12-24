import os
from google.cloud import automl_v1beta1
from google.cloud import storage
from google.cloud import bigquery
import pandas as pd
import json
import uuid
from datetime import datetime

class AutoMLHandler:
    def __init__(self, project_id, compute_region):
        self.project_id = project_id
        self.compute_region = compute_region
        self.client = automl_v1beta1.AutoMlClient()
        self.storage_client = storage.Client()
        self.dataset_id = None
        self.model_id = None
        
    def prepare_dataset(self, file_path):
        """Prepare the dataset for AutoML"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return self._prepare_tabular_dataset(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return self._prepare_vision_dataset(file_path)
        elif file_extension == '.txt':
            return self._prepare_text_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _prepare_tabular_dataset(self, file_path):
        """Prepare tabular data for AutoML Tables"""
        df = pd.read_csv(file_path)
        
        # Create a unique bucket name
        bucket_name = f"automl-{self.project_id}-{uuid.uuid4()}"
        bucket = self.storage_client.create_bucket(bucket_name)
        
        # Upload CSV to GCS
        blob = bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        
        # Create dataset
        dataset = {
            "display_name": f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "tables_dataset_metadata": {
                "target_column_spec_id": df.columns[-1]  # Assume last column is target
            }
        }
        
        response = self.client.create_dataset(
            parent=f"projects/{self.project_id}/locations/{self.compute_region}",
            dataset=dataset
        )
        
        self.dataset_id = response.name
        return "tabular", bucket.name, os.path.basename(file_path)

    def _prepare_vision_dataset(self, file_path):
        """Prepare image data for AutoML Vision"""
        bucket_name = f"automl-vision-{self.project_id}-{uuid.uuid4()}"
        bucket = self.storage_client.create_bucket(bucket_name)
        
        blob = bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        
        dataset = {
            "display_name": f"vision_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "image_classification_dataset_metadata": {}
        }
        
        response = self.client.create_dataset(
            parent=f"projects/{self.project_id}/locations/{self.compute_region}",
            dataset=dataset
        )
        
        self.dataset_id = response.name
        return "vision", bucket_name, os.path.basename(file_path)

    def train_model(self, dataset_type):
        """Train AutoML model"""
        model = {
            "display_name": f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "dataset_id": self.dataset_id,
            "train_budget_milli_node_hours": 1000
        }
        
        response = self.client.create_model(
            parent=f"projects/{self.project_id}/locations/{self.compute_region}",
            model=model
        )
        
        self.model_id = response.name
        return self.model_id

    def get_model_evaluation(self):
        """Get model evaluation metrics"""
        evaluations = self.client.list_model_evaluations(parent=self.model_id)
        return [evaluation for evaluation in evaluations]

    def predict(self, input_data):
        """Make predictions using the trained model"""
        prediction_client = automl_v1beta1.PredictionServiceClient()
        
        if isinstance(input_data, pd.DataFrame):
            # For tabular data
            payload = {
                "row": {"values": input_data.values.tolist()[0]}
            }
        else:
            # For image/text data
            payload = {"image": {"image_bytes": input_data}}
        
        request = prediction_client.predict(
            name=self.model_id,
            payload=payload
        )
        
        return request.payload

    def export_model(self, output_path):
        """Export the trained model"""
        model_export = self.client.export_model(name=self.model_id)
        
        # Wait for export to complete
        model_export.result()
        
        # Download the exported model
        bucket_name = f"exported-model-{uuid.uuid4()}"
        bucket = self.storage_client.create_bucket(bucket_name)
        
        model_blob = bucket.blob("model.pkl")
        model_blob.download_to_filename(output_path)
        
        return output_path

    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        evaluations = self.get_model_evaluation()
        
        report = {
            "model_id": self.model_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {},
            "feature_importance": {},
            "model_parameters": {}
        }
        
        for evaluation in evaluations:
            if hasattr(evaluation, "classification_evaluation_metrics"):
                report["metrics"]["accuracy"] = evaluation.classification_evaluation_metrics.au_roc
                report["metrics"]["precision"] = evaluation.classification_evaluation_metrics.precision
                report["metrics"]["recall"] = evaluation.classification_evaluation_metrics.recall
            elif hasattr(evaluation, "regression_evaluation_metrics"):
                report["metrics"]["rmse"] = evaluation.regression_evaluation_metrics.root_mean_squared_error
                report["metrics"]["mae"] = evaluation.regression_evaluation_metrics.mean_absolute_error
                report["metrics"]["r2"] = evaluation.regression_evaluation_metrics.r_squared
                
        return report