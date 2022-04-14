from taxifare.data import clean_df, holdout, get_data_using_pandas, get_data_using_blob
from taxifare.metrics import compute_rmse
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
import joblib
from taxifare.mlflow import MLFlowBase
from taxifare.gcp import save_model_to_gcp

class Trainer(MLFlowBase):
    
    def __init__(self):
        super().__init__("[FR][Marseille/nice][JulienD] RandomForest v1", "https://wagon-mlflow-students.herokuapp.com/")
        
    
    def train(self):
        
        line_count=100
        model_name = "random_forest"
        
        self.mlflow_create_run()
        
        self.mlflow_log_param("line_count", line_count)
        self.mlflow_log_param("model_name", model_name)
        
        df = get_data_using_blob(line_count)
        print("J'ai récupéré les données !")
        df = clean_df(df)
        
        X_train, X_test, y_train, y_test = holdout(df)
        
        model = get_model(model_name)
        
        pipeline = get_pipeline(model)
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        rmse = compute_rmse(y_pred, y_test)
        
        self.mlflow_log_param("rmse", rmse)
        
        joblib.dump(pipeline, "model.joblib")
        print("J'ai créé le model.joblib !")
        
        save_model_to_gcp()
        print("J'ai sauvegardé le modèle sur GCP !")
        
        return pipeline

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()