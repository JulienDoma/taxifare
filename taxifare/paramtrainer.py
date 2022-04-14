from taxifare.data import get_data, clean_df, holdout
from taxifare.metrics import compute_rmse
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
import joblib
from taxifare.mlflow import MLFlowBase
from sklearn.model_selection import GridSearchCV

class ParamTrainer(MLFlowBase):
    
    def __init__(self):
        super().__init__("[FR][Marseille/nice][JulienD] RandomForest v1", "https://mlflow.lewagon.co/")
        
    
    def train(self, params):
        
        
        models = {}
        
        for model_name, model_params in params.items():
            
            line_count = model_params["line_count"]
            hyper_params = model_params["hyper_params"]
            
            self.mlflow_create_run()
            
            self.mlflow_log_param("line_count", line_count)
            self.mlflow_log_param("model_name", model_name)
            for key, value in hyper_params.items():
                self.mlflow_log_param(key, value)
            
            df = get_data(line_count)
            df = clean_df(df)
            
            X_train, X_test, y_train, y_test = holdout(df)
            
            model = get_model(model_name)
            
            pipeline = get_pipeline(model)
            
            grid_search = GridSearchCV(
                pipeline, 
                param_grid=hyper_params,
                cv=5
            )
            
            grid_search.fit(X_train, y_train)
            
            score = grid_search.score(X_test, y_test)
            
            self.mlflow_log_param("score", score)
            
            joblib.dump(pipeline, f"{model_name}.joblib")
            
            models[model_name] = grid_search
        
        return models
    
        