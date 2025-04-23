from metaflow import FlowSpec, step, Flow, kubernetes, conda, retry, catch, conda_base
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

@conda_base(libraries={
    'pandas': '1.3.0',
    'numpy': '1.21.0',
    'scikit-learn': '1.0.2',
    'mlflow': '1.20.0',
    'protobuf': '3.20.0',
    'xgboost': '1.4.2'
}, python='3.9.16')
class ScoreFlow(FlowSpec):
    
    @conda(libraries={'mlflow': '1.20.0'})
    @step
    def start(self):
        run = Flow('TrainFlow').latest_run
        self.train_run_id = run.pathspec
        self.model = run['end'].task.data.model
        
        # Get the test data from the splitdata step
        self.X_test = run['splitdata'].task.data.X_test
        self.y_test = run['splitdata'].task.data.y_test
        
        print("Loaded model and test data from training run")
        self.next(self.evaluate)

    
    @kubernetes(cpu=1, memory=4096)
    @conda(libraries={'scikit-learn': '1.0.2', 'mlflow': '1.20.0'})
    @retry(times=3)
    @catch(var='eval_error')
    @step
    def evaluate(self):
        self.predictions = self.model.predict(self.X_test)
        
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.r2 = r2_score(self.y_test, self.predictions)
        
        print(f"Evaluation metrics: MSE={self.mse:.2f}, R2={self.r2:.2f}")
        
        mlflow.set_tracking_uri('https://lab7-mlflow-453133021138.us-west2.run.app')
        mlflow.set_experiment('income-data-exp')
        
        with mlflow.start_run(run_name="model_evaluation"):
            mlflow.log_metric("mse", self.mse)
            mlflow.log_metric("r2", self.r2)
            mlflow.end_run()
            
        self.next(self.end)

    @step
    def end(self):
        print("Scoring completed successfully")
        print(f"Final metrics: MSE={self.mse:.2f}, R2={self.r2:.2f}")


if __name__ == '__main__':
    ScoreFlow()
