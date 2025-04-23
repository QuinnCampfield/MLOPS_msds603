from metaflow import FlowSpec, step, Parameter, kubernetes, conda, retry, catch
from metaflow import FlowSpec, step, conda_base, schedule
import pandas as pd
import numpy as np
import mlflow
import os

@conda_base(libraries={'pandas': '1.3.0', 'numpy': '1.21.0', 'scikit-learn': '1.0.2', 'mlflow': '1.20.0', 'protobuf': '3.20.0', 'fsspec': '2023.6.0', 'gcsfs': '2023.6.0'}, python='3.9.16')
class TrainFlow(FlowSpec):

    @conda(libraries={'scikit-learn': '1.0.2', 'pandas': '1.3.5', 'numpy': '1.21.6', 'mlflow': '1.20.0', 'psycopg2-binary': '2.9.9', 'protobuf': '3.20.0', 'fsspec': '2023.6.0', 'gcsfs': '2023.6.0'})
    @step
    def start(self):
        self.df_income = pd.read_csv("gs://lab7-data-bucket/IncomeSurveyDataset.csv")
        self.df_income.drop(['CONDMP', 'RENTM', 'Self_emp_income.1'], axis=1, inplace=True)
        self.df_income.replace(99999996, np.nan, inplace=True)
        nan_replacements = {
            'Highschool': 9,
            'Emp_week_ref': 96,
            'Work_yearly': 99,
            'Total_hour_ref': 99999,
            'Year_immigrant': 9,
            'income_after_tax': 999999,
            'Marital_status': 99,
            'Highest_edu': 9
        }
        self.df_income.loc[self.df_income['Emp_week_ref'] == 99, 'Emp_week_ref'] = np.nan

        for col, val in nan_replacements.items():
            self.df_income.loc[self.df_income[col] == val, col] = np.nan

        self.df_income.dropna(inplace=True)
        ####
        print("Data loaded successfully")
        self.next(self.splitdata)

    
    @conda(libraries={'scikit-learn': '1.0.2'})
    @step
    def splitdata(self):
        from sklearn.model_selection import train_test_split
        y = self.df_income["Total_income"]
        X = self.df_income.drop(columns=["Total_income"])
        X_train_test, self.X_val, y_train_test, self.y_val = train_test_split(X, y, test_size=0.2, random_state=26)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=26)

        print("Data split successfully")
        self.next(self.train_dt, self.train_rf, self.train_xgb)

    
    @kubernetes(cpu=0.5, memory=2048)
    @conda(libraries={'scikit-learn': '1.0.2'})
    @retry(times=3)
    @catch(var='train_error')
    @step
    def train_dt(self):
        from sklearn.tree import DecisionTreeRegressor
        import mlflow

        mlflow.set_tracking_uri('https://lab7-mlflow-453133021138.us-west2.run.app')
        mlflow.set_experiment('income-data-exp')

        with mlflow.start_run(run_name='decision_tree'):
            self.model = DecisionTreeRegressor()
            self.model.fit(self.X_train, self.y_train)
            self.model_name = "DecisionTree"
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'decision_tree',
                'max_depth': self.model.get_params()['max_depth'],
                'min_samples_split': self.model.get_params()['min_samples_split'],
                'min_samples_leaf': self.model.get_params()['min_samples_leaf']
            })
            
            # Log training score
            train_score = self.model.score(self.X_train, self.y_train)
            mlflow.log_metric('train_score', train_score)
            
        self.next(self.choose_model)

    
    @kubernetes(cpu=1, memory=4096)
    @conda(libraries={'scikit-learn': '1.0.2'})
    @retry(times=3)
    @catch(var='train_error')
    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        import mlflow

        mlflow.set_tracking_uri('https://lab7-mlflow-453133021138.us-west2.run.app')
        mlflow.set_experiment('income-data-exp')

        with mlflow.start_run(run_name='random_forest'):
            self.model = RandomForestRegressor(n_estimators=20, max_depth=5)
            self.model.fit(self.X_train, self.y_train)
            self.model_name = "RandomForest"
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'random_forest',
                'n_estimators': 20,
                'max_depth': 5
            })
            
            # Log training score
            train_score = self.model.score(self.X_train, self.y_train)
            mlflow.log_metric('train_score', train_score)
            
        self.next(self.choose_model)

    
    @kubernetes(cpu=1, memory=4096)
    @conda(libraries={'xgboost': '1.4.2'})
    @retry(times=3)
    @catch(var='train_error')
    @step
    def train_xgb(self):
        from xgboost import XGBRegressor
        import mlflow

        mlflow.set_tracking_uri('https://lab7-mlflow-453133021138.us-west2.run.app')
        mlflow.set_experiment('income-data-exp')

        with mlflow.start_run(run_name='xgboost'):
            self.model = XGBRegressor()
            self.model.fit(self.X_train, self.y_train)
            self.model_name = "XGBoost"
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'xgboost',
                'max_depth': self.model.get_params()['max_depth'],
                'learning_rate': self.model.get_params()['learning_rate'],
                'n_estimators': self.model.get_params()['n_estimators']
            })
            
            # Log training score
            train_score = self.model.score(self.X_train, self.y_train)
            mlflow.log_metric('train_score', train_score)
            
        self.next(self.choose_model)

    
    @kubernetes(cpu=0.5, memory=2048)
    @conda(libraries={'mlflow': '1.20.0', 'protobuf': '3.20.0', 'xgboost': '1.4.2'})
    @retry(times=3)
    @catch(var='choose_error')
    @step
    def choose_model(self, inputs):
        mlflow.set_tracking_uri('https://lab7-mlflow-453133021138.us-west2.run.app')
        
        # Get or create the experiment
        experiment_name = 'income-data-exp'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # If the experiment doesn't exist, create it
            try:
                mlflow.create_experiment(experiment_name)
                print(f"Experiment '{experiment_name}' created.")
            except Exception as e:
                print(f"Error creating experiment: {str(e)}")
        else:
            print(f"Experiment '{experiment_name}' already exists.")

        # Set the experiment as the current experiment
        mlflow.set_experiment(experiment_name)

        def score(inp):
            return inp.model, inp.model.score(inp.X_test, inp.y_test)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.best_score = self.results[0][1]

        with mlflow.start_run():
            mlflow.log_metric("score", self.best_score)
            mlflow.sklearn.log_model(
                self.model,
                artifact_path='lab6_train',
                registered_model_name="lab6-model"
            )
            # mlflow.end_run()
        self.next(self.end)

    @conda(libraries={'xgboost': '1.4.2'})
    @step
    def end(self):
        print('Model Scores:')
        for model, score in self.results:
            print(f'{model}: {score:.4f}')
        print(f'Best score {self.best_score:.4f}')


if __name__ == '__main__':
    TrainFlow()
