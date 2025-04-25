from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
import mlflow
import os
import joblib


class TrainFlow(FlowSpec):
    @step
    def start(self):
        self.df_income = pd.read_csv("src/IncomeSurveyDataset.csv")
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
        print("Data loaded successfully")
        self.next(self.splitdata)

    @step
    def splitdata(self):
        from sklearn.model_selection import train_test_split
        y = self.df_income["Total_income"]
        X = self.df_income.drop(columns=["Total_income"])
        X_train_test, self.X_val, y_train_test, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=26
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_train_test, y_train_test, test_size=0.2, random_state=26
        )

        print("Data split successfully")
        self.next(self.train_dt, self.train_rf)#, self.train_xgb)

    @step
    def train_dt(self):
        from sklearn.tree import DecisionTreeRegressor

        self.model = DecisionTreeRegressor()
        self.model.fit(self.X_train, self.y_train)
        self.model_name = "DecisionTree"
        self.next(self.choose_model)

    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=20, max_depth=5)
        self.model.fit(self.X_train, self.y_train)
        self.model_name = "RandomForest"
        self.next(self.choose_model)

    # @step
    # def train_xgb(self):
    #     from xgboost import XGBRegressor

    #     self.model = XGBRegressor()
    #     self.model.fit(self.X_train, self.y_train)
    #     self.model_name = "XGBoost"
    #     self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment('income-prediction')

        def score(inp):
            return inp.model, inp.model.score(inp.X_test, inp.y_test)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.best_score = self.results[0][1]

        with mlflow.start_run():
            mlflow.log_metric("score", self.best_score)
            mlflow.sklearn.log_model(
                self.model,
                artifact_path='model',
                registered_model_name="income-prediction-model"
            )
            # Save the model locally for the FastAPI app
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, "models/income_model.joblib")
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print('Model Scores:')
        for model, score in self.results:
            print(f'{model}: {score:.4f}')
        print(f'Best score {self.best_score:.4f}')


if __name__ == '__main__':
    TrainFlow()
