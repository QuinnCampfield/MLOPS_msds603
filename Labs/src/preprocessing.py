import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

def load_data(train_name, col_names):
    # Load data
    train_data = pd.read_csv(train_name)
    return train_data

def process_data(df_income, chi2percentile):
    df_income.drop(['CONDMP', 'RENTM', 'Self_emp_income.1'], axis=1, inplace=True)
    df_income.replace(99999996, np.nan, inplace=True)
    nan_replacements = {
        'Highschool': 9,
        'Emp_week_ref': 99,
        'Emp_week_ref': 96,
        'Work_yearly': 99,
        'Total_hour_ref': 99999,
        'Year_immigrant': 9,
        'income_after_tax': 999999,
        'Marital_status': 99,
        'Highest_edu': 9
    }

    for col, val in nan_replacements.items():
        df_income.loc[df_income[col] == val, col] = np.nan

    df_income.dropna(inplace=True)

    # Encode and impute values for target variable
    train_y = df_income["Total_income"]
    
    train_y = train_y.values.reshape((-1,1))

    impy = SimpleImputer(strategy="most_frequent")
    impy.fit(train_y)
    train_y = impy.transform(train_y)

    # Drop target variable
    train_data = df_income.drop(columns=["Total_income"])

    # Create pipeline for imputing and scaling numeric variables
    # one-hot encoding categorical variables, and select features based on chi-squared value
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=chi2percentile)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include = ['int', 'float'])),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude = ['int', 'float'])),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor)]
    )

    # Create new train data using the pipeline
    clf.fit(train_data, train_y)
    train_new = clf.transform(train_data)

    # Transform to dataframe and save as a csv
    train_new = pd.DataFrame(clf.transform(train_data))
    train_new['Total_income'] = train_y
    return train_new, clf

def save_data(train_new, train_name, clf, clf_name):
    train_new.to_csv(train_name)

    # Save pipeline
    with open(clf_name,'wb') as f:
        pickle.dump(clf,f)

if __name__=="__main__":

    params = yaml.safe_load(open("params.yaml"))["features"]
    income_file = params["income_path"]
    chi2percentile = params["chi2percentile"]
    col_names = ['PersonID', 'Weight', 'Province', 'MBMREGP', 'Age_gap',
                 'Gender', 'Marital_status', 'Highschool', 'Highest_edu', 'Work_ref',
                 'Work_yearly', 'Emp_week_ref', 'Total_hour_ref', 'paid_emp_ref',
                 'self_emp_ref', 'Immigrant', 'Year_immigrant', 'income_after_tax',
                 'Cap_gain', 'Childcare_expe', 'Child_benefit', 'CPP_QPP', 'Earning',
                 'Guaranteed_income', 'Investment', 'Old_age_pension', 'Private_pension',
                 'Self_emp_income', 'Pension', 'Self_emp_income.1', 'Total_income',
                 'Emp_insurance', 'Salary_wages', 'compensation', 'Family_mem', 'CFCOMP',
                 'CONDMP', 'RENTM']
    income_data = load_data(income_file, col_names)
    income_new, clf = process_data(income_data, chi2percentile)
    save_data(income_new, 'data/processed_income_data.csv', clf, 'data/pipeline_income.pkl')
