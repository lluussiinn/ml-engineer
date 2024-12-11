import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

import pandas as pd
from sklearn.impute import SimpleImputer

def check_missing_values(df):
    """
    Checks for missing values in the dataset and returns a DataFrame with counts and percentages.
    """
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    return pd.DataFrame({
        "Missing Values": missing_data,
        "Percentage (%)": missing_percentage
    }).sort_values(by="Missing Values", ascending=False)

def impute_missing_values(df, strategy, fill_value=None):
    """
    Imputes missing (NaN) values in the dataset for columns that have missing values.

    Parameters:
        df (pd.DataFrame): The dataset with missing values.
        strategy (str): The imputation strategy. Options:
            - "mean": Replace NaN with the mean of the column (numerical only).
            - "median": Replace NaN with the median of the column (numerical only).
            - "most_frequent": Replace NaN with the mode (most frequent value).
            - "constant": Replace NaN with a specified constant value.
        fill_value: The constant value to use if strategy="constant".

    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed.
    """
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()]
    
    # Initialize the imputer
    if strategy == "constant" and fill_value is not None:
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)

    # Impute only columns with missing values
    df[missing_columns] = imputer.fit_transform(df[missing_columns])

    return df


def handle_duplicates(data):
    """
    Removes duplicate rows from the dataset.
    """
    return data.drop_duplicates()

def set_index(df, column = 'pickup_date'):
    # Convert 'pickup_date' to datetime format
    #df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df.set_index(column, inplace=True)


    return df