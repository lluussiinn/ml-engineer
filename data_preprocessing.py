import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

import pandas as pd
from sklearn.impute import SimpleImputer

def check_missing_values(df):
    """
    Checks for missing values in the dataset and returns a DataFrame with counts and percentages
    """
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    return pd.DataFrame({"Missing Values": missing_data,"Percentage (%)": missing_percentage}).sort_values(by="Missing Values", ascending=False)


def impute_missing_values(df, strategy, fill_value=None):
    """
    Imputes missing  values 

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



def calculate_outliers(df):
    """
    Calculate the number of outliers in each numeric feature using the IQR method.

    """
    features = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_stats = []
    total_rows = len(df)
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
        outlier_percentage = (outliers / total_rows) * 100

        outlier_stats.append({
            'Feature': feature,
            'Outliers': outliers,
            'Outlier Percentage (%)': outlier_percentage})

    return pd.DataFrame(outlier_stats)

def log_transform_features(df, features):
    """
    Log transform specified numeric columns to reduce the effect of outliers.

    """
    transformed_df = df.copy()
    for feature in features:
        if (transformed_df[feature] <= 0).any():
            print(f"Warning: Column '{feature}' contains non-positive values. Skipping log transformation.")
            continue
        transformed_df[feature] = np.log1p(transformed_df[feature])
    return transformed_df


def remove_outliers(df, features):
    """
    Remove rows with outliers from the DataFrame using the IQR method.
    """
    cleaned_df = df.copy()
    
    for feature in features:
        Q1 = cleaned_df[feature].quantile(0.25)
        Q3 = cleaned_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove rows with outliers
        cleaned_df = cleaned_df[(cleaned_df[feature] >= lower_bound) & (cleaned_df[feature] <= upper_bound)]
    
    return cleaned_df