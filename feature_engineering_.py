import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np



def scale_features(data, method='standard'):
    """
    Scales numerical features using the specified scaling method.
    
    Parameters:
        data (pd.DataFrame): Data containing numerical features.
        method (str): Scaling method ('standard', 'minmax').
        
    Returns:
        pd.DataFrame: Scaled data.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)



def encode_categorical_features(data, columns, method='onehot'):
    """
    Encodes categorical features.
    
    Parameters:
        data (pd.DataFrame): Data containing categorical features.
        columns (list): List of categorical columns to encode.
        method (str): Encoding method ('onehot', 'label').
        
    Returns:
        pd.DataFrame: Data with encoded categorical features.
    """
    if method == 'onehot':
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded = encoder.fit_transform(data[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
        return pd.concat([data.drop(columns, axis=1), encoded_df], axis=1)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        return data
    else:
        raise ValueError("Unsupported method. Choose 'onehot' or 'label'.")



def select_k_best_features(data, target, k=10):
    """
    Selects the top K features based on the ANOVA F-statistic.
    
    Parameters:
        data (pd.DataFrame): Data containing features.
        target (pd.Series): Target variable.
        k (int): Number of top features to select.
        
    Returns:
        pd.DataFrame: Data with selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_data = selector.fit_transform(data, target)
    selected_columns = data.columns[selector.get_support()]
    return pd.DataFrame(selected_data, columns=selected_columns)


def create_lag_features(df, target_column, lag=1):
    """
    Create lag features for a given target column
    """
    df[f'{target_column}_lag{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)  # Drop rows with NaN (due to lagging)
    return df

def add_features(df):

    # Extract features from pickup_date (assumed to be a datetime column)
    df['day_of_week'] = df['pickup_date'].dt.dayofweek
    df['month'] = df['pickup_date'].dt.month

    return df

def setting_index(df):
    # Convert 'pickup_date' to datetime format
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df.set_index('pickup_date', inplace=True)

    return df

# Target Encoding (Mean Encoding) Function
def target_encode(df, categorical_columns, target_column, alpha=5):
    global_mean = df[target_column].mean()
    for col in categorical_columns:
        # Group by the categorical feature and calculate the mean of the target
        category_means = df.groupby(col)[target_column].mean()

        # Calculate the count of each category in the column
        category_counts = df[col].value_counts()

        # Apply smoothing (this prevents overfitting for categories with few observations)
        smoothed_means = category_means * category_counts / (category_counts + alpha) + global_mean * alpha / (category_counts + alpha)
        # Replace categorical values with the smoothed target encoding
        df[col] = df[col].map(smoothed_means).fillna(global_mean)  # Use the global mean for missing values

    return df

def log_transform_features(df, features):
    """
    Log transform specified numeric columns to reduce the effect of outliers.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of numeric feature names to transform.
    
    Returns:
        pd.DataFrame: DataFrame with log-transformed features.
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
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of numeric feature names to check for outliers.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing outliers removed.
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