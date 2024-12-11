import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



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