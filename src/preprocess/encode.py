import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode_datasets(train_df, test_df):
    """
    Perform label encoding on categorical features in both train and test datasets.
    
    Parameters:
    train_df (pd.DataFrame): Training dataset
    test_df (pd.DataFrame): Testing dataset
    
    Returns:
    pd.DataFrame, pd.DataFrame: Label encoded train and test datasets
    """
    # Initialize the label encoder
    label_encoders = {}
    
    # Concatenate train and test data to ensure consistency in encoding
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    # Identify categorical columns
    categorical_cols = combined_df.select_dtypes(include=['object']).columns
    
    # Label encode each categorical column
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        combined_df[col] = label_encoders[col].fit_transform(combined_df[col])
    
    # Split the combined dataframe back into train and test sets
    train_encoded = combined_df.iloc[:len(train_df), :].reset_index(drop=True)
    test_encoded = combined_df.iloc[len(train_df):, :].reset_index(drop=True)
    
    return train_encoded, test_encoded