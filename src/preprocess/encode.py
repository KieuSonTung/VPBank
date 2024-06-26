import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def label_encode_datasets(train_df):
    """
    Perform label encoding on categorical features in both train and test datasets.
    
    Parameters:
    train_df (pd.DataFrame): Training dataset
    
    Returns:
    pd.DataFrame, pd.DataFrame: Label encoded train and test datasets
    """
    # Initialize the label encoder
    label_encoders = {}
    
    # Create a copy of the data to avoid modifying the original dataframes
    train_data_encoded = train_df.copy()

    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    
    # Label encode each categorical column
    for column in categorical_cols:
        le = LabelEncoder()
        
         # Fit the label encoder on the training data
        train_data_encoded[column] = le.fit_transform(train_df[column])

        # Store the label encoder
        label_encoders[column] = le
    
    if not os.path.exists('../weights/label_encoders'):
        os.mkdir('../weights/label_encoders')
            
    joblib.dump(label_encoders, f'../weights/label_encoders/label_encoders.pkl')

    return train_data_encoded