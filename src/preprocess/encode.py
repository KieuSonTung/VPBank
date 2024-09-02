import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

def label_encode_datasets(df: pd.DataFrame = None, train=True):
    """
    Perform label encoding on categorical features in both train and test datasets.
    
    Parameters:
    df (pd.DataFrame): dataset
    train (bool): Whether to fit the label encoders on the train dataset or use existing encoders for the test dataset.
    
    Returns:
    pd.DataFrame: Label encoded train or test dataset
    """
    # Initialize the label encoders dictionary
    label_encoders = {}
    
    if train:

        print('ENCODING TRAIN SET...')
        # Create a copy of the data to avoid modifying the original dataframe
        train_data_encoded = df.copy()

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Label encode each categorical column
        for column in categorical_cols:
            le = LabelEncoder()
            train_data_encoded[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        
        # Ensure the directory exists for storing label encoders
        if not os.path.exists('../weights/label_encoders'):
            os.makedirs('../weights/label_encoders')
        
        # Save the label encoders to a file
        joblib.dump(label_encoders, '../weights/label_encoders/label_encoders.pkl')

        return train_data_encoded
    
    else:

        print('ENCODING TEST SET...')
        # Load the label encoders from the file
        if not os.path.exists('../weights/label_encoders/label_encoders.pkl'):
            raise FileNotFoundError("label_encoders.pkl not found. Ensure the label encoders are trained and saved.")
        
        label_encoders = joblib.load('../weights/label_encoders/label_encoders.pkl')
        
        # Create a copy of the test data
        test_data_encoded = df.copy()

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Encode the test data using the existing label encoders
        for column in categorical_cols:
            if column in label_encoders:
                le = label_encoders[column]
                test_data_encoded[column] = le.transform(df[column])
            else:
                raise ValueError(f"Column {column} was not found in the trained label encoders.")
        
        return test_data_encoded
