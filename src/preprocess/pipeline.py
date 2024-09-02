import pandas as pd
from src.preprocess import bin_column_by_custom_quantiles, bin_column_by_quantile, convert_edu, convert_marital_status, fill_missing_categorical, label_encode_datasets, remove_spaces
from src.model.lgbm import LGBM


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    
    # Strip
    df = remove_spaces(df)

    # Binning
    # df = bin_column_by_custom_quantiles(df, 'age', quantiles=[0.2, 0.4, 0.6, 0.8])
    df['edu_new'] = df['edu'].apply(convert_edu)
    df['marital_status_new'] = df['marital_status'].apply(convert_marital_status)

    return df



