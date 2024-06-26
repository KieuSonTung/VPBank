def remove_spaces(df):
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = df[col].str.strip()
    
    return df