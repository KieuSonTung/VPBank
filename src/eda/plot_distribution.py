import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_distribution(df, column):
    '''
    Compare distribution of train and test set
    '''
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

    # Check if the column is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[column]):
        print(f'{column} is numeric')
        # Numeric feature: Use line chart
        sns.histplot(df[column], bins=30, kde=True, ax=axs, color='blue', label='train ' + column, alpha=0.5)

        # Add density plots for a smooth distribution
        sns.kdeplot(df[column], ax=axs, color='blue', alpha=0.5)
    else:
        print(f'{column} is categorical')
        # Categorical feature: Use bar chart
        sns.countplot(x=column, data=df, ax=axs, color='blue', alpha=0.5)

    # Set titles and labels
    axs.set_title(f'df {column} Distribution')
    axs.set_xlabel('')
    axs.set_ylabel('Count')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def compare_dataframes(df1, df2):
    """
    This function takes two pandas DataFrames as input and draws plots to compare
    each feature's distribution of the two DataFrames using Seaborn. Bar charts are 
    used for categorical features, and line charts are used for numeric features.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    """
    
    # Ensure that both DataFrames have the same columns
    # if not df1.columns.equals(df2.columns):
    #     raise ValueError("DataFrames must have the same columns")
    
    # Loop over each column in the DataFrames
    for column in df1.columns:
        if column not in ['high_income', 'id']:
            # Create a figure with 2 subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

            # Check if the column is numeric or categorical
            if pd.api.types.is_numeric_dtype(df1[column]):
                print(f'{column} is numeric')
                # Numeric feature: Use line chart
                sns.histplot(df1[column], bins=30, kde=True, ax=axs[0], color='blue', label='train ' + column, alpha=0.5)
                sns.histplot(df2[column], bins=30, kde=True, ax=axs[1], color='red', label='test ' + column, alpha=0.5)

                # Add density plots for a smooth distribution
                sns.kdeplot(df1[column], ax=axs[0], color='blue', alpha=0.5)
                sns.kdeplot(df2[column], ax=axs[1], color='red', alpha=0.5)
            else:
                print(f'{column} is categorical')
                # Categorical feature: Use bar chart
                sns.countplot(x=column, data=df1, ax=axs[0], color='blue', alpha=0.5)
                sns.countplot(x=column, data=df2, ax=axs[1], color='red', alpha=0.5)

            # Set titles and labels
            axs[0].set_title(f'df1 {column} Distribution')
            axs[0].set_xlabel('')
            axs[0].set_ylabel('Count')
            axs[1].set_title(f'df2 {column} Distribution')
            axs[1].set_xlabel(column)
            axs[1].set_ylabel('Count')

            # Adjust layout and show the plots
            plt.tight_layout()
            plt.show()