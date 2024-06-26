import pandas as pd


def bin_column_by_quantile(df1, column, num_bins=4, labels=None):
    """
    Divides a specified column in a DataFrame into bins based on quantiles.

    Parameters:
    df1 (pd.DataFrame): The DataFrame containing the column to be binned.
    column (str): The name of the column to be binned.
    num_bins (int): The number of quantile-based bins to create. Default is 4 (quartiles).
    labels (list): Optional list of labels for the bins. Must be the same length as num_bins.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for the binned values.
    """
    
    df = df1.copy()
    if labels is not None and len(labels) != num_bins:
        raise ValueError("Length of labels must be equal to num_bins")
    
    # Create the bin edges based on quantiles
    bin_edges = pd.qcut(df[column], q=num_bins, labels=labels, retbins=True)[1]
    
    # Bin the column based on the calculated bin edges
    binned_column = pd.cut(df[column], bins=bin_edges, labels=labels, include_lowest=True)
    
    # Create a new column name for the binned values
    binned_column_name = column + '_binned'
    
    # Add the binned column to the DataFrame
    df[binned_column_name] = binned_column
    
    return df

def bin_column_by_custom_quantiles(df, column, quantiles=[0.25, 0.75], labels=None):
    """
    Divides a specified column in a DataFrame into bins based on custom quantiles.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to be binned.
    column (str): The name of the column to be binned.
    quantiles (list): A list of quantiles to use for binning. Default is [0.25, 0.75].
    labels (list): Optional list of labels for the bins. Must be one less than the length of quantiles + 1.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for the binned values.
    """
    
    # Calculate the quantile values
    quantile_values = df[column].quantile(quantiles).tolist()
    
    # Define the bin edges
    bins = [-float('inf')] + quantile_values + [float('inf')]
    
    # If labels are not provided, create default labels
    if labels is None:
        labels = [i+1 for i in range(len(bins) - 1)]
    
    # Ensure the number of labels matches the number of bins
    if len(labels) != len(bins) - 1:
        raise ValueError("The number of labels must be one less than the number of bins")
    
    # Bin the column based on the calculated bin edges
    binned_column = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    
    # Create a new column name for the binned values
    binned_column_name = column + '_binned'
    
    # Add the binned column to the DataFrame
    df[binned_column_name] = binned_column
    
    return df
