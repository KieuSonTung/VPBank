import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import optuna
import lightgbm as lgb

from src.eda.plot_distribution import plot_distribution, compare_dataframes
from src.preprocess import bin_column_by_custom_quantiles, bin_column_by_quantile, convert_edu, convert_marital_status, fill_missing_categorical, label_encode_datasets, remove_spaces
from src.model.lgbm import LGBM


def preprocess_pipeline(df):
    
    # Strip
    df = remove_spaces(df)

    # Binning
    df = bin_column_by_custom_quantiles(df, 'age', quantiles=[0.2, 0.4, 0.6, 0.8])
    df['edu_new'] = df['edu'].apply(convert_edu)
    df['marital_status_new'] = df['marital_status'].apply(convert_marital_status)

    return df



