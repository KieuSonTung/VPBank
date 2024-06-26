from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

from src.preprocess import label_encode_datasets
from src.utils.save_output import generate_new_file_path
from src.evaluate.evaluate import plot_confusion_matrix, plot_roc


class LGBM():
    def __init__(self, train_df) -> None:
        self.train_df = train_df
    
    def _process_X_y(self):
        income_mapping = {'no': 0, 'yes': 1}
        self.train_df['high_income'] = self.train_df['high_income'].map(income_mapping)

        self.train_df = label_encode_datasets(self.train_df)

        self.X = self.train_df.drop('high_income', axis=1)
        self.y = self.train_df['high_income']

    def _find_best_params(self, n_splits, n_trials):
        def objective(trial):
            
            # Define the hyperparameters to be tuned
            param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'feature_pre_filter': False,
                'n_jobs': -1,
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 50, 1000, step=20),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0, step=0.1),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0, step=0.1),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
                "max_bin": trial.suggest_int("max_bin", 200, 300),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.2, 0.95, step=0.1
                ),
            }

            # Initialize StratifiedKFold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            acc_train, f1_train, prec_train, recall_train, auc_train = [], [], [], [], []
            
            accuracies = []
            f1_scores = []
            precisions = []
            recalls = []
            aucs = []
            
            # Perform Stratified K-Fold cross-validation
            for train_index, valid_index in skf.split(self.X, self.y):
                train_x, valid_x = self.X.iloc[train_index], self.X.iloc[valid_index]
                train_y, valid_y = self.y.iloc[train_index], self.y.iloc[valid_index]
                
                # Train the model
                dtrain = lgb.Dataset(train_x, label=train_y)
                dvalid = lgb.Dataset(valid_x, label=valid_y)
                gbm = lgb.train(param, dtrain, valid_sets=[dvalid])

                # Predict and evaluate on trainset
                preds = gbm.predict(train_x)
                pred_labels = (preds > 0.5).astype(int)
                
                acc_train.append(accuracy_score(train_y, pred_labels))
                f1_train.append(f1_score(train_y, pred_labels))
                prec_train.append(precision_score(train_y, pred_labels))
                recall_train.append(recall_score(train_y, pred_labels))
                auc_train.append(roc_auc_score(train_y, preds))
                
                # Predict and evaluate on validset
                preds = gbm.predict(valid_x)
                pred_labels = (preds > 0.5).astype(int)
                
                accuracies.append(accuracy_score(valid_y, pred_labels))
                f1_scores.append(f1_score(valid_y, pred_labels))
                precisions.append(precision_score(valid_y, pred_labels))
                recalls.append(recall_score(valid_y, pred_labels))
                aucs.append(roc_auc_score(valid_y, preds))
            
            # Calculate mean of the metrics
            mean_accuracy = np.mean(acc_train)
            mean_f1 = np.mean(f1_train)
            mean_precision = np.mean(prec_train)
            mean_recall = np.mean(recall_train)
            mean_auc = np.mean(auc_train)

            print(f"TRAIN: Trial {trial.number} - Accuracy: {mean_accuracy}, F1: {mean_f1}, Precision: {mean_precision}, Recall: {mean_recall}, AUC: {mean_auc}")
            
            # Calculate mean of the metrics
            mean_accuracy = np.mean(accuracies)
            mean_f1 = np.mean(f1_scores)
            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)
            mean_auc = np.mean(aucs)
            
            # Print mean metrics for this trial
            print(f"VALID: Trial {trial.number} - Accuracy: {mean_accuracy}, F1: {mean_f1}, Precision: {mean_precision}, Recall: {mean_recall}, AUC: {mean_auc}")
            
            # Return a single metric (mean accuracy in this case) for optimization
            return mean_f1

        # Create the study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        self.best_trial = study.best_trial

        print('  Value: {}'.format(self.best_trial.value))
        print('  Params: ')

        for key, value in self.best_trial.params.items():
            print('    {}: {}'.format(key, value))

    def plot_feat_imp(self, model, n_features=5):
        importance = model.feature_importance()

        # Create a DataFrame with feature names and their importance
        feature_importance_df = pd.DataFrame({
            'feature': self.X.columns,
            'importance': importance
        })
        
        # Get the top 5 most important features
        top_features_df = feature_importance_df.nlargest(n_features, 'importance')
        
        # Plot the top 5 features
        plt.figure(figsize=(10, 6))
        plt.barh(top_features_df['feature'], top_features_df['importance'], color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Top 5 Most Important Features')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
        plt.show()
    
    def save_best_params(self):
        best_params = self.best_trial.params
        best_params['objective'] = 'binary'
        best_params['metric'] = 'binary_logloss'
        best_params['boosting_type'] = 'gbdt'
        best_params['verbose'] = -1
        best_params['feature_pre_filter'] = False
        best_params['n_jobs'] = -1

        return best_params

    def save_best_model(self):
        best_params = self.save_best_params()
        train_x, val_x, train_y, val_y = train_test_split(self.X, self.y)

        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y)

        best_gbm = lgb.train(best_params, dtrain, num_boost_round=100)

        # Plot feature importance
        self.plot_feat_imp(best_gbm)

        # Infer
        y_prob = best_gbm.predict(val_x)
        y_pred = (y_prob > 0.5).astype(int)

        # Plot confusion matrix
        plot_confusion_matrix(val_y, y_pred)

        # Plot ROC
        plot_roc(val_y, y_prob)

        # Save the best model
        joblib.dump(best_gbm, generate_new_file_path('../weights/best_lgbm.pkl'))
    
    def finetune(self, n_splits=5, n_trials=10):
        # Process data
        self._process_X_y()

        # Find best hyperparameters
        self._find_best_params(n_splits=n_splits, n_trials=n_trials)

        # Save best model as pickle file
        self.save_best_model()