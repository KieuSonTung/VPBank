from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

from src.preprocess import label_encode_datasets
from src.utils.save_output import generate_new_file_path
from src.evaluate.evaluate import plot_confusion_matrix, plot_roc


class LGBM():
    def __init__(self, train_df: pd.DataFrame) -> None:
        '''
        Params:
            train_df: train set
        Args:
            cls_report: metrics
            best_model: model with highest F1 Score
            best_params: hyperparameters with highest F1 Score
        '''
        self.train_df = train_df
        self.cls_report = None
        self.best_model = None
        self.best_params = None
    
    def _process_X_y(self):
        income_mapping = {'no': 0, 'yes': 1}
        self.train_df['high_income'] = self.train_df['high_income'].map(income_mapping)

        # self.train_df = label_encode_datasets(self.train_df)

        X = self.train_df.drop('high_income', axis=1)
        y = self.train_df['high_income']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
            for train_index, valid_index in skf.split(self.X_train, self.y_train):
                train_x, valid_x = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
                train_y, valid_y = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]

                train_x = label_encode_datasets(df=train_x, train=True)
                valid_x = label_encode_datasets(df=valid_x, train=False)
                
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
            
            trial.set_user_attr(key="best_booster", value=gbm)
            # Return a single metric (mean accuracy in this case) for optimization
            return mean_f1
        
        def callback(study, trial):
            if study.best_trial.number == trial.number:
                study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])

        # Create the study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[callback]
        )

        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        self.best_trial = study.best_trial

        self.best_model = study.user_attrs["best_booster"]

        print('  Value: {}'.format(self.best_trial.value))

    def plot_feat_imp(self, model, n_features=5):
        importance = model.feature_importance()

        # Create a DataFrame with feature names and their importance
        feature_importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
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
        self.best_params = self.best_trial.params
        self.best_params['objective'] = 'binary'
        self.best_params['metric'] = 'binary_logloss'
        self.best_params['boosting_type'] = 'gbdt'
        self.best_params['verbose'] = -1
        self.best_params['feature_pre_filter'] = False
        self.best_params['n_jobs'] = -1

    def evaluate(self):
        self.save_best_params()

        # Preprocess test set
        self.X_test = label_encode_datasets(self.X_test, train=False)

        # Plot feature importance
        self.plot_feat_imp(self.best_model)

        # Infer
        y_prob = self.best_model.predict(self.X_test)
        y_pred = (y_prob > 0.5).astype(int)

        # Classification report
        self.cls_report = classification_report(y_pred=y_pred, y_true=self.y_test, output_dict=True)

        # Plot confusion matrix
        plot_confusion_matrix(self.y_test, y_pred)

        # Plot ROC
        plot_roc(self.y_test, y_prob)

        # Save the best model
        joblib.dump(self.best_model, generate_new_file_path('../weights/best_lgbm.pkl'))
    
    def run(self, n_splits=2, n_trials=2):
        # Process data
        self._process_X_y()

        # Find best hyperparameters
        self._find_best_params(n_splits=n_splits, n_trials=n_trials)

        # Save best model as pickle file
        self.evaluate()