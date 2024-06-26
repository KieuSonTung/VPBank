from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import optuna
import lightgbm as lgb


def finetune(X_encoded, y):
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
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        acc_train, f1_train, prec_train, recall_train, auc_train = [], [], [], [], []
        
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        aucs = []
        
        # Perform Stratified K-Fold cross-validation
        for train_index, valid_index in skf.split(X_encoded, y):
            train_x, valid_x = X_encoded.iloc[train_index], X_encoded.iloc[valid_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[valid_index]
            
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
    study.optimize(objective, n_trials=100)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')

    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))