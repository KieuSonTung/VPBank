from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(model, trial, x, y):
    preds = model.predict(x)
    pred_labels = (preds > 0.5).astype(int)
    
    accuracy = accuracy_score(y, pred_labels)
    f1 = f1_score(y, pred_labels)
    precision = precision_score(y, pred_labels)
    recall = recall_score(y, pred_labels)
    auc = roc_auc_score(y, preds)

    print(f"Trial {trial.number} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, AUC: {auc}")

    return f1