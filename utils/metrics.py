from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred_bin),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }

