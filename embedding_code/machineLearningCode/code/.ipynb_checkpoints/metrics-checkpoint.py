import torch
import torch.nn as nn
from sklearn import metrics as skm

#03/20/25 Varun added

def calc_metrics(output, labels):
    preds = torch.argmax(output, axis=1)
    m = nn.Sigmoid()
    sigmoid_output = m(output)

    metrics_dict = {
        'accuracy_score':skm.accuracy_score(labels, preds),
        'f1_score':skm.f1_score(labels, preds),
        'roc_auc_score':skm.roc_auc_score(labels, output[: ,1])
    }

    return metrics_dict