from operator import index
import numpy as np
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score

import dataset
from engine import test_eval_fn
from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def test_evaluate(test_df, test_data_loader, model, device):
    y_pred, y_test = test_eval_fn(test_data_loader, model, device)
    acc = torchmetrics.Accuracy(y_test, y_pred)
    precision = torchmetrics.Precision(y_test, y_pred, average='weighted')
    recall = torchmetrics.Recall(y_test, y_pred, average='weighted')
    f1 = torchmetrics.F1(y_test, y_pred, average='weighted')
    
    print('Accuracy::', acc)
    print('Precision::', precision)
    print('Recall::', recall)
    print('F_score::', f1)
    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    pred_test.to_csv(f'{args.output_path}test_acc---{acc}.csv', index = False)

    conf_mat = torchmetrics.ConfusionMatrix(y_test,y_pred)
    print(conf_mat)
