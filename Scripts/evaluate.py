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
    print(y_pred, y_test)
    acc = torchmetrics.Accuracy()
    precision = torchmetrics.Precision()
    recall = torchmetrics.Recall()
    f1 = torchmetrics.F1()
    acc_val = acc(torch.tensor(y_pred), torch.tensor(y_test))
    precision_val = precision(torch.tensor(y_pred), torch.tensor(y_test), average='weighted')
    recall_val = recall(torch.tensor(y_pred), torch.tensor(y_test), average='weighted')
    f1_val = f1(torch.tensor(y_pred), torch.tensor(y_test), average='weighted')
    print('Accuracy::', acc_val.item())
    print('Precision::', precision_val.item())
    print('Recall::', recall_val.item())
    print('F_score::', f1_val.item())

    test_df['y_pred'] = y_pred
    #pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    test_df.to_csv(f'{args.output_path}test_acc---{acc}.csv', index = False)

    conf_mat = torchmetrics.ConfusionMatrix()
    print(conf_mat(torch.tensor(y_pred), torch.tensor(y_test)))
