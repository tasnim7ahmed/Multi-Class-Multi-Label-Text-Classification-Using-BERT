from visualization import save_acc_curves, save_loss_curves

import pandas as pd;
import numpy as np;
import torch
from transformers import BertModel, AdamW, get_scheduler
from collections import defaultdict
from sklearn.metrics import f1_score
import warnings

import engine
from model import BertSAUC
from dataset import Dataset
from utils import train_validate_test_split, set_device
from common import get_parser
from evaluate import test_evaluate

parser = get_parser()
args = parser.parse_args()
warnings.filterwarnings("ignore")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def generate_datasets():
    train_path = ""
    if(args.augmentation=="True"):
        train_path = args.aug_train_file
    else:
        train_path = args.train_file
    val_path = args.val_file
    test_path = args.test_file

    df = pd.read_csv(train_path).dropna()
    train_dataset = Dataset(text=df.comment.values, target=df.label.values)
    len_train = len(df)
    del df
    df = pd.read_csv(val_path).dropna()
    val_dataset = Dataset(text=df.comment.values, target=df.label.values)
    del df
    df = pd.read_csv(test_path).dropna()
    test_df = df
    test_dataset = Dataset(text=df.comment.values, target=df.label.values)
    del df

    return train_dataset, val_dataset, test_dataset, len_train, test_df

def run():
    train_dataset, valid_dataset, test_dataset, len_train, test_df = generate_datasets()
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.train_batch_size,
        shuffle = True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = args.valid_batch_size,
        shuffle = True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = args.test_batch_size,
        shuffle = False
    )

    device = set_device()
    model = BertSAUC()
    model = model.to(device)

    num_train_steps = int(len_train / args.train_batch_size * args.epochs)
    
    optimizer = AdamW(
        params = model.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        eps = args.adamw_epsilon
    )

    scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = num_train_steps
    )

    print("---Starting Training---")

    history = defaultdict(list)
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-'*10)

        train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(f'Epoch {epoch + 1} --- Training loss: {train_loss} Training accuracy: {train_acc}')
        val_acc, val_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f'Epoch {epoch + 1} --- Validation loss: {val_loss} Validation accuracy: {val_acc}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc>best_acc:
            torch.save(model.state_dict(), f"{args.model_path}{args.pretrained_model_name}---val_acc---{val_acc}.bin")

    save_acc_curves(history)
    save_loss_curves(history)
    print("##################################### Testing ############################################")
    test_evaluate(test_df, test_data_loader, model, device)    
    del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("##################################### Task End ############################################")

if __name__=="__main__":
    run()