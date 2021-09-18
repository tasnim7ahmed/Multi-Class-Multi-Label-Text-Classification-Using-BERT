import torch
import pandas as pd
from transformers import BertTokenizer
import numpy as np


from common import get_parser
parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def format_label(label):
    label = (str(label))
    label = label[1:len(label)-2]
    for char in label:
        label = label.replace(".","")
    return list(map(int, label.split(" ")))

class Dataset:
    def __init__(self, text, target):
        self.text = text
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_tokenizer_name)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "token_type_ids":torch.tensor(token_type_ids, dtype = torch.long),
            "target":torch.tensor(format_label(self.target[item]), dtype = torch.long)
        }

def generate_dsfile(path):
    df = pd.read_csv(path).dropna()
    Label_Columns = df.columns.tolist()[3::2]
    print(len(Label_Columns))
    print(df[Label_Columns].sum().sort_values())
    categorized_comment = df[df[Label_Columns].sum(axis=1) > 0]
    uncategorized_comment = df[df[Label_Columns].sum(axis=1) == 0]
    print(f'Categorized - {len(categorized_comment)}, Uncategorized - {len(uncategorized_comment)}')

    comments = []
    labels = []
    
    for i in range(0,len(df.comment.values)):
        current_comment = df.comment.values[i]
        current_label = df[Label_Columns].values[i]
        comments.append(current_comment)
        labels.append(current_label)
    
    sample_count = [0,0,0,0,0,0,0,0]
    for label in labels:
        for i in range(0,len(label)):
            if(label[i]==1):
               sample_count[i]+=1 
    print(sample_count)
    

    ds_data = pd.DataFrame()
    ds_data["comment"] = comments
    ds_data["label"] = labels
    ds_data.to_csv("../Dataset/mod.csv")
    del ds_data


if __name__=="__main__":
    #generate_dsfile("../Dataset/train.csv")
    if(args.augmentation=="True"):
        data_dir = args.aug_train_file
    else:
        data_dir = args.train_file
    df = pd.read_csv(data_dir).dropna()
    print(df.head())
    dataset = Dataset(text=df.comment.values, target=df.label.values)
    print(dataset[0])

    
