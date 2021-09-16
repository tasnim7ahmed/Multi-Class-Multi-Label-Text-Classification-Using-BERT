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


if __name__=="__main__":
    if(args.augmentation=="True"):
        data_dir = args.aug_dataset_file
    else:
        data_dir = args.dataset_file
    df = pd.read_csv(data_dir).dropna()
    print(df.head())
    dataset = Dataset(text=df.comment.values, target=df.label.values)
    print(dataset[0])

    
