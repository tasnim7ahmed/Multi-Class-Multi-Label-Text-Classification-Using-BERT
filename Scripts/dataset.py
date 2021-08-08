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
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

if __name__=="__main__":
    df = pd.read_csv(args.dataset_file).dropna()
    Label_Columns = df.columns.tolist()[3::2]
    print(len(Label_Columns))
    print(df[Label_Columns].sum().sort_values())
    categorized_comment = df[df[Label_Columns].sum(axis=1) > 0]
    uncategorized_comment = df[df[Label_Columns].sum(axis=1) == 0]
    print(f'Categorized - {len(categorized_comment)}, Uncategorized - {len(uncategorized_comment)}')
    sample_row = df.iloc[0]
    print(sample_row.comment)
    print(sample_row[Label_Columns].to_dict())
    dataset = Dataset(text=df.comment.values, target=df[Label_Columns].values)
    print(df.iloc[1]['comment'])
    print(dataset[1])

    
