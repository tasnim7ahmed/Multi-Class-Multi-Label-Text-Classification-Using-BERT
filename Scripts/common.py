import argparse

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_length", default=128, type=int,  help='Maximum number of words in a sample')
    parser.add_argument("--train_batch_size", default=16, type=int,  help='Training batch size')
    parser.add_argument("--valid_batch_size", default=32, type=int,  help='Validation batch size')
    parser.add_argument("--test_batch_size", default=32, type=int,  help='Test batch size')
    parser.add_argument("--epochs", default=1, type=int,  help='Number of training epochs')
    parser.add_argument("-lr","--learning_rate", default=3e-5, type=float,  help='The learning rate to use')
    parser.add_argument("-wd","--weight_decay", default=1e-4, type=float,  help=' Decoupled weight decay to apply')
    parser.add_argument("--adamw_epsilon", default=1e-8, type=float,  help='Adam’s epsilon for numerical stability')
    parser.add_argument("--warmup_steps", default=0, type=int,  help='The number of steps for the warmup phase.')
    parser.add_argument("--classes", default=8, type=int, help='Number of output classes')
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="gpu", help="Training device - cpu/gpu")

    parser.add_argument("--num_aug", default=10, required=False, type=int, help="number of augmented sentences per original sentence")
    parser.add_argument("--alpha", default=0.15, required=False, type=float, help="percent of words in each sentence to be changed")

    parser.add_argument("--pretrained_tokenizer_name", default="bert-base-uncased", type=str, help='Name of the pretrained tokenizer')
    parser.add_argument("--pretrained_model_name", default="bert-base-uncased", type=str, help='Name of the pretrained model')
    parser.add_argument("--bert_hidden", default=768, type=int, help='Number of layer for Bert')

    parser.add_argument("--augmentation", default="True", type=str, help="Augmentation - True/False")
    parser.add_argument("--train_file", default="../Dataset/train.csv", type=str, help='Path to training dataset file')
    parser.add_argument("--val_file", default="../Dataset/val.csv", type=str, help='Path to validation dataset file')
    parser.add_argument("--test_file", default="../Dataset/test.csv", type=str, help='Path to testing dataset file')
    parser.add_argument("--aug_train_file", default="../Dataset/aug_train.csv", type=str, help='Path to augmented dataset file')
    parser.add_argument("--aug_healthy_dataset_file", default="../Dataset/aug_healthy.csv", type=str, help='Path to augmented healthy dataset file')
    parser.add_argument("--model_path", default="../Models/", type=str, help='Save best model')
    parser.add_argument("--output_path", default="../Output/", type=str, help='Get predicted labels for test data')
    parser.add_argument("--figure_path", default="../Figures/", type=str, help='Directory for accuracy and loss plots')

    return parser