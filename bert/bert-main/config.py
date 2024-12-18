import argparse
import os.path

def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data", "cnews.train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data", "cnews.val.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data", "cnews.test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join("./data", "class.txt"))
    parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese")
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learn_rate", type=float, default=1e-5)
    parser.add_argument("--num_filters", type=int, default=768)
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args
