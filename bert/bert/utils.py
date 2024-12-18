import os
from config import parsers
# The transformer library is a library that integrates a variety of pre-trained models together, and once imported, you can selectively use the models you want to use, in this case the BERT model used.
# So the bert model, and bert's disambiguator, are imported, and here is the use of bert, not bert's own source code.
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import re

class_dict={"体育":0,
            "娱乐":1,
            "财经":2,
            "房产":3,
            "家居":4,
            "教育":5,
            "科技":6,
            "时尚":7,
            "时政":8,
            "游戏":9,
            }
def read_data(file):
    # Read file
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # Get the maximum length of all text, all tags, sentences
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            # text, label = data.split("\t")
            label, text = data.split("\t")
            label = class_dict[label]
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)
    # Returns different content based on different datasets
    if os.path.split(file)[1] == "cnews.train.txt":
        max_len = max(max_length)
        return texts, labels, max_len
    return texts, labels,

class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)

    def __getitem__(self, index):
        # Take a piece of data and truncate the length
        text = self.all_text[index][:self.max_len]
        label = self.all_label[index]

        # participle
        text_id = self.tokenizer.tokenize(text)
        #  Add the start flag
        text_id = ["[CLS]"] + text_id

        #  encodings
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        # Mask -
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # After encoding - “Consistent length
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))
        # str -》 int
        label = int(label)
        #Converted to tensor
        token_ids = torch.tensor(token_ids)
        mask = torch.tensor(mask)
        label = torch.tensor(label)

        return (token_ids, mask), label

    def __len__(self):
        # Get the length of the text
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label, max_len = read_data("./data/cnews.val.txt")
    print(train_text[0], train_label[0])
    trainDataset = MyDataset(train_text, train_label, max_len)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    for batch_text, batch_label in trainDataloader:
        print(batch_text, batch_label)
