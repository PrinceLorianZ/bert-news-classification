import torch.nn as nn
from transformers import BertModel
from config import parsers
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.args = parsers()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Load bert Chinese pre-training model
        self.bert = BertModel.from_pretrained(self.args.bert_pred)
        # Let the bert model be fine-tuned (parameters change during training)
        for param in self.bert.parameters():
            param.requires_grad = True
        # full connectivity layer
        self.linear = nn.Linear(self.args.num_filters, self.args.class_num)

    def forward(self, x):
        input_ids, attention_mask = x[0].to(self.device), x[1].to(self.device)
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)  # Controls whether the results of all encoder layers are output.
        # shape (batch_size, hidden_size)  pooler_output -->  hidden_out[0]
        pred = self.linear(hidden_out.pooler_output)
        # Return to Prediction Results
        return pred


