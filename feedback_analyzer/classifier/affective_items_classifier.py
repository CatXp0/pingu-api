# importam bibliotecile si modulele necesare
import json

from torch import nn
from transformers import BertModel # model bert pre-antrenat de la hugging face

with open("config.json") as json_file:
    config = json.load(json_file)


#  definim o clasa pentru clasificator
class AffectiveItemsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(AffectiveItemsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"]) # initializam modelul bert
        self.drop = nn.Dropout(p=0.3) # layer de dropout pt regularizare
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) # layer output pt clasificare

    def forward(self, input_ids, attention_mask):
        # trecem inputul prin model
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        # aplicam dropout
        output = self.drop(pooled_output)
        # Output layer pt predictia de clase
        return self.out(output)
