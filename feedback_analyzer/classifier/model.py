# importam bibliotecile si modulele necesare
import json
import torch # pt operatii pe tensori si deep learning
import torch.nn.functional as F # # modul pentru componente de retele neurale

from transformers import BertTokenizer
from .affective_items_classifier import AffectiveItemsClassifier

# incarcam configuratia din fisierul config.json
with open("config.json") as json_file:
    config = json.load(json_file)

# definim clasa Model pentru clasificare
class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # initializam tokenizer-ul BERT utilizand modelul specificat in configuratie
        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])
        # initializam classifier-ul cu numarul de clase (5)
        classifier = AffectiveItemsClassifier(len(config["CLASS_NAMES"]))
        # incarcam starea pre-antrenata a classifier-ului din fisierul specificat in configuratie
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        # setam classifier-ul in modul de evaluare si il mutam pe device-ul corespunzator
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        # encodam textul de intrare folosind tokenizer-ul BERT
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=config["MAX_SEQUENCE_LEN"], # lungimea maxima a secventei specificata in configuratie
            add_special_tokens=True, # adaugam token-urile speciale necesare pentru BERT
            return_token_type_ids=False, # nu returnam ID-urile de tip token
            padding="max_length", # adaugam PAD pana la lungimea maxima
            return_attention_mask=True, # returnam masca de atentie
            return_tensors="pt",  # returnam rezultatele ca tensori PyTorch
        )
        # mutam input_ids si attention_mask pe device-ul corespunzator
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)
        # facem predictia fara a calcula gradientul (nu este necesar pentru predictie)
        with torch.no_grad():
            # calculam probabilitatile pentru fiecare clasa
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        # determinam clasa cu probabilitatea maxima si increderea in predictie
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        # returnam numele clasei prezise, increderea in predictie si un dictionar cu toate probabilitatile
        return (
            config["CLASS_NAMES"][predicted_class],
            confidence,
            dict(zip(config["CLASS_NAMES"], probabilities)),
        )

# instantiem modelul
model = Model()

# functie pentru a obtine modelul instantiat
def get_model():
    return model
