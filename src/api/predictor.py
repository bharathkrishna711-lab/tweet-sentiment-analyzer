import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing.clean import clean_tweet

DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'models' / 'distilbert_final'
MODEL_PATH = os.getenv('MODEL_PATH', str(DEFAULT_PATH))
MAX_LEN = 64


class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        n = self.model.config.num_labels
        self.labels = [self.model.config.id2label[i] for i in range(n)]

    def predict(self, text):
        cleaned = clean_tweet(text)
        enc = self.tokenizer(cleaned, truncation=True, max_length=MAX_LEN,
                             return_tensors='pt', return_token_type_ids=False)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        best = max(range(len(probs)), key=lambda i: probs[i])
        return {
            'label': self.labels[best],
            'confidence': round(probs[best], 4),
            'probabilities': {l: round(p, 4) for l, p in zip(self.labels, probs)},
            'cleaned_text': cleaned,
        }