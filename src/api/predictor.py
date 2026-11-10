import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.api.parts import split_parts
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

    def _probs(self, texts):
        enc = self.tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True,
                             return_tensors='pt', return_token_type_ids=False)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).tolist()

    def _top(self, probs):
        best = max(range(len(probs)), key=lambda i: probs[i])
        return self.labels[best], round(probs[best], 4)

    def predict(self, text):
        cleaned = clean_tweet(text)
        pieces = split_parts(cleaned) or [cleaned]
        many = len(pieces) > 1

        all_probs = self._probs([cleaned] + (pieces if many else []))
        whole = all_probs[0]
        part_probs = all_probs[1:] if many else [whole]

        label, confidence = self._top(whole)
        parts = []
        for piece, p in zip(pieces, part_probs):
            part_label, part_conf = self._top(p)
            parts.append({'text': piece, 'label': part_label, 'confidence': part_conf})

        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {l: round(p, 4) for l, p in zip(self.labels, whole)},
            'cleaned_text': cleaned,
            'parts': parts,
        }

    def predict_batch(self, texts, batch_size=32):
        cleaned = [clean_tweet(t) for t in texts]
        results = [None] * len(texts)
        valid = [i for i, c in enumerate(cleaned) if c]
        valid.sort(key=lambda i: len(cleaned[i]))

        for start in range(0, len(valid), batch_size):
            chunk = valid[start:start + batch_size]
            probs = self._probs([cleaned[i] for i in chunk])
            for i, p in zip(chunk, probs):
                label, confidence = self._top(p)
                results[i] = {
                    'label': label,
                    'confidence': confidence,
                    'probabilities': {l: round(x, 4) for l, x in zip(self.labels, p)},
                    'cleaned_text': cleaned[i],
                    'error': None,
                }

        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    'label': None,
                    'confidence': None,
                    'probabilities': None,
                    'cleaned_text': '',
                    'error': 'Nothing left to analyse after cleaning.',
                }
        return results