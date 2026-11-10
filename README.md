# Tweet sentiment analyser

An end-to-end sentiment analysis project: a fine-tuned DistilBERT model, a FastAPI service on Google Cloud Run, and a React page that colours each part of a tweet as negative, neutral or positive.

**Live demo:** https://tweet-sentiment-analyzer-five.vercel.app
**API docs:** https://tweet-sentiment-api-945063549145.asia-south1.run.app/docs
**Model:** https://huggingface.co/Bharath-krish-11/tweet-sentiment-distilbert

The first request after a quiet period can take up to 30 seconds, because the service scales to zero and has to load the model again.

## What it does

You paste a tweet about an airline. The page shows:

- the tweet split into parts, each coloured by its own sentiment
- an overall verdict for the whole tweet
- the probability of negative, neutral and positive

## Results

Data: the Kaggle Twitter US Airline Sentiment dataset (3 classes, about 64% negative, 20% neutral, 15% positive). After cleaning and de-duplication there are 14,017 tweets, split 70/15/15 (9,811 train, 2,103 validation, 2,103 test), stratified, with a fixed seed.

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |
|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.743 | 0.75 | 0.80 |
| DistilBERT, fine-tuned | 0.788 | 0.802 | about 0.84 |

DistilBERT per-class F1 on the test set: negative 0.91, neutral 0.70, positive 0.80. Macro-F1 is the main metric because the classes are imbalanced. All choices (settings, best epoch) were made on the validation set, and the test set was scored once at the end.

## How it works

1. **Cleaning** (`src/preprocessing/clean.py`): unescape HTML, remove links and @mentions, drop the `#` sign, turn emoticons and emoji into words, normalise spaces. The same function is used in training and in the API, so the model sees the same kind of text in both.
2. **De-duplication and split:** duplicates are found on the lower-cased cleaned text. If duplicates disagree on the label, the majority wins, and exact ties are dropped. Anything learned from data is fitted on the training set only.
3. **Baseline:** TF-IDF (1 and 2-grams) with logistic regression, class-balanced, with C chosen on the validation set.
4. **DistilBERT:** fine-tuned for 3 epochs (batch size 32, learning rate 2e-5, weight decay 0.01) with a class-weighted cross-entropy loss. The best epoch was chosen by validation macro-F1.
5. **API:** FastAPI loads the model once at start-up and exposes `/predict` and `/health`. It runs on CPU in a Docker container on Cloud Run (2 GiB memory, at most 2 instances).
6. **Page:** React (Vite) on Vercel, calling the API.

**Part-level colouring.** The cleaned tweet is cut with simple rules: at `. ! ?`, and before "but", "however", "although" and "though". Parts with a single word are joined to the previous part. Each part goes through the same model. This is a demo feature: the dataset only has whole-tweet labels, so the part colours were checked by eye and not scored.


## Batch endpoint

`POST /predict/batch` takes `{"texts": [...]}` with 1 to 200 texts (each up to 500 characters) and returns one result per text, in the same order. Texts are processed in groups of 32, sorted by length to reduce padding. A text that is empty after cleaning (for example, only a link) gets an `error` message, and the rest of the batch is still scored. Part-by-part colouring is not included in batch mode.

Measured speed for 200 tweets (a single run each, with only a few distinct texts):

| Where | One batch of 200 | One request per tweet (estimated) |
|---|---|---|
| Laptop CPU | 1.4 s | 3.6 s |
| Cloud Run (1 vCPU), measured from my laptop including network | 8.7 s | not measured |

For millions of tweets, a scheduled batch job on a larger machine would be a better fit than this API.

## Known limitations

- **Sarcasm.** "Love how @united lost my bag for the third time. Amazing service!" is predicted positive with 98% confidence.
- **Negation and non-US English.** A tweet praising a cabin crew, written with a negated phrase in informal Indian English, was predicted negative with 95% confidence. The model was trained on US airline tweets only.
- **Confidence is not correctness.** The model is sometimes very confident and wrong.
- **Neutral is the weakest class** (F1 0.70).
- **Rule-based splitting.** Tweets without punctuation stay as one part, and the word "but" does not always mean a contrast.
- **Label noise and one random seed.** Differences of about one point on the validation set are within noise.

## Run it yourself

API (CPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-api.txt
export MODEL_PATH=Bharath-krish-11/tweet-sentiment-distilbert
uvicorn src.api.main:app
```

Docker:

```bash
docker build -t tweet-sentiment-api .
docker run -p 8080:8080 tweet-sentiment-api
```

Frontend:

```bash
cd frontend
echo VITE_API_URL=http://127.0.0.1:8000 > .env.local
npm install
npm run dev
```

The allowed web origins for the API are set with the `CORS_ORIGINS` environment variable (comma separated).

## Repository layout

- `notebooks/` 01 EDA, 02 cleaning, 03 preprocessing, 04 baseline, 05 DistilBERT (trained on a Colab GPU), 06 error analysis
- `src/preprocessing/` shared cleaning code
- `src/api/` FastAPI app, predictor, part splitter
- `frontend/` React page
- `Dockerfile`, `requirements-api.txt` container build

## Future work

- Try a model trained on tweets (for example BERTweet or a Twitter RoBERTa), compared on validation only
- Add data from other regions and airlines
- Better clause splitting, and a way to evaluate part-level labels
- API tests and rate limiting
