import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.api.predictor import SentimentPredictor

state = {}

ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')


@asynccontextmanager
async def lifespan(app):
    state['predictor'] = SentimentPredictor()
    yield
    state.clear()


app = FastAPI(title='Tweet sentiment API', version='0.2.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class Part(BaseModel):
    text: str
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    parts: list[Part]


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'predictor' in state}


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse(url='/docs')


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail='Text must not be blank.')
    result = state['predictor'].predict(req.text)
    if not result['cleaned_text']:
        raise HTTPException(
            status_code=422,
            detail='Nothing left to analyse after cleaning (for example, only a link or a mention).',
        )
    return result