import pytest
from fastapi.testclient import TestClient

from src.api.main import app

LABELS = {'negative', 'neutral', 'positive'}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'model_loaded': True}


def test_root_redirects_to_docs(client):
    r = client.get('/', follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers['location'] == '/docs'


def test_predict_normal_tweet(client):
    r = client.post('/predict', json={'text': 'worst flight ever, lost my bag again'})
    assert r.status_code == 200
    body = r.json()
    assert body['label'] in LABELS
    assert 0 <= body['confidence'] <= 1
    assert set(body['probabilities']) == LABELS
    assert abs(sum(body['probabilities'].values()) - 1) < 0.01
    assert len(body['parts']) >= 1


def test_predict_returns_parts(client):
    r = client.post('/predict', json={'text': 'thanks for the quick help, but I waited two hours on hold'})
    assert r.status_code == 200
    parts = r.json()['parts']
    assert len(parts) == 2
    assert all(p['label'] in LABELS for p in parts)
    assert all(0 <= p['confidence'] <= 1 for p in parts)


def test_blank_text_is_rejected(client):
    r = client.post('/predict', json={'text': '   '})
    assert r.status_code == 422


def test_link_only_tweet_is_rejected(client):
    r = client.post('/predict', json={'text': '@united https://t.co/abc123'})
    assert r.status_code == 422
    assert 'Nothing left' in r.json()['detail']


def test_text_over_500_characters_is_rejected(client):
    r = client.post('/predict', json={'text': 'a' * 501})
    assert r.status_code == 422


def test_missing_text_field_is_rejected(client):
    r = client.post('/predict', json={})
    assert r.status_code == 422

def test_batch_returns_one_result_per_text_in_order(client):
    texts = ['worst flight ever, lost my bag again', 'thanks for the quick help', 'Flight 482 now boards at gate B12.']
    r = client.post('/predict/batch', json={'texts': texts})
    assert r.status_code == 200
    results = r.json()['results']
    assert len(results) == 3
    assert all(x['label'] in LABELS for x in results)
    assert results[0]['cleaned_text'] == texts[0]


def test_batch_matches_single_prediction(client):
    text = 'worst flight ever, lost my bag again'
    single = client.post('/predict', json={'text': text}).json()
    batch = client.post('/predict/batch', json={'texts': [text, 'thanks for the quick help']}).json()['results'][0]
    assert batch['label'] == single['label']
    assert abs(batch['confidence'] - single['confidence']) < 0.01


def test_batch_bad_item_does_not_fail_the_batch(client):
    r = client.post('/predict/batch', json={'texts': ['@united https://t.co/abc123', 'great flight, thank you']})
    assert r.status_code == 200
    results = r.json()['results']
    assert results[0]['label'] is None
    assert 'Nothing left' in results[0]['error']
    assert results[1]['label'] in LABELS


def test_batch_empty_list_is_rejected(client):
    assert client.post('/predict/batch', json={'texts': []}).status_code == 422


def test_batch_too_many_texts_is_rejected(client):
    assert client.post('/predict/batch', json={'texts': ['ok'] * 201}).status_code == 422


def test_batch_text_over_500_characters_is_rejected(client):
    assert client.post('/predict/batch', json={'texts': ['a' * 501]}).status_code == 422