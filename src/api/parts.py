import re

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
CLAUSE_BREAK = re.compile(r'(?:\s*[,;\-\u2013\u2014]+\s*|\s+)(?=(?:but|however|although|though)\b)', re.IGNORECASE)
MIN_WORDS = 2


def split_parts(text):
    pieces = []
    for sentence in SENTENCE_END.split(text):
        pieces.extend(CLAUSE_BREAK.split(sentence))
    pieces = [p.strip() for p in pieces if p.strip()]

    merged = []
    for p in pieces:
        if merged and len(p.split()) < MIN_WORDS:
            merged[-1] += ' ' + p
        else:
            merged.append(p)
    return merged