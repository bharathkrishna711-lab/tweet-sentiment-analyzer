import html
import re

import emoji

EMOTICONS = {
    ':)': 'smile', ':-)': 'smile', ':D': 'smile', ':-D': 'smile',
    ':(': 'sad', ':-(': 'sad',
    ';)': 'wink', ';-)': 'wink',
    '<3': 'love',
}


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def convert_emoticons(text):
    for emo, word in EMOTICONS.items():
        pattern = r'(?<!\S)' + re.escape(emo) + r'(?=\s|$|[.!?,])'
        text = re.sub(pattern, word, text)
    return text


def convert_emojis(text):
    text = emoji.demojize(text, delimiters=(' ', ' '), language='alias')
    text = emoji.demojize(text, delimiters=(' ', ' '))  # emojis with no alias
    return text.replace('_', ' ')


def clean_tweet(text):
    """Shared cleaning rules: used for training data and at inference."""
    text = html.unescape(text)
    text = re.sub(r'http\S+', '', text)       # URLs
    text = re.sub(r'@\w+', '', text)          # @mentions
    text = text.replace('#', '')              # hashtag symbol, keep the word
    text = normalize_whitespace(text)
    text = convert_emoticons(text)
    text = convert_emojis(text)
    return normalize_whitespace(text)
