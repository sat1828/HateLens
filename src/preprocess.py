"""
preprocess.py
=============
Canonical text preprocessing module for HateLens.

WHY THIS FILE EXISTS:
The preprocessing logic lives here — and only here. The notebook imports from
this file rather than redefining it inline. One canonical implementation prevents
train/inference mismatch, which is the most common silent failure mode in
deployed ML pipelines.

DESIGN NOTE (v2):
This version matches the preprocessing used during model training (the inline
version that was run in the notebook). Key choices:
  - Hashtag words are REMOVED entirely (not just the # symbol)
  - All non-alphabetic characters are stripped (including apostrophes)
  - No explicit negation-word preservation
These choices were made during training. Changing them now would create a
feature space mismatch between the saved model and new inference calls.
"""

import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data on first import.
# WHY: NLTK ships without its corpora; they must be downloaded once.
# This block makes the module self-contained.
for _resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(
            f'tokenizers/{_resource}' if 'punkt' in _resource else f'corpora/{_resource}'
        )
    except LookupError:
        nltk.download(_resource, quiet=True)

# Initialize once at module level — not inside the function.
# WHY: Reinstantiating these objects for every tweet wastes significant
# time across 24,000 rows. Module-level initialization runs exactly once.
_stemmer = PorterStemmer()
_stopwords = set(stopwords.words('english'))


def preprocess_tweet(text: str) -> str:
    """
    Apply the HateLens text normalization pipeline to a single tweet.

    Pipeline (order matters):
      1. Guard against null/non-string input
      2. Lowercase
      3. Remove URLs
      4. Remove @mentions, #hashtag-words, and $cashtags
      5. Strip all non-alphabetic characters
      6. Normalize whitespace
      7. Tokenize
      8. Remove stopwords and single-character tokens
      9. Porter stem
     10. Rejoin into a single string

    Parameters
    ----------
    text : str
        Raw tweet text.

    Returns
    -------
    str
        Cleaned, stemmed tweet ready for TF-IDF vectorization.
    """

    # Guard: handle NaN from pandas and non-string input.
    # WHY: Real-world data pipelines produce malformed inputs.
    # Guard at the entry point so callers never need to check.
    if pd.isna(text) or not isinstance(text, str):
        return ''

    # Step 1 — Lowercase.
    # WHY: 'HATE', 'Hate', 'hate' must map to the same TF-IDF feature.
    text = text.lower()

    # Step 2 — Remove URLs.
    # WHY: 'http://t.co/abc' adds no lexical signal and pollutes the
    # vocabulary with thousands of unique non-generalizable tokens.
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3 — Remove @mentions, #hashtag-words, and $cashtags entirely.
    # WHY @mentions: Twitter infrastructure, not content.
    # WHY #hashtags: '#word' is removed as a unit. Hashtags in this dataset
    # frequently denote metadata (#ff, #tbt) rather than semantic content.
    # Removing the full token is consistent with the training procedure.
    # WHY $cashtags: not relevant to hate speech detection.
    text = re.sub(r'@\w+|#\w+|\$\w+', '', text)

    # Step 4 — Strip all non-alphabetic characters.
    # WHY: Numbers, punctuation, and special characters do not contribute
    # discriminative signal for hate speech in a bag-of-words model.
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 5 — Normalize whitespace.
    # WHY: Previous substitutions leave multiple consecutive spaces.
    # Clean token boundaries matter for word_tokenize accuracy.
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 6 — Tokenize.
    # WHY word_tokenize over str.split: handles edge cases better and is
    # consistent with NLTK's own tokenization contract.
    words = word_tokenize(text)

    # Step 7 — Filter and stem.
    # WHY filter stopwords: 'the', 'is', 'a' appear everywhere and carry
    # no discriminative signal for hate speech classification.
    # WHY len > 1: single-character tokens are almost always noise.
    # WHY stem: 'hating', 'hated', 'hater' all reduce to 'hate',
    # shrinking vocabulary and improving generalization across inflected forms.
    words = [
        _stemmer.stem(w)
        for w in words
        if w not in _stopwords and len(w) > 1
    ]

    # Step 8 — Rejoin.
    # WHY: TfidfVectorizer.transform() expects a string, not a list of tokens.
    return ' '.join(words)


if __name__ == '__main__':
    """Smoke test — run directly to verify pipeline behavior."""
    samples = [
        ("I HATE you @user!! Visit http://t.co/abc #killall right now!!!", 1),
        ("I do not hate people of any religion. #love", 0),
        ("RT @someone: these people are disgusting animals", 1),
        ("Never again will we stand against hatred in our community", 0),
        ("go back to your own country you don't belong here", 1),
    ]
    print("=== preprocess.py smoke test ===\n")
    for text, label in samples:
        print(f"  LABEL: {label}")
        print(f"  RAW  : {text}")
        print(f"  CLEAN: {preprocess_tweet(text)}")
        print()
