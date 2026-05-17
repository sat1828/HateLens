"""
train.py
========
Model training script for HateLens.

WHY THIS FILE EXISTS:
Training logic is separated from the notebook so that the model can be
retrained from the command line without reopening Jupyter. This supports
the production recommendation of regular retraining on updated data.

Usage:
    python src/train.py --data data/binary_labeled_data.csv \
                        --model_out outputs/model.joblib \
                        --vectorizer_out outputs/vectorizer.joblib
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our canonical preprocessing function.
# WHY: We must use the exact same preprocessing at training time AND
# inference time. If we used different logic, the model would see
# different feature distributions at test time — a subtle data leakage bug.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_tweet


def build_vectorizer() -> TfidfVectorizer:
    """
    Construct the TF-IDF vectorizer with HateLens-specific settings.

    WHY THESE SETTINGS:
    - max_features=50000: caps vocabulary size to control memory and
      prevent overfitting to extremely rare terms.
    - ngram_range=(1,2): includes bigrams like "go kill" or "you deserve"
      which carry meaning that individual words do not.
    - min_df=3: terms appearing in fewer than 3 documents are almost
      certainly typos or noise — ignoring them reduces overfitting.
    - sublinear_tf=True: applies log(1 + tf) instead of raw tf.
      Prevents very frequent terms from dominating the feature space.
    """
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
    )


def train(data_path: str, model_out: str, vectorizer_out: str):
    """
    Full training pipeline: load → split → vectorize → train → save.

    Parameters
    ----------
    data_path : str
        Path to binary_labeled_data.csv (output of Section 2).
    model_out : str
        Where to save the trained LogisticRegression model (joblib).
    vectorizer_out : str
        Where to save the fitted TfidfVectorizer (joblib).
    """

    # --- Load data ---
    # WHY: We load from binary_labeled_data.csv (not the raw file) so that
    # the binary label collapse (Section 2) is already applied.
    print(f"[1/5] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # --- Preprocess text if not already done ---
    # WHY: If the CSV already has a 'clean_tweet' column (written by the
    # notebook), we reuse it to save time. If not, we apply preprocessing.
    if 'clean_tweet' not in df.columns:
        print("[2/5] Preprocessing tweets (this may take ~60 seconds)...")
        df['clean_tweet'] = df['tweet'].apply(preprocess_tweet)
    else:
        print("[2/5] Found existing 'clean_tweet' column — skipping preprocessing.")

    # --- Train/test split (stratified) ---
    # WHY stratify=y: With ~16:1 class imbalance, a random split risks putting
    # too few hate speech examples in the test set, making evaluation
    # unreliable. Stratification preserves the class ratio in both splits.
    print("[3/5] Splitting data (80/20, stratified)...")
    X = df['clean_tweet']
    y = df['label']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- TF-IDF vectorization ---
    # CRITICAL: We fit ONLY on the training set, then transform both sets.
    # WHY: Fitting on the test set would expose the model to test vocabulary
    # during feature construction — a form of data leakage that inflates
    # performance metrics and leads to over-optimistic results in production.
    print("[4/5] Fitting TF-IDF vectorizer on training data only...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"      Training matrix shape: {X_train_tfidf.shape}")

    # --- Train class-weighted Logistic Regression ---
    # WHY class_weight='balanced': The hate speech class (~14% of data) would
    # otherwise be swamped by the majority class. 'balanced' weights each
    # class inversely proportional to its frequency, penalizing errors on
    # the minority class more heavily during gradient updates.
    # This is mathematically equivalent to oversampling without generating
    # artificial data — making it more reliable for sparse TF-IDF vectors.
    print("[5/5] Training Logistic Regression (class_weight='balanced')...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs',
    )
    model.fit(X_train_tfidf, y_train)

    # --- Save artifacts ---
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(vectorizer, vectorizer_out)
    print(f"\nModel saved to:      {model_out}")
    print(f"Vectorizer saved to: {vectorizer_out}")
    print("\nTraining complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HateLens classifier.")
    parser.add_argument('--data', default='data/binary_labeled_data.csv')
    parser.add_argument('--model_out', default='outputs/model.joblib')
    parser.add_argument('--vectorizer_out', default='outputs/vectorizer.joblib')
    args = parser.parse_args()
    train(args.data, args.model_out, args.vectorizer_out)
