<div align="center">

<img width="900" height="200" alt="banner" src="https://github.com/user-attachments/assets/7935ab11-507f-41bb-bf33-14131ecee99c" />

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.10%2B-00C896?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-4A90D9?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Davidson%202017-lightgrey?style=flat-square)
![Class Imbalance](https://img.shields.io/badge/Class%20Imbalance-16.33%3A1-DC3737?style=flat-square)
![Hate Recall](https://img.shields.io/badge/Hate%20Recall-75.52%25-brightgreen?style=flat-square)

<br/>

*Most ML projects optimize a metric. This one optimizes for the person on the receiving end of the content it misses.*

</div>

---

## What This Is

A binary hate speech classifier built the way a real Trust & Safety team would build it — not just the model.

Every production T&S operation has layers that never appear in academic papers: the grey zone your model is uncertain about, the policy document your non-technical manager needs to sign off on, the manual annotations with disagreements logged, the explicit reasoning for why the threshold isn't wherever F1 peaks. HateLens is all of those things, end to end.

The model itself is Logistic Regression on TF-IDF features. That's intentional — a transformer would score better, but the point isn't the model. The point is the workflow.

**What sets this apart from a standard classifier project:**

- A `POLICY_RATIONALE.md` written for a T&S operations manager with zero ML background — covers the core tradeoff, the chosen operating point, and 6 documented failure modes as operational facts
- A threshold of **0.35** chosen because it catches 37 more real hate speech posts than 0.50 — not because it maximizes any metric
- 630 grey-zone tweets identified, 100 manually annotated with a structured protocol (session limits, disagreement logging, escalation flags)
- 14 documented disagreements with the Davidson dataset labels — the AAVE over-labeling problem, found in practice
- Every code cell has a `# WHY:` comment, not just `# WHAT:`

---

## Pipeline Architecture

<img width="900" height="320" alt="pipeline" src="https://github.com/user-attachments/assets/919ba7d5-b88b-4da1-912c-b867b6e29dce" />

The pipeline has a single job: **rank and route**. Nothing gets auto-removed. Posts above the threshold enter a human review queue where a moderator makes the final call. The system handles recall; humans handle precision.

```
Raw tweet → Preprocess → TF-IDF vectorize → Logistic Regression → Probability score
                                                                          │
                           ┌──────────────────────────────────────────────┤
                           ▼                                              ▼
                      p < 0.35                                    0.35–0.65 / p > 0.65
                      NOT HATE                                     HUMAN REVIEW QUEUE
                      (pass)                                       (moderator decides)
```

**No post is removed without a human making the final decision.** This is load-bearing architecture — it's why the threshold can be 0.35 instead of 0.50. When false positives go to a human reviewer and get cleared rather than causing user harm, you can afford to cast a wider net.

---

## Class Imbalance: The Problem You Can't Ignore

<img width="900" height="310" alt="imbalance" src="https://github.com/user-attachments/assets/99e3ac92-f05e-4bab-822c-3dc59b85e0f1" />

```
Davidson et al. 2017 — Original 3-class distribution:
  Class 0: Hate speech       →   1,430   (5.77%)
  Class 1: Offensive lang.   →  19,190  (77.43%)
  Class 2: Neither           →   4,163  (16.80%)
                                 ──────
  Total                         24,783

Binary collapse for this project:
  Label 1 (HATE):      1,430  tweets   ──  5.77%
  Label 0 (NOT HATE): 23,353  tweets   ── 94.23%
  ─────────────────────────────────────────────
  Imbalance ratio:     16.33 : 1
```

Most papers on this dataset cite a 7:1 imbalance. That figure comes from the 3-class split. When you collapse to binary — mapping both *offensive language* and *neither* to label=0 — the minority class shrinks to 1,430 samples against 23,353. The task is harder than usually reported.

**The baseline model's 94.47% accuracy is a trap.** A model that predicts "not hate" for everything would score 94.23%. The baseline, without any imbalance handling, caught only **11.2%** of actual hate speech. It was statistically rewarded for ignoring the problem.

---

## Preprocessing Pipeline

<img width="900" height="310" alt="preprocess" src="https://github.com/user-attachments/assets/7c17467f-6aa4-4071-9ebc-73cf00153b08" />

All text goes through `src/preprocess.py` before the vectorizer sees it:

```python
def preprocess_tweet(tweet: str) -> str:
    tweet = re.sub(r'@\w+', '', tweet)          # strip @mentions
    tweet = re.sub(r'http\S+', '', tweet)        # strip URLs
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)   # strip special chars
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)
```

This function is the single source of truth for preprocessing — imported identically by the notebook, by `train.py`, and by any inference endpoint. There's no version drift between training and serving.

---

## Model Comparison

<img width="900" height="360" alt="results" src="https://github.com/user-attachments/assets/229477d3-8458-456d-b755-beee810abb5d" />

Four variants were trained and compared. Each one was a deliberate step:

| Model | Accuracy | Hate F1 | Hate Recall | Hate Precision |
|---|---|---|---|---|
| Baseline (no fix) | 0.9447 | 0.1893 | 0.1119 | 0.6154 |
| + SMOTE | 0.8943 | 0.3907 | 0.5874 | 0.2927 |
| + Class Weights | 0.8953 | 0.4082 | 0.6259 | 0.3029 |
| **✓ Class Weights + Threshold 0.35** | **0.8301** | **0.3391** | **0.7552** | **0.2186** |

**Why Class Weights over SMOTE?** SMOTE generates synthetic minority-class samples in feature space. For text, that means synthesized TF-IDF vectors that don't correspond to real linguistic patterns. Class Weights adjust the loss function directly — a misclassified hate speech post costs 16.33× more than a misclassified not-hate post. No data fabrication required.

**Why does the final model have lower F1 than Class Weights alone?** Because F1 is not what this system optimizes. See below.

---

## The Threshold Decision

<img width="900" height="340" alt="threshold" src="https://github.com/user-attachments/assets/6b35efe1-e429-4da5-b212-b7b22520d30e" />

The F1-maximizing threshold is 0.50. The chosen threshold is 0.35. Here's the exact trade on the test set:

```
Test set: 4,957 posts · 286 actual hate speech instances
─────────────────────────────────────────────────────────
                      Threshold 0.35    Threshold 0.50
Posts flagged:             988               591
True positives:            216               179
False positives:           772               412
Hate posts MISSED:          70               107
─────────────────────────────────────────────────────────
                           ▲ 37 more hate speech posts caught
                           ▼ 360 more posts enter the review queue
```

37 posts. That's real people whose targeted harassment gets surfaced for review vs. remaining on the platform undetected. The 360 additional false positives go to a human reviewer, get cleared, and the author is never penalized.

When the two failure modes have asymmetric consequences — one causes psychological harm to a targeted person, the other costs a reviewer 30 seconds — you tolerate more of the less harmful failure. That's not a novel idea. Meta, YouTube, and every platform that publishes transparency reports documents this reasoning for their classifier calibration. This project does the same thing, in `POLICY_RATIONALE.md`, for an audience of one future T&S manager.

---

## Grey Zone Annotation

<img width="900" height="320" alt="annotation" src="https://github.com/user-attachments/assets/16f6b590-ea87-48de-86af-df3d597438db" />

After training, **630 tweets** fell in the uncertain zone (predicted probability 0.35–0.65). 100 were manually reviewed using a formal annotation protocol.

**Protocol:**
- Maximum 50 tweets per session
- Mandatory 15-minute break between sessions
- Binary label per tweet (0 = not hate, 1 = hate)
- Confidence flag for genuinely ambiguous cases
- Disagreements with Davidson labels explicitly logged

**Results from 100 annotations:**

```
Label distribution:
  Not hate (0):    91 / 100   ──  91%
  Hate (1):         9 / 100   ──   9%

Disagreements with Davidson labels:  14
  Pattern: AAVE with reclaimed slurs
  Davidson labeled as hate → human labeled as not hate

Escalation flags (ambiguous):  2
  → Would route to senior reviewer in production

Violations by protected category:
  Race / ethnicity      ███  3
  Sexual orientation    ███  3
  Disability             █   1
  Religion               █   1
  National origin        █   1
```

The 14 disagreements with Davidson are not annotation errors. The dataset has a documented bias toward labeling African American Vernacular English as hate speech when certain reclaimed slurs appear regardless of speaker identity or in-group usage context. Finding 14 instances in 100 annotations is the bias showing up in practice, not a coincidence.

---

## Repository Architecture

<img width="900" height="360" alt="architecture" src="https://github.com/user-attachments/assets/0a8f68f2-fe26-4362-a173-c8cd658c08e0" />

```
HateLens/
│
├── README.md                              ← this file
├── POLICY_RATIONALE.md                    ← threshold justification for non-technical stakeholders
├── requirements.txt                       ← 11 dependencies
├── .gitignore
│
├── notebooks/
│   └── HateLens_Full_Pipeline.ipynb      ← 10-section complete pipeline
│                                            all outputs committed
│                                            every cell has a WHY: comment
│
├── src/
│   ├── preprocess.py                     ← canonical preprocessing (single source of truth)
│   ├── train.py                          ← CLI retraining script (argparse)
│   └── evaluate.py                       ← metrics, PR curves, confusion matrices
│
├── data/
│   ├── labeled_data.csv                  ← NOT committed (Davidson's IP — download separately)
│   ├── binary_labeled_data.csv           ← generated by Section 2
│   ├── policy_decision_log_template.csv  ← 500-row grey zone annotation template
│   └── policy_decision_log_filled.csv    ← 100 completed human annotations
│
└── outputs/
    ├── class_distribution.png
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_final.png        ← threshold=0.35
    ├── precision_recall_curve.png
    ├── model_comparison_chart.png        ← all 4 variants
    ├── grey_zone_categories.png
    ├── model.joblib                      ← serialized LogisticRegression
    └── vectorizer.joblib                 ← fitted TF-IDF vectorizer
```

**Notebook structure — 10 sections:**

| Section | Content |
|---|---|
| 1 | Setup & imports |
| 2 | Data loading + binary label collapse + distribution analysis |
| 3 | Text preprocessing — tweet cleaning + NLTK pipeline |
| 4 | TF-IDF vectorization — unigrams + bigrams, tuned min/max_df |
| 5 | Baseline model — no imbalance handling (establishing the floor) |
| 6 | SMOTE experiment — oversampling in feature space |
| 7 | Class weights — selected as base model |
| 8 | Threshold analysis — precision-recall curve, operating point selection |
| 9 | Grey zone annotation — protocol, 100 labels, findings |
| 10 | Final results table + model serialization |

---

## Stack

| Layer | Library | Version | Role |
|---|---|---|---|
| Data | `pandas` | ≥ 1.5 | Ingestion, label collapse, splits |
| ML | `scikit-learn` | ≥ 1.2 | TF-IDF, LogReg, metrics |
| Imbalance | `imbalanced-learn` | ≥ 0.10 | SMOTE (compared, not final) |
| NLP | `nltk` | ≥ 3.8 | Tokenization, stopwords, lemmatization |
| Persistence | `joblib` | ≥ 1.2 | Model + vectorizer serialization |
| Visualization | `matplotlib` + `seaborn` | ≥ 3.6 / ≥ 0.12 | All output charts |
| Word clouds | `wordcloud` | ≥ 1.9 | Grey zone vocabulary analysis |
| Environment | `jupyter` + `ipykernel` | ≥ 1.0 / ≥ 6.0 | Notebook execution |

---

## Running It

### Clone and install

```bash
git clone https://github.com/sat1828/HateLens.git
cd HateLens
pip install -r requirements.txt
```

### Get the dataset

`labeled_data.csv` is Davidson et al.'s work — not ours to redistribute.

```bash
# Download from:
# https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data
# Place at: data/labeled_data.csv
```

### Full pipeline notebook

```bash
jupyter notebook notebooks/HateLens_Full_Pipeline.ipynb
# Kernel → Restart & Run All
# Expected runtime: 3–5 minutes
# Section 10 prints all final numbers
```

### Retrain from CLI

```bash
python src/train.py \
  --data data/binary_labeled_data.csv \
  --model_out outputs/model.joblib \
  --vectorizer_out outputs/vectorizer.joblib
```

### Smoke test the saved model

```python
import joblib, sys
sys.path.insert(0, 'src')
from preprocess import preprocess_tweet

model      = joblib.load('outputs/model.joblib')
vectorizer = joblib.load('outputs/vectorizer.joblib')

tweets = [
    "I love how diverse our community is becoming",
    "People from that country are all criminals and should be deported",
    "Go back to where you came from, you don't belong here",
]

for tweet in tweets:
    vec   = vectorizer.transform([preprocess_tweet(tweet)])
    prob  = model.predict_proba(vec)[0][1]
    label = 'HATE' if prob >= 0.35 else 'NOT HATE'
    print(f"{label} ({prob:.3f}): {tweet[:60]}")
```

---

## Known Limitations

These aren't disclaimers. They're operational facts for anyone deploying this system.

**Coded language failure** — The TF-IDF model cannot detect dog-whistle terms, numerical codes (14/88), recently coined euphemisms, or in-group terminology that evolves faster than retraining cycles. Requires either continuous manual lexicon updates by a subject-matter expert, or a context-aware transformer.

**Context blindness** — "I hate racists" and "I hate [racial group]" may score similarly. TF-IDF sees word presence, not intent. A news article quoting hate speech and an actual hate speech post may receive the same score.

**English-only** — Trained on English Twitter. Not validated on other languages, transliterations, or code-switching. Should not be used for multilingual content without retraining.

**Static training data** — The Davidson 2017 dataset is nearly a decade old. Coded language evolves. This model reflects hate speech vocabulary from 2017. Recommended minimum retraining cycle: 6 months, with drift monitoring to trigger earlier retraining.

**Platform specificity** — Performance on Reddit, YouTube, Discord, or any platform other than Twitter is not validated and may differ substantially due to different norms, vocabulary, and communication styles.

**Severe class imbalance** — 16.33:1 is harder than the commonly cited 7:1. Even with class weighting, hate class F1 at the selected threshold is 0.34. This is the practical ceiling for TF-IDF + LogReg on this distribution.

---

## Path to Production

The `POLICY_RATIONALE.md` documents the full production roadmap. Summary:

1. **Upgrade to HateBERT** — BERT pretrained on banned Reddit content understands context, not just word presence. Expected hate F1 improvement: 0.34 → 0.70+.

2. **Add character-level features** — Intentional misspellings ("h4te", "k!ll") designed to evade word-level detection require character n-gram features.

3. **Build active learning** — Every human reviewer decision is a labeled training sample. Feeding those back into the model via an active learning pipeline turns the review queue into a continuous training signal.

4. **Deploy drift detection** — Monitor for distribution shift between training data and live content. Vocabulary divergence is the earliest signal that the model is going stale.

5. **Reviewer wellbeing program** — Content moderation has documented psychological consequences from repeated exposure to harmful material. Any production deployment requires session limits, mandatory breaks, psychological support resources, and rotation off high-severity queues. This is in the production recommendations because it belongs there.

---

## Dataset Credit

Davidson, T., Warmsley, D., Macy, M. & Weber, I. (2017). "Automated Hate Speech Detection and the Problem of Offensive Language." *Proceedings of the 11th AAAI International Conference on Web and Social Media.*

Source: [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)

---

<div align="center">

*Built for the T&S operations teams who never see the model — only the queue it fills.*

</div>
