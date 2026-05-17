# HateLens: Policy Rationale & Threshold Decision

> **Audience:** This document is written for a non-technical audience —
> specifically, a Trust & Safety policy analyst or operations manager
> who will use this system's output to manage a content review queue.

---

## 1. Purpose

This document explains the calibration decisions made in building the HateLens classifier.
It is intended to be readable by anyone involved in platform policy or content operations,
regardless of their technical background.

HateLens is a **first-pass filter** — not an auto-removal system. Its job is to rank
incoming posts by their probability of being hate speech, so that human reviewers can
focus their attention on the most likely violations first. No post is removed without
a human making the final call.

---

## 2. The Core Trade-Off

Every content moderation classifier operates between two failure modes:

**False Negative (Under-removal):**
A genuinely hateful post is classified as "not hate" and remains on the platform.
A real person is exposed to content targeting them based on race, religion, gender,
or other protected characteristics. The targeted user experiences harm. The platform
bears reputational and legal risk.

**False Positive (Over-removal):**
A legitimate post is incorrectly classified as hate speech and sent to the human review queue.
A real person's speech is temporarily suppressed pending review. If this disproportionately
affects counter-speech or educational content, the moderation system itself causes harm.
Additionally, each false positive consumes reviewer time and budget.

These two failure modes are in direct tension. Lowering the threshold catches more real
hate speech (fewer false negatives) but also flags more legitimate content (more false positives).
There is no threshold that eliminates both. The choice of operating point is a **policy decision**,
not a technical one — it depends on the platform's values, reviewer capacity, and legal obligations.

---

## 3. Our Chosen Operating Point

We selected a decision threshold of **0.35** (compared to the default of 0.50), which produces:

| Metric | Value |
|--------|-------|
| Precision on hate class | 0.2186 |
| Recall on hate class    | 0.7552 |
| F1 on hate class        | 0.3391 |

In plain English:
- For every 100 posts flagged as hate speech, approximately **22 are genuine violations**.
- For every 100 genuine hate speech posts in the content stream, approximately **76 are caught** and sent to human review.

Compared to the F1-maximizing threshold (0.50):

| | Threshold = 0.35 | Threshold = 0.50 |
|---|---|---|
| Precision | 0.2186 | 0.3029 |
| Recall    | 0.7552 | 0.6259 |
| Posts caught (test set) | **216 / 286** | 179 / 286 |
| Posts missed | **70** | 107 |

We accept lower precision (and lower F1) to catch **37 more hate speech posts** per test-set equivalent.
Those 37 posts would otherwise remain on the platform undetected.

---

## 4. Rationale for This Threshold

We prioritized **RECALL over PRECISION** for three reasons:

### 4.1 Severity Asymmetry

The harm caused by a missed hate speech post is not symmetric with the harm caused by
a wrongly flagged post.

A **missed hate speech post** means a real person — targeted because of their race, religion,
gender, sexual orientation, disability, or national origin — is exposed to content designed
to dehumanize or threaten them. This causes direct psychological harm and, in documented cases,
has preceded offline violence.

A **wrongly flagged post** means someone's tweet enters a human review queue. If the human
reviewer determines it is not hate speech, the post is restored and the author is not penalized.
This causes inconvenience and, in some cases, justifiable frustration. It does not cause
direct harm to a third party.

When the two failure modes have asymmetric consequences, the rational approach is to tolerate
more of the less harmful failure (false positives) to reduce the more harmful failure (false negatives).

### 4.2 Human-in-the-Loop Buffer

This system does **NOT auto-remove content.**
Posts above the threshold enter a **HUMAN REVIEW QUEUE.**
A human moderator makes the final removal decision.

This is critical context for understanding the threshold choice:
- Higher recall → more posts in the queue
- More posts in queue → more reviewer time required
- **Zero posts are removed without human confirmation**

The false positive rate matters for **reviewer workload**, not for user harm.
At threshold=0.35, the additional review volume vs. threshold=0.50 is approximately
**305 extra posts per test set** (4,957 posts). Scaled to a live platform, this cost
must be weighed explicitly against reviewer capacity.

### 4.3 Alignment With Industry Practice

Platforms including Meta and YouTube have documented (in transparency reports and
policy papers) that high-severity content categories — incitement to violence,
hate speech targeting protected groups, CSAM proactive detection — use
recall-optimized thresholds at the classifier level, with precision protected
at the human review stage.

The industry rationale is identical to ours: automated systems catch; humans verify.

---

## 5. Known Limitations of This System

These limitations are not disclaimers. They are **operational facts** that anyone using
this system's output must understand.

### 5.1 Coded Language Failure

The TF-IDF model cannot detect dog-whistle terms, coded references, or recently coined slang.
Examples: numerical codes (14/88), recently adopted euphemisms, and in-group terminology
that changes faster than the model can be retrained.

These require either:
- Continuous manual lexicon updates by a subject-matter expert
- A context-aware model (e.g., fine-tuned BERT) that understands semantic relationships

### 5.2 Context Blindness

"I hate racists" and "I hate [racial group]" may score similarly in this model.
TF-IDF sees the word "hate" — it does not see intent.

Similarly, a news article quoting hate speech, a researcher analyzing it, and
an actual hate speech post may all receive similar scores because they share vocabulary.

### 5.3 English-Only

This model was trained on English-language Twitter data.
It should **not** be used for multilingual content without retraining.
Hate speech in other languages, transliterations, or code-switching will not be
detected reliably and may not be detected at all.

### 5.4 Static Training Data

Language evolves. New slurs emerge, coded terms shift meaning, and previously
innocent phrases become associated with hate movements. This model reflects the
vocabulary of hate speech as it existed in the Davidson 2017 dataset — now nearly
a decade out of date.

**This model requires regular retraining on updated data to remain effective.**
We recommend a minimum retraining cycle of every 6 months, with drift monitoring
to trigger earlier retraining when vocabulary patterns shift significantly.

### 5.5 Platform-Specificity

The Davidson 2017 dataset is from Twitter. Performance on other platforms —
Reddit comments, YouTube comments, WhatsApp messages, Discord servers —
is **not validated** and may differ substantially due to different norms,
vocabulary, and communication styles on each platform.

### 5.6 Severe Class Imbalance

The dataset contains 16.33:1 imbalance (not hate : hate). This is more extreme than the
"approximately 7:1" figure referenced in many papers on this dataset, because this project
maps both "offensive language" and "neither" to the not-hate class for the binary task.
This imbalance materially limits model performance — even with class weighting, the model
achieves F1=0.34 on the hate class at the selected threshold.

---

## 6. Recommendation for Production Deployment

This system is suitable as a **first-pass filter** to prioritize the human review queue.
It is **NOT suitable** as a standalone removal system.

For production use, we recommend:

1. **Upgrade to a fine-tuned transformer model.** HateBERT (BERT pre-trained on
   banned Reddit content) substantially outperforms TF-IDF + Logistic Regression
   on hate speech detection because it understands context, not just word presence.
   Expected F1 improvement: 0.34 → 0.70+.

2. **Add character-level features** to catch intentional misspellings
   ("h4te", "k!ll", "n!gger") that are designed to evade word-level detection.

3. **Build an active learning pipeline** where human reviewer decisions
   continuously feed back into model retraining. Each human label is a training
   signal; wasting it is a resource failure.

4. **Monitor for distribution shift.** Deploy a drift detection system that
   alerts when the vocabulary or distribution of incoming content diverges
   from the training distribution. This is the earliest warning sign that
   the model is becoming stale.

5. **Maintain a reviewer wellbeing program.** Content moderation involves
   repeated exposure to harmful material. This has documented psychological
   consequences. Any production deployment must include review session limits,
   mandatory breaks, psychological support resources, and regular rotation off
   high-severity queues.

---

*Document version: 1.1 | Dataset: Davidson et al. 2017 | Model: Logistic Regression + TF-IDF*
*Class imbalance: 16.33:1 | Final threshold: 0.35 | Hate class recall: 0.7552 | Hate class precision: 0.2186*
