# Naive Bayes — App Review Sentiment Analysis

> NLP classification pipeline that predicts whether a Google Play Store review is positive or negative: text preprocessing with CountVectorizer, MultinomialNB selected over Gaussian and Bernoulli variants on theoretical grounds, then optimised with RandomizedSearchCV — with a Random Forest comparison showing the tradeoff between interpretability and ensemble power.

---

## Problem

Classify Google Play Store user reviews as positive or negative sentiment. Automating this lets app developers monitor user satisfaction at scale without manually reading thousands of reviews. This is a binary NLP classification problem.

## Dataset

- **Source:** Google Play Store Reviews dataset (GitHub / 4Geeks)
- **Target:** `polarity` — 1 = positive sentiment, 0 = negative sentiment
- **Raw features:** `package_name` (dropped), `review` (text — the only predictor)

## Pipeline

| Step | Tool / Approach |
|---|---|
| Text cleaning | Strip whitespace, convert to lowercase |
| Vectorization | `CountVectorizer(stop_words="english")` → bag-of-words word count matrix |
| Train/test split | 80/20, random_state=42 |
| Model selection | MultinomialNB chosen — counts-based features match its probabilistic assumptions |
| Optimisation | `RandomizedSearchCV` (50 iterations, 5-fold CV) over `alpha` and `fit_prior` |
| Best hyperparams | `alpha=1.918`, `fit_prior=False` |

**Why MultinomialNB over Gaussian or Bernoulli?**
- **MultinomialNB** is designed for count data (word frequencies) — a natural fit for Bag-of-Words
- **GaussianNB** assumes continuous normally-distributed features — incorrect for integer word counts
- **BernoulliNB** treats each word as binary (present/absent) — loses frequency information
- All three were trained and compared; MultinomialNB confirmed best empirically and theoretically

## Model Comparison

| Model | Notes |
|---|---|
| MultinomialNB (baseline) | Default alpha=1.0 |
| MultinomialNB (tuned) | alpha=1.918, fit_prior=False — improved accuracy |
| RandomForestClassifier (n_estimators=60) | ~80% accuracy — strong ensemble baseline |

The Random Forest achieves higher raw accuracy but at the cost of interpretability. Naive Bayes is faster to train, requires far less memory on high-dimensional text data, and its word-level probabilities are directly inspectable.

## Key Takeaways

- **Bag-of-Words is a powerful baseline:** Ignoring word order and treating reviews as unordered word counts still captures enough signal to classify sentiment reliably — a strong reminder that simpler representations often work well before reaching for embeddings.
- **Algorithm choice should follow the data type:** The theoretical reason for choosing MultinomialNB (count data) over GaussianNB (continuous data) is as important as the empirical accuracy comparison.
- **`fit_prior=False` flattens the class prior:** When classes are imbalanced, letting the model learn the prior from training data can bias it toward the majority class — setting `fit_prior=False` assumes equal priors and improves minority-class recall.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NLTK (stop words)`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_NaiveBayes_MTK.git
cd ML_NaiveBayes_MTK
pip install -r requirements.txt
jupyter notebook src/app.ipynb
```

The trained model is saved to `models/` via `pickle`.

## What I'd Do Next

- Replace CountVectorizer with **TF-IDF** to down-weight common words that appear across all reviews and up-weight distinctive terms
- Add a **confusion matrix heatmap** and per-class precision/recall — overall accuracy on imbalanced sentiment data is a misleading single number
- Try **word embeddings** (word2vec or sentence-transformers) to capture semantic meaning rather than just word frequency, and compare against the BoW baseline

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
