# Feedback2Strategy

Turn customer feedback into ranked, evidence-linked investigation queues. The app clusters English comments with TF-IDF and deterministic K-means, scores tone with VADER, and ranks themes using a visible negative-volume × severity heuristic. Every theme includes verbatim feedback examples and a concrete validation step.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use `data/sample_feedback.csv` for an explicitly synthetic demonstration or upload a CSV with a `text` column. An optional `feedback_id` column must be unique; an optional `segment` column supports breakdowns.

## Method and limits

- TF-IDF with unigrams/bigrams and K-means proposes exploratory themes; clusters are not ground-truth categories.
- VADER is a lexicon/rule sentiment baseline designed for English text. Domain language, sarcasm, multilingual input, and cultural context can produce errors.
- The score is `negative comment count × mean absolute VADER compound score` within a theme. It measures a triage signal only. It does not estimate churn, revenue impact, causal effect, or customer value.
- Next steps are fixed, evidence-linked investigation prompts, not AI-generated strategy. Review the underlying comments and validate root causes with customers and product data before acting.

All sample comments are fictional and synthetic. Do not upload personal, sensitive, or confidential data to an unapproved environment. Redact identifiers before local analysis.

## Checks

```bash
python -m pytest
```

## License

MIT. See `LICENSE`.
