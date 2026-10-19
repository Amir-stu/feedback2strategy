"""Evidence-linked customer feedback triage using reproducible NLP baselines."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class FeedbackValidationError(ValueError):
    """Raised when feedback input is malformed or unsafe to analyze."""


_SENTIMENT = SentimentIntensityAnalyzer()


def validate_feedback(frame: pd.DataFrame) -> pd.DataFrame:
    if "text" not in frame.columns:
        raise FeedbackValidationError("CSV must contain a text column.")
    if frame.empty:
        raise FeedbackValidationError("No feedback rows were supplied.")
    if len(frame) > 20_000:
        raise FeedbackValidationError("This local prototype accepts at most 20,000 feedback rows per run.")
    data = frame.copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    if data["text"].eq("").any():
        raise FeedbackValidationError("Feedback text cannot be blank. Remove empty rows and retry.")
    if data["text"].str.len().gt(10_000).any():
        raise FeedbackValidationError("Each feedback item must be 10,000 characters or fewer.")
    if "feedback_id" not in data:
        data["feedback_id"] = [f"FB-{i:05d}" for i in range(1, len(data) + 1)]
    else:
        data["feedback_id"] = data["feedback_id"].astype(str).str.strip()
        if data["feedback_id"].eq("").any() or data["feedback_id"].duplicated().any():
            raise FeedbackValidationError("feedback_id values must be present and unique.")
    if "segment" not in data:
        data["segment"] = "Unspecified"
    data["segment"] = data["segment"].fillna("Unspecified").astype(str).str.strip().replace("", "Unspecified")
    return data.reset_index(drop=True)


def _theme_label(terms: list[str]) -> str:
    if not terms:
        return "General feedback"
    return " · ".join(term.title() for term in terms[:3])


def analyze_feedback(frame: pd.DataFrame, requested_themes: int = 4, seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one annotated feedback row per input and an evidence-linked theme table."""
    data = validate_feedback(frame)
    if not 2 <= requested_themes <= 10:
        raise FeedbackValidationError("Choose between 2 and 10 themes.")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=3000, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(data["text"])
    except ValueError as exc:
        raise FeedbackValidationError("Feedback text has no analyzable English terms after tokenization.") from exc
    signatures = set()
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        signatures.add((tuple(matrix.indices[start:end]), tuple(np.round(matrix.data[start:end], 10))))
    unique_rows = len(signatures)
    n_clusters = min(requested_themes, len(data), unique_rows)
    if n_clusters < 1:
        n_clusters = 1
    if n_clusters == 1:
        labels = np.zeros(len(data), dtype=int)
        centers = np.asarray(matrix.mean(axis=0))
    else:
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20, algorithm="lloyd")
        labels = model.fit_predict(matrix)
        centers = model.cluster_centers_

    terms = np.asarray(vectorizer.get_feature_names_out())
    annotated = data.copy()
    annotated["theme_id"] = labels
    scores = annotated["text"].map(_SENTIMENT.polarity_scores).map(lambda result: result["compound"])
    annotated["sentiment_compound"] = scores.astype(float)
    annotated["sentiment"] = np.select([scores <= -0.05, scores >= 0.05], ["Negative", "Positive"], default="Neutral")

    themes = []
    for cluster_id in sorted(annotated["theme_id"].unique()):
        subset = annotated.loc[annotated["theme_id"] == cluster_id]
        if n_clusters == 1:
            center = centers.reshape(-1)
        else:
            center = centers[cluster_id]
        top_indices = np.argsort(center)[::-1]
        top_terms = [term for term in terms[top_indices] if re.search(r"[a-zA-Z]", term)][:5]
        negative = subset.loc[subset["sentiment"] == "Negative", "sentiment_compound"]
        severity = float(negative.abs().mean()) if len(negative) else 0.0
        negative_count = int(len(negative))
        score = float(negative_count * severity)

        indices = subset.index.to_numpy()
        if n_clusters == 1:
            representative_index = indices[0]
        else:
            selected = matrix[indices]
            center = centers[cluster_id]
            distances = np.asarray(selected.multiply(selected).sum(axis=1)).reshape(-1) - 2 * selected.dot(center) + center.dot(center)
            representative_index = indices[int(np.argmin(distances))]
        evidence = str(annotated.loc[representative_index, "text"])
        label = _theme_label(top_terms)
        themes.append(
            {
                "theme_id": int(cluster_id),
                "theme": label,
                "feedback_count": int(len(subset)),
                "negative_count": negative_count,
                "negative_share": float(negative_count / len(subset)),
                "mean_negative_severity": severity,
                "priority_score": score,
                "top_terms": ", ".join(top_terms),
                "representative_feedback": evidence,
                "recommended_next_step": f"Review the {len(subset)} comments tagged '{label}', confirm the root cause with affected customers, then assign an owner and outcome metric.",
            }
        )
    summary = pd.DataFrame(themes).sort_values(["priority_score", "feedback_count"], ascending=False).reset_index(drop=True)
    return annotated, summary
