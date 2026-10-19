import pandas as pd
import pytest

from analysis import FeedbackValidationError, analyze_feedback, validate_feedback


def sample():
    return pd.DataFrame({"feedback_id": ["a", "b", "c", "d"], "text": [
        "The checkout payment page is slow and confusing.",
        "Payment checkout keeps failing and support is unhelpful.",
        "I love the clear dashboard and fast reports.",
        "Great reports, the dashboard is easy to use.",
    ]})


def test_validate_adds_stable_ids_and_segment():
    result = validate_feedback(pd.DataFrame({"text": ["Good service"]}))
    assert result.loc[0, "feedback_id"] == "FB-00001"
    assert result.loc[0, "segment"] == "Unspecified"


def test_rejects_blank_text():
    with pytest.raises(FeedbackValidationError, match="cannot be blank"):
        validate_feedback(pd.DataFrame({"text": [""]}))


def test_rejects_duplicate_ids():
    with pytest.raises(FeedbackValidationError, match="unique"):
        validate_feedback(pd.DataFrame({"feedback_id": ["x", "x"], "text": ["One", "Two"]}))


def test_analysis_is_reproducible_and_evidence_linked():
    annotated1, themes1 = analyze_feedback(sample(), requested_themes=2)
    annotated2, themes2 = analyze_feedback(sample(), requested_themes=2)
    assert annotated1["theme_id"].tolist() == annotated2["theme_id"].tolist()
    assert themes1["theme"].tolist() == themes2["theme"].tolist()
    assert themes1["representative_feedback"].str.len().gt(0).all()
    assert (themes1["priority_score"] >= 0).all()
