from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import FeedbackValidationError, analyze_feedback


st.set_page_config(page_title="Feedback2Strategy", page_icon="🧭", layout="wide")
st.title("Feedback2Strategy")
st.caption("Find recurring themes, inspect the customer evidence, and build a prioritized investigation queue.")

uploaded = st.file_uploader("Upload customer feedback CSV", type=["csv"])
try:
    source = pd.read_csv(uploaded) if uploaded else pd.read_csv(Path(__file__).parent / "data" / "sample_feedback.csv")
except pd.errors.ParserError as exc:
    st.error(f"Could not parse the CSV. Check that every feedback field containing commas is quoted. Details: {exc}")
    st.stop()
if not uploaded:
    st.info("Showing fictional synthetic example feedback.")

theme_count = st.sidebar.slider("Requested themes", min_value=2, max_value=10, value=4)
try:
    annotated, themes = analyze_feedback(source, theme_count)
except (FeedbackValidationError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

left, middle, right = st.columns(3)
left.metric("Feedback records", f"{len(annotated):,}")
middle.metric("Negative tone baseline", f"{annotated['sentiment'].eq('Negative').mean():.0%}")
right.metric("Themes found", len(themes))

st.subheader("Investigation queue")
st.caption("Priority = negative comment count × mean absolute VADER compound score. This is a triage heuristic, not a business outcome forecast.")
st.dataframe(themes[["theme", "feedback_count", "negative_count", "negative_share", "priority_score", "top_terms", "representative_feedback", "recommended_next_step"]], width="stretch", hide_index=True)

fig = px.bar(themes.sort_values("priority_score"), x="priority_score", y="theme", orientation="h", color="negative_share", color_continuous_scale="Reds", labels={"priority_score": "Triage score", "theme": "Exploratory theme", "negative_share": "Negative share"}, title="Themes ranked by negative-volume and severity heuristic")
st.plotly_chart(fig, width="stretch")

with st.expander("Browse annotated feedback"):
    st.dataframe(annotated[["feedback_id", "segment", "theme_id", "sentiment", "sentiment_compound", "text"]], width="stretch", hide_index=True)
    st.download_button("Download annotated feedback", annotated.to_csv(index=False).encode("utf-8"), "feedback_analysis.csv", "text/csv")

st.warning("Clusters are exploratory and VADER is an English-language baseline. Read the examples and validate themes with customers before making product or commercial decisions.")
