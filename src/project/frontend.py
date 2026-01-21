"""Streamlit frontend for financial sentiment analysis inference."""

import streamlit as st
import requests

st.set_page_config(page_title="Financial Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
    }
    .sentiment-positive {
        color: #10B981;
        font-weight: bold;
        font-size: 2rem;
    }
    .sentiment-negative {
        color: #EF4444;
        font-weight: bold;
        font-size: 2rem;
    }
    .sentiment-neutral {
        color: #6B7280;
        font-weight: bold;
        font-size: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("📊 Financial Sentiment Analyzer")
st.markdown("Analyze the sentiment of financial news articles using AI")

API_BASE_URL = "http://localhost:8000"

# Input Section
st.header("Input Article")
col1, col2 = st.columns([4, 1])

with col1:
    article_url = st.text_input(
        "Enter article URL",
        placeholder="https://example.com/financial-news",
        label_visibility="collapsed",
    )

with col2:
    analyze_button = st.button("🔍 Analyze", use_container_width=True)

# Analysis Section
if analyze_button:
    if not article_url:
        st.error("Please enter a URL")
    elif not article_url.startswith(("http://", "https://")):
        st.error("Please enter a valid URL (starting with http:// or https://)")
    else:
        with st.spinner("🔄 Fetching article and analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={"url": article_url},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display Results
                    st.success("✅ Analysis Complete!")

                    # Overall Sentiment - Large Display
                    col1, col2, col3 = st.columns(3)

                    sentiment = result.get("overall_sentiment", "neutral").upper()
                    distribution = result.get("sentiment_distribution", {})
                    sentences_count = result.get("sentences_analyzed", 0)

                    with col1:
                        st.metric("Sentences Analyzed", sentences_count)

                    with col2:
                        # Color code based on sentiment
                        if sentiment == "POSITIVE":
                            st.markdown('<div class="sentiment-positive">😊 POSITIVE</div>', unsafe_allow_html=True)
                        elif sentiment == "NEGATIVE":
                            st.markdown('<div class="sentiment-negative">😞 NEGATIVE</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="sentiment-neutral">😐 NEUTRAL</div>', unsafe_allow_html=True)

                    with col3:
                        st.metric("Primary Sentiment", sentiment)

                    # Sentiment Distribution Bar Chart
                    st.subheader("Sentiment Distribution")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = distribution.get("positive", 0)
                        st.metric("Positive 😊", positive_count)

                    with col2:
                        neutral_count = distribution.get("neutral", 0)
                        st.metric("Neutral 😐", neutral_count)

                    with col3:
                        negative_count = distribution.get("negative", 0)
                        st.metric("Negative 😞", negative_count)

                    # Distribution Chart
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 4))
                    sentiments = ["Positive", "Neutral", "Negative"]
                    counts = [
                        distribution.get("positive", 0),
                        distribution.get("neutral", 0),
                        distribution.get("negative", 0),
                    ]
                    colors = ["#10B981", "#6B7280", "#EF4444"]
                    ax.bar(sentiments, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
                    ax.set_ylabel("Count", fontsize=12)
                    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
                    ax.grid(axis="y", alpha=0.3)
                    st.pyplot(fig, use_container_width=True)

                    # Detailed Predictions
                    st.subheader("Sentence-Level Analysis")

                    predictions = result.get("predictions", [])

                    if predictions:
                        for i, pred in enumerate(predictions[:10], 1):  # Show top 10
                            with st.expander(f"Sentence {i}: {pred['sentiment'].upper()}", expanded=False):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.write(f"**Text:** {pred['text']}")

                                with col2:
                                    sentiment_label = pred["sentiment"].upper()
                                    if sentiment_label == "POSITIVE":
                                        st.markdown(
                                            f'<span style="color: #10B981; font-weight: bold;">{sentiment_label}</span>',
                                            unsafe_allow_html=True,
                                        )
                                    elif sentiment_label == "NEGATIVE":
                                        st.markdown(
                                            f'<span style="color: #EF4444; font-weight: bold;">{sentiment_label}</span>',
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.markdown(
                                            f'<span style="color: #6B7280; font-weight: bold;">{sentiment_label}</span>',
                                            unsafe_allow_html=True,
                                        )

                                # Confidence and probabilities
                                confidence = pred.get("confidence", 0)
                                st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                                probs = pred.get("probabilities", {})
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.caption(f"Positive: {probs.get('positive', 0):.2%}")
                                with col2:
                                    st.caption(f"Neutral: {probs.get('neutral', 0):.2%}")
                                with col3:
                                    st.caption(f"Negative: {probs.get('negative', 0):.2%}")

                        if len(predictions) > 10:
                            st.info(f"📌 Showing 10 out of {len(predictions)} sentences")

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The article might be too large or the API is slow.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the backend is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Financial Sentiment Analyzer** uses machine learning to detect sentiment
        in financial news articles.

        ### How it works:
        1. Enter a financial news article URL
        2. System extracts relevant sentences
        3. AI predicts sentiment for each sentence
        4. Results are aggregated and displayed

        ### Sentiment Labels:
        - **Positive** 😊: Bullish/optimistic language
        - **Neutral** 😐: Factual/balanced language
        - **Negative** 😞: Bearish/pessimistic language
        """
    )

    st.divider()

    st.header("🔧 Configuration")
    st.caption("API Status")
    if st.button("Check API Health"):
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is online")
        else:
            st.error("⚠️ API returned error")

    st.caption(f"API URL: {API_BASE_URL}")
