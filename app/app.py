import re
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Amazon Customer Voice Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    /* Global readable text */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown li,
    label {
        color: #172337 !important;
    }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] a {
        color: #f9fafb !important;
    }

    /* Fix visibility of Streamlit metric values */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #172337 !important;
    }


    .main-title {
        color: #172337; font-size: 38px; font-weight: 800;
        margin-bottom: 2px;
    }
    .subtitle { color: #667085; font-size: 16px; margin-bottom: 18px; }
    .section-title {
        color: #172337; font-size: 22px; font-weight: 800;
        margin-top: 14px; margin-bottom: 10px;
    }
    .metric-card {
        background: white; padding: 18px 14px; border-radius: 15px;
        border: 1px solid #e5e7eb; box-shadow: 0 3px 12px rgba(0,0,0,.06);
        text-align: center; min-height: 105px;
    }
    .metric-title { color: #667085; font-size: 13px; font-weight: 700; }
    .metric-value { color: #172337; font-size: 25px; font-weight: 850; margin-top: 7px; }
    .insight-card {
        background: white; padding: 17px; border-radius: 14px;
        border: 1px solid #e5e7eb; box-shadow: 0 3px 10px rgba(0,0,0,.05);
        margin-bottom: 10px;
    }
    .insight-title { color: #172337; font-size: 16px; font-weight: 800; }
    .insight-text { color: #667085; font-size: 13px; margin-top: 5px; }
    .status-pill {
        display: inline-block; padding: 5px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 800;
    }
    .footer-note { color:#98a2b3; font-size:12px; text-align:center; padding:20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA LOADING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Reviews_with_VADER.csv"


SUMMARY_PATH = BASE_DIR / "data" / "sentiment_summary.csv"
RATING_PATH = BASE_DIR / "data" / "rating_summary.csv"

@st.cache_data
def load_summary():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"sentiment_summary.csv was not found at: {SUMMARY_PATH}"
        )
    return pd.read_csv(SUMMARY_PATH)

@st.cache_data
def load_ratings():
    if not RATING_PATH.exists():
        raise FileNotFoundError(
            f"rating_summary.csv was not found at: {RATING_PATH}"
        )
    return pd.read_csv(RATING_PATH)

sentiment_summary = load_summary()
rating_summary = load_ratings()


df = pd.read_csv(
    BASE_DIR / "data" / "review_sample.csv"
)

# =========================================================
# DATA PREPARATION
# =========================================================
df["Score"] = pd.to_numeric(df.get("Score"), errors="coerce")
df = df.dropna(subset=["Score"]).copy()

df["Sentiment"] = df["Score"].apply(
    lambda x: "Positive" if x >= 4 else ("Neutral" if x == 3 else "Negative")
)

for col in ["Summary", "Text"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

if "HelpfulnessNumerator" in df.columns:
    df["HelpfulnessNumerator"] = pd.to_numeric(
        df["HelpfulnessNumerator"], errors="coerce"
    ).fillna(0)
else:
    df["HelpfulnessNumerator"] = 0

if "HelpfulnessDenominator" in df.columns:
    df["HelpfulnessDenominator"] = pd.to_numeric(
        df["HelpfulnessDenominator"], errors="coerce"
    ).fillna(0)
else:
    df["HelpfulnessDenominator"] = 0

df["Helpfulness_Ratio"] = (
    df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"].replace(0, pd.NA)
).fillna(0) * 100

df["Review_Length"] = (df["Text"] + " " + df["Summary"]).str.split().str.len()

# Optional columns from the user's NLP notebook are supported automatically.
has_vader = "VADER_Sentiment" in df.columns
if has_vader:
    df["VADER_Sentiment"] = df["VADER_Sentiment"].fillna("Unknown").astype(str)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🛒 Customer Voice")
    st.caption("Amazon Review Intelligence")
    st.markdown("---")

    page = st.radio(
        "📍 Navigate",
        [
            "🏠 Overview",
            "💬 Customer Insights",
            "⭐ Review Intelligence",
            "🤖 Review Analyzer",
        ],
    )

    st.markdown("---")
    st.markdown("### 🎯 Key Areas")
    st.markdown("🏆 Product Quality")
    st.markdown("😋 Taste & Flavor")
    st.markdown("📦 Purchase Experience")
    st.markdown("💰 Price & Value")
    st.markdown("🐾 Pet & Food")
    st.markdown("⚡ Convenience")
    st.markdown("---")
    st.caption("NLP • Sentiment • Helpfulness • Customer Voice")

# =========================================================
# COMMON METRICS
# =========================================================
total_reviews = len(df)
positive_pct = (df["Sentiment"].eq("Positive").mean() * 100)
neutral_pct = (df["Sentiment"].eq("Neutral").mean() * 100)
negative_pct = (df["Sentiment"].eq("Negative").mean() * 100)
average_rating = df["Score"].mean()
avg_helpfulness = df["Helpfulness_Ratio"].mean()
avg_length = df["Review_Length"].mean()


def metric_card(title, value, icon="📌"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{icon} {title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_kpis(include_helpfulness=False):
    values = [
        ("Total Reviews", f"{total_reviews:,}", "📝"),
        ("Positive", f"{positive_pct:.2f}%", "😊"),
        ("Neutral", f"{neutral_pct:.2f}%", "😐"),
        ("Negative", f"{negative_pct:.2f}%", "😞"),
        ("Avg Rating", f"{average_rating:.2f} / 5", "⭐"),
    ]
    if include_helpfulness:
        values[-1] = ("Avg Helpfulness", f"{avg_helpfulness:.2f}%", "👍")
    cols = st.columns(len(values))
    for col, (title, value, icon) in zip(cols, values):
        with col:
            metric_card(title, value, icon)


# =========================================================
# OVERVIEW
# =========================================================
if page == "🏠 Overview":
    st.markdown('<div class="main-title">🛒 Amazon Customer Voice Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">🧠 Turning customer reviews into actionable business insights</div>',
        unsafe_allow_html=True,
    )

    show_kpis()
    st.markdown("")

    left, right = st.columns(2)
    with left:
        
        st.markdown(
            '<div class="section-title">📊 Sentiment Overview</div>',
            unsafe_allow_html=True
        )

        sentiment_counts = (
            df["Sentiment"]
            .value_counts()
            .reindex(
                ["Negative", "Neutral", "Positive"],
                fill_value=0
            )
        )

        st.bar_chart(
            sentiment_counts,
            use_container_width=True
        )
    
    with right:
        st.markdown('<div class="section-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(df["Score"].value_counts().sort_index())

    st.markdown('<div class="section-title">🎯 Quick Customer Insights</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            f'<div class="insight-card"><div class="insight-title">😊 Strong Positive Feedback</div>'
            f'<div class="insight-text">{positive_pct:.2f}% of reviews are positive based on the rating-derived sentiment label.</div></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f'<div class="insight-card"><div class="insight-title">🚨 Customer Pain Signals</div>'
            f'<div class="insight-text">{negative_pct:.2f}% of reviews are negative and can be investigated for recurring issues.</div></div>',
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            f'<div class="insight-card"><div class="insight-title">⭐ Rating Snapshot</div>'
            f'<div class="insight-text">Average rating across the dataset is {average_rating:.2f} out of 5.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">🔎 Explore Customer Reviews</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        rating_filter = st.selectbox("⭐ Rating", ["All Ratings", "5 Stars", "4 Stars", "3 Stars", "2 Stars", "1 Star"])
    with f2:
        sentiment_filter = st.selectbox("💬 Sentiment", ["All Sentiments", "Positive", "Neutral", "Negative"])
    with f3:
        review_limit = st.selectbox("📝 Rows", [10, 20, 50, 100], index=0)

    search_text = st.text_input(
        "🔍 Search reviews",
        placeholder="Try: coffee, taste, delivery, price, packaging…",
    )

    filtered_df = df.copy()
    if rating_filter != "All Ratings":
        filtered_df = filtered_df[filtered_df["Score"] == int(rating_filter[0])]
    if sentiment_filter != "All Sentiments":
        filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
    if search_text.strip():
        q = search_text.strip()
        mask = (
            filtered_df["Summary"].str.contains(q, case=False, regex=False, na=False)
            | filtered_df["Text"].str.contains(q, case=False, regex=False, na=False)
        )
        filtered_df = filtered_df.loc[mask]

    st.caption(f"📌 {len(filtered_df):,} reviews match your filters")
    display_columns = [c for c in ["Id", "Score", "Summary", "Text", "Sentiment"] if c in filtered_df.columns]
    st.dataframe(filtered_df[display_columns].head(review_limit), use_container_width=True, hide_index=True, height=420)

    if len(filtered_df) > 0:
        st.download_button(
            "⬇️ Download filtered reviews",
            filtered_df[display_columns].to_csv(index=False).encode("utf-8"),
            file_name="amazon_filtered_reviews.csv",
            mime="text/csv",
        )

# =========================================================
# CUSTOMER INSIGHTS
# =========================================================
elif page == "💬 Customer Insights":
    st.markdown('<div class="main-title">💬 Customer Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🔎 Explore what customers love, dislike, and repeatedly mention</div>', unsafe_allow_html=True)

    theme_data = pd.DataFrame({
        "Theme": [
            "Product Quality", "Taste & Flavor", "Pet & Food Experience",
            "Price & Value", "Convenience", "Purchase Experience"
        ],
        "Positive": [68.88, 62.38, 22.81, 23.14, 21.09, 38.29],
        "Negative": [54.02, 57.97, 24.37, 25.44, 24.07, 48.11],
    })
    theme_data["Gap"] = theme_data["Positive"] - theme_data["Negative"]

    theme_icons = {
        "Product Quality": "🏆", "Taste & Flavor": "😋",
        "Pet & Food Experience": "🐾", "Price & Value": "💰",
        "Convenience": "⚡", "Purchase Experience": "📦",
    }
    theme_data["Display Theme"] = theme_data["Theme"].map(lambda x: f"{theme_icons[x]} {x}")

    strongest_theme = theme_data.loc[theme_data["Gap"].idxmax(), "Theme"]
    biggest_problem = theme_data.loc[theme_data["Gap"].idxmin(), "Theme"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Strongest Theme", strongest_theme, "🏆")
    with c2: metric_card("Largest Negative Gap", biggest_problem, "🚨")
    with c3: metric_card("Positive Themes", str((theme_data["Gap"] > 0).sum()), "🟢")
    with c4: metric_card("Risk Themes", str((theme_data["Gap"] < 0).sum()), "🔴")

    st.markdown('<div class="section-title">📊 Theme Sentiment Comparison</div>', unsafe_allow_html=True)
    st.bar_chart(theme_data.set_index("Display Theme")[["Positive", "Negative"]])
    st.caption("📌 Percentages represent the share of the positive/negative review samples mentioning each theme.")

    st.markdown('<div class="section-title">🔍 Theme Explorer</div>', unsafe_allow_html=True)
    selected_theme = st.selectbox(
        "Choose a theme",
        list(theme_icons.keys()),
        format_func=lambda x: f"{theme_icons[x]} {x}",
    )
    selected_row = theme_data.loc[theme_data["Theme"] == selected_theme].iloc[0]
    gap = float(selected_row["Gap"])

    x1, x2, x3 = st.columns(3)
    with x1: metric_card("Positive Mentions", f"{selected_row['Positive']:.2f}%", "🟢")
    with x2: metric_card("Negative Mentions", f"{selected_row['Negative']:.2f}%", "🔴")
    with x3: metric_card("Sentiment Gap", f"{gap:+.2f}%", "📊")

    if gap >= 10:
        st.success(f"🏆 {selected_theme}: positive mentions exceed negative mentions by {gap:.2f} percentage points.")
    elif gap > 0:
        st.info(f"👍 {selected_theme}: positive mentions exceed negative mentions by {gap:.2f} percentage points.")
    elif gap <= -5:
        st.error(f"🚨 {selected_theme}: negative mentions exceed positive mentions by {abs(gap):.2f} percentage points.")
    else:
        st.warning(f"⚠️ {selected_theme}: the sentiment gap is relatively balanced at {gap:+.2f} percentage points.")

    interpretations = {
        "Product Quality": "Customers frequently discuss product quality, with a higher positive share in the analyzed sample.",
        "Taste & Flavor": "Taste and flavor are major parts of the customer conversation and show a positive sentiment gap.",
        "Pet & Food Experience": "Pet and food experiences are relatively balanced and are worth examining at review level.",
        "Price & Value": "Price and value show slightly more negative than positive mentions, suggesting a useful area for deeper investigation.",
        "Convenience": "Convenience has more negative than positive mentions; packaging and ease-of-use language can be examined further.",
        "Purchase Experience": "Purchase experience has the largest negative sentiment gap in this theme analysis and includes ordering, delivery, and transaction language.",
    }
    st.markdown('<div class="section-title">💡 What This Means</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-card"><div class="insight-title">{theme_icons[selected_theme]} {selected_theme}</div>'
        f'<div class="insight-text">{interpretations[selected_theme]}</div></div>',
        unsafe_allow_html=True,
    )

    # ---- FIXED REVIEW SEARCH: all masks are created and applied to the same dataframe ----
    st.markdown('<div class="section-title">🔎 Find Reviews Related to This Theme</div>', unsafe_allow_html=True)
    theme_keywords = {
        "Product Quality": ["quality", "product", "good", "great", "best", "perfect"],
        "Taste & Flavor": ["taste", "flavor", "delicious", "coffee", "tea", "snack"],
        "Price & Value": ["price", "money", "cost", "value", "expensive", "cheap"],
        "Purchase Experience": ["bought", "buy", "amazon", "order", "ordered", "delivery"],
        "Pet & Food Experience": ["food", "dog", "cat", "cats", "pet", "pets"],
        "Convenience": ["easy", "quick", "box", "packaging", "package"],
    }

    pattern = "|".join(re.escape(k) for k in theme_keywords[selected_theme])
    related_reviews = df.copy()

    summary_mask = related_reviews["Summary"].str.contains(pattern, case=False, regex=True, na=False)
    text_mask = related_reviews["Text"].str.contains(pattern, case=False, regex=True, na=False)
    related_reviews = related_reviews.loc[summary_mask | text_mask].copy()

    st.caption(f"📌 Found {len(related_reviews):,} reviews mentioning terms related to {theme_icons[selected_theme]} {selected_theme}.")

    review_sentiment = st.selectbox(
        "💬 Filter these reviews by sentiment",
        ["All", "Positive", "Neutral", "Negative"],
        key="theme_sentiment_fixed",
    )
    if review_sentiment != "All":
        related_reviews = related_reviews.loc[related_reviews["Sentiment"] == review_sentiment].copy()

    review_columns = [c for c in ["Id", "Score", "Summary", "Text", "Sentiment"] if c in related_reviews.columns]
    st.dataframe(related_reviews[review_columns].head(20), use_container_width=True, hide_index=True, height=430)

# =========================================================
# REVIEW INTELLIGENCE
# =========================================================
elif page == "⭐ Review Intelligence":
    st.markdown('<div class="main-title">⭐ Review Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🧠 Identify influential feedback, engagement signals, duplicates, and sentiment conflicts</div>', unsafe_allow_html=True)

    show_kpis(include_helpfulness=True)

    st.markdown('<div class="section-title">👍 Helpfulness Intelligence</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1: metric_card("Avg Review Length", f"{avg_length:.1f} words", "📝")
    with h2: metric_card("Avg Helpfulness Ratio", f"{avg_helpfulness:.2f}%", "👍")
    corr = df["Review_Length"].corr(df["HelpfulnessNumerator"]) if df["HelpfulnessNumerator"].nunique() > 1 else 0
    with h3: metric_card("Length ↔ Helpful Votes", f"{corr:.3f}", "🔗")

    engagement_threshold = st.slider("🎯 High-engagement threshold (helpfulness votes)", 1, 100, 10)
    engaged = df.loc[df["HelpfulnessDenominator"] >= engagement_threshold].copy()
    st.caption(f"📌 {len(engaged):,} reviews have at least {engagement_threshold} helpfulness votes recorded in the denominator.")

    if len(engaged):
        engaged_summary = engaged.groupby("Sentiment")["Helpfulness_Ratio"].mean().reindex(
            ["Positive", "Neutral", "Negative"]
        )
        st.markdown("**Average helpfulness ratio by sentiment**")
        st.bar_chart(engaged_summary)

        top_engaged = engaged.sort_values(
            ["HelpfulnessDenominator", "Helpfulness_Ratio"], ascending=False
        ).head(20)
        cols = [c for c in ["Id", "Score", "Summary", "HelpfulnessNumerator", "HelpfulnessDenominator", "Helpfulness_Ratio", "Sentiment"] if c in top_engaged.columns]
        st.dataframe(top_engaged[cols], use_container_width=True, hide_index=True, height=400)

    st.markdown('<div class="section-title">♻️ Duplicate Review Signals</div>', unsafe_allow_html=True)
    # Normalize only when this page is opened, so the main dashboard stays fast.
    normalized = (
        df["Summary"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        + " | "
        + df["Text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    duplicate_mask = normalized.duplicated(keep=False) & normalized.ne(" | ")
    duplicate_count = int(duplicate_mask.sum())
    unique_duplicate_texts = int(normalized[duplicate_mask].nunique())
    d1, d2 = st.columns(2)
    with d1: metric_card("Duplicate Records", f"{duplicate_count:,}", "♻️")
    with d2: metric_card("Unique Duplicated Texts", f"{unique_duplicate_texts:,}", "🔁")

    if duplicate_count:
        dup_view = df.loc[duplicate_mask, [c for c in ["Id", "Score", "Summary", "Sentiment"] if c in df.columns]].copy()
        st.dataframe(dup_view.head(25), use_container_width=True, hide_index=True, height=300)

    st.markdown('<div class="section-title">⚡ Rating vs Text Sentiment Conflicts</div>', unsafe_allow_html=True)
    if has_vader:
        conflict = df.loc[
            df["VADER_Sentiment"].ne(df["Sentiment"])
            & df["VADER_Sentiment"].isin(["Positive", "Neutral", "Negative"])
        ].copy()
        st.caption(f"📌 {len(conflict):,} reviews have different rating-derived and VADER sentiment labels.")
        conflict_table = conflict[[c for c in ["Id", "Score", "Sentiment", "VADER_Sentiment", "Summary", "Text"] if c in conflict.columns]]
        st.dataframe(conflict_table.head(25), use_container_width=True, hide_index=True, height=330)
    else:
        st.info("ℹ️ VADER_Sentiment is not present in Reviews.csv. If you export the NLP-enriched dataset from your notebook with that column, this section will automatically activate.")


# ============================================================
# REVIEW ANALYZER
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Individual Review Analyzer</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter a customer review to detect sentiment signals, customer themes, "
    "and possible business concerns."
)

# Sentiment vocabulary
positive_words = {
    "good", "great", "excellent", "amazing", "love", "loved",
    "best", "perfect", "nice", "awesome", "delicious", "tasty",
    "fresh", "happy", "easy", "fast", "quick", "recommend",
    "recommended", "wonderful", "fantastic", "quality", "worth"
}

negative_words = {
    "bad", "terrible", "awful", "hate", "hated", "poor", "worst",
    "broken", "late", "damaged", "disappointed", "disappointing",
    "expensive", "refund", "waste", "worse", "problem", "problems",
    "return", "returned", "slow", "stale", "missing", "wrong"
}

negative_phrases = {
    "not on time",
    "late delivery",
    "poor delivery",
    "bad delivery",
    "not delivered",
    "poor packaging",
    "bad packaging",
    "not proper packing",
    "not properly packed",
    "poorly packed",
    "damaged package",
    "damaged packaging",
    "wrong item",
    "wrong product",
    "poor quality",
    "bad quality",
    "not fresh",
    "too expensive",
    "not worth",
    "waste of money"
}

analyzer_themes = {
    "😋 Taste & Flavor": [
        "taste", "flavor", "coffee", "tea", "delicious", "sweet", "salty"
    ],
    "🏆 Product Quality": [
        "quality", "product", "broken", "fresh", "damaged", "works"
    ],
    "💰 Price & Value": [
        "price", "cost", "money", "expensive", "cheap", "value", "worth"
    ],
    "📦 Purchase Experience": [
        "delivery", "delivered", "order", "ordered", "shipping",
        "package", "packaging", "amazon"
    ],
    "🐾 Pet & Food Experience": [
        "dog", "cat", "pet", "food", "treat", "puppy", "kitten"
    ],
    "⚡ Convenience": [
        "easy", "quick", "fast", "simple", "box", "package"
    ],
}

review_text = st.text_area(
    "📝 Enter a customer review",
    placeholder=(
        "Example: The product is great, but the delivery was late "
        "and the packaging was poor."
    ),
    height=130
)

analyze = st.button(
    "🔍 Analyze Review",
    use_container_width=True
)

if analyze:

    text = review_text.strip()

    if not text:
        st.warning("✍️ Please enter a review first.")

    else:

        text_lower = text.lower()
        tokens = re.findall(r"[a-zA-Z']+", text_lower)

        # Count positive and negative signals
        pos = sum(token in positive_words for token in tokens)
        neg = sum(token in negative_words for token in tokens)

        # Detect negative phrases
        for phrase in negative_phrases:
            if phrase in text_lower:
                neg += 1

        # Determine sentiment
                # Determine sentiment
        if pos > 0 and neg > 0:
            # Both positive and negative signals are present
            if abs(pos - neg) <= 1:
                label = "Mixed"
                icon = "🟡"
            elif pos > neg:
                label = "Positive"
                icon = "😊"
            else:
                label = "Negative"
                icon = "😞"

        elif pos > 0:
            label = "Positive"
            icon = "😊"

        elif neg > 0:
            label = "Negative"
            icon = "😞"

        else:
            label = "Neutral"
            icon = "😐"

        # Detect themes
        theme_hits = {}

        for theme, words in analyzer_themes.items():

            hits = sum(
                word in tokens
                for word in words
            )

            if hits > 0:
                theme_hits[theme] = hits

        # ----------------------------------------------------
        # Result cards
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{icon} Predicted Sentiment</div>
                    <div class="metric-value">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">👍 Positive Signals</div>
                    <div class="metric-value">{pos}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">⚠️ Negative Signals</div>
                    <div class="metric-value">{neg}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Customer themes
        # ----------------------------------------------------

        st.markdown("### 🔎 Detected Customer Themes")

        if theme_hits:

            theme_cols = st.columns(min(len(theme_hits), 3))

            for i, (theme, count) in enumerate(theme_hits.items()):

                with theme_cols[i % len(theme_cols)]:
                    st.info(
                        f"**{theme}**\n\n"
                        f"{count} signal(s) detected"
                    )

        else:
            st.info("No specific customer theme detected.")

        # ----------------------------------------------------
        # Business interpretation
        # ----------------------------------------------------

        st.markdown("### 💼 Business Interpretation")

        if label == "Positive":
            st.success(
                "Customers are showing mostly positive signals. "
                "This type of feedback can help identify product strengths."
            )

        elif label == "Negative":
            st.error(
                "Customers are showing negative signals. "
                "This feedback may indicate areas that need attention."
            )

        else:
            st.warning(
                "This review contains mixed or balanced signals. "
                "It may be useful to inspect the specific positive and negative points."
            )