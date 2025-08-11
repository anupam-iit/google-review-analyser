import streamlit as st
import requests
from transformers import pipeline
import plotly.express as px

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray */
        color: black; /* Ensure text stays black */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Search box style */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 6px;
        padding: 8px;
    }

    /* Button style */
    .stButton > button {
        background-color: #4CAF50; /* Green button */
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }

    /* Button hover effect */
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Change background for st.success messages */
    .stAlert.success {
        background-color: #e6f7ff; /* Light blue background */
        color: black;              /* Black text */
        border: 1px solid #91d5ff; /* Soft blue border */
        border-radius: 6px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Google Review Sentiment Analyzer", page_icon="🌍")

st.markdown("""
    <h2 style='text-align: center;'>🌍 Google Review Sentiment Analyzer</h2>
""", unsafe_allow_html=True)

# Load sentiment model
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model=model_name)

analyzer = load_model()

# Convert stars to label
def star_to_sentiment(label):
    stars = int(label[0])
    if stars >= 4:
        return "POSITIVE"
    elif stars == 3:
        return "AVERAGE"
    else:
        return "NEGATIVE"

# ✅ Your Google Places API Key

GOOGLE_API_KEY = st.secrets["google_api_key"]

# Get Place ID by text search
def get_place_id(business_name):
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": business_name,
        "key": GOOGLE_API_KEY
    }
    res = requests.get(url, params=params).json()
    results = res.get("results", [])
    if not results:
        return None, "❌ Business not found. Try refining the name."
    return results[0]["place_id"], None

# Get reviews using Place ID
def get_reviews_by_place_id(place_id):
    url = f"https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,review,user_ratings_total",
        "key": GOOGLE_API_KEY
    }
    res = requests.get(url, params=params).json()
    result = res.get("result", {})
    reviews = result.get("reviews", [])
    return reviews, result.get("name", "Unknown")

# Analyze sentiments
def analyze_reviews(reviews):
    results = []
    counts = {"POSITIVE": 0, "AVERAGE": 0, "NEGATIVE": 0}
    for r in reviews:
        text = r.get("text", "")
        if not text.strip():
            continue
        prediction = analyzer(text)[0]
        label_raw = prediction['label']
        label = star_to_sentiment(label_raw)
        score = prediction['score']
        results.append({
            "text": text,
            "label": label,
            "stars": label_raw,
            "score": score
        })
        if label in counts:
            counts[label] += 1
    return results, counts

# 🔍 UI Input
st.markdown("### 🏢 Enter Business Name")
business_name = st.text_input("Business Name (e.g., Jahangirnagar University)")

# 🔍 Start search
if st.button("🔍 Analyze Reviews"):
    if not business_name:
        st.warning("Please enter a business name.")
    else:
        with st.spinner("🔍 Searching on Google Maps..."):
            place_id, error = get_place_id(business_name)
            if error:
                st.error(error)
            else:
                reviews, place_name = get_reviews_by_place_id(place_id)
                if not reviews:
                    st.warning("⚠️ No reviews found for this business.")
                else:
                    with st.spinner("⚙️ Analyzing sentiments..."):
                        results, counts = analyze_reviews(reviews)

                    st.success(f"✅ {len(results)} reviews analyzed for: **{place_name}**")

                    # Display sentiment counts
                    st.markdown(f"""
                        - 😊 Positive: {counts.get('POSITIVE', 0)}
                        - 😐 Average: {counts.get('AVERAGE', 0)}
                        - 😠 Negative: {counts.get('NEGATIVE', 0)}
                    """)

                    # Pie chart
                    labels = ['Positive', 'Average', 'Negative']
                    values = [counts.get('POSITIVE', 0), counts.get('AVERAGE', 0), counts.get('NEGATIVE', 0)]
                    fig = px.pie(
                        names=labels,
                        values=values,
                        color=labels,
                        color_discrete_map={'Positive': 'green', 'Average': 'gray', 'Negative': 'red'},
                        title="Sentiment Distribution"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    # Sample Reviews
                    st.markdown("### 📋 Sample Reviews")
                    for r in results[:10]:
                        label = (
                            "🟢 Positive" if r["label"] == "POSITIVE" else
                            "🟡 Average" if r["label"] == "AVERAGE" else
                            "🔴 Negative"
                        )
                        st.markdown(f"> **{label}** ({r['stars']} | {r['score']:.2f}): {r['text']}")

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; font-size: 13px; color: gray;'>
        Developed by Anupam Roy, A Student of IIT, JU
    </p>
    <p style='text-align: center; font-size: 13px; color: gray;'>
        anu.rex@gmail.com || 01723928176
    </p>
""", unsafe_allow_html=True)
