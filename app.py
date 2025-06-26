import streamlit as st
import requests
from transformers import pipeline
import plotly.express as px

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

# Convert stars to custom label
def star_to_sentiment(label):
    stars = int(label[0])
    if stars >= 4:
        return "POSITIVE"
    elif stars == 3:
        return "AVERAGE"
    else:
        return "NEGATIVE"

# UI Input
st.markdown("### 🏢 Enter Business Info")
business = st.text_input("Business Name (e.g., Jahangirnagar University)")
location = st.text_input("Location (e.g., Dhaka, Bangladesh)")

SERP_API_KEY = "15aae7e05c594f10ec289cdbbf03a6e116934c8c542e36634c08f6782cb6c56b"  # Replace with your own if not working

# Fetch reviews using SerpAPI
def fetch_reviews(business, location):
    search_url = "https://serpapi.com/search.json"
    search_params = {
        "engine": "google_maps",
        "q": f"{business} {location}",
        "type": "search",
        "api_key": SERP_API_KEY
    }
    search_res = requests.get(search_url, params=search_params).json()
    try:
        data_id = search_res['local_results'][0]['data_id']
    except:
        return [], "❌ Business not found or SerpAPI error."

    review_params = {
        "engine": "google_maps_reviews",
        "data_id": data_id,
        "api_key": SERP_API_KEY
    }
    review_res = requests.get(search_url, params=review_params).json()
    return review_res.get("reviews", []), None

# Analyze sentiments
def analyze_reviews(reviews):
    results = []
    counts = {"POSITIVE": 0, "AVERAGE": 0, "NEGATIVE": 0}

    for r in reviews:
        text = r.get("snippet", "")
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

# Main logic
if st.button("🔍 Analyze Reviews"):
    if business and location:
        with st.spinner("Fetching reviews..."):
            reviews, error = fetch_reviews(business, location)

        if error:
            st.error(error)
        elif not reviews:
            st.warning("⚠️ No reviews found.")
        else:
            with st.spinner("Analyzing sentiments..."):
                results, counts = analyze_reviews(reviews)

            st.success(f"✅ {len(results)} reviews analyzed.")

            # Show counts safely
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

            st.markdown("### 📋 Sample Reviews")
            for r in results[:10]:
                label = (
                    "🟢 Positive" if r["label"] == "POSITIVE" else
                    "🟡 Average" if r["label"] == "AVERAGE" else
                    "🔴 Negative"
                )
                st.markdown(f"> **{label}** ({r['stars']} | {r['score']:.2f}): {r['text']}")
    else:
        st.warning("Please enter both Company Name and Location Accurately.")

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; font-size: 13px; color: gray;'>
        Developed by Anupam Roy, A Student of IIT, JU
    </p>
""", unsafe_allow_html=True)
