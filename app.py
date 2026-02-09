import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="AI Tourism Recommender", layout="wide")

# --------------------------------------------------
# PREMIUM DARK STYLE
# --------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color:#0e1117;
    color:#E6EDF3 !important;
}

h1, h2, h3, h4 {
    color:white !important;
}

.card {
    background:#161b22;
    padding:20px;
    border-radius:14px;
    margin-bottom:20px;
}

[data-testid="stSidebar"] {
    background:#11161c;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
# 🎬 AI Tourism Recommender
##### Personalized AI travel discovery powered by hybrid intelligence
""")

# --------------------------------------------------
# LOAD DATA + MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pickle.load(open("data.pkl","rb"))

@st.cache_resource
def load_model():
    return pickle.load(open("clf_model.pkl","rb"))

df = load_data()
clf = load_model()

# --------------------------------------------------
# LABEL MAP
# --------------------------------------------------

visit_mode_map = {
    0: "Business",
    1: "Couples",
    2: "Family",
    3: "Friends",
    4: "Solo"
}

# --------------------------------------------------
# SIDEBAR INPUT
# --------------------------------------------------

st.sidebar.header("User Input")

user_id = st.sidebar.selectbox(
    "Select User ID",
    df["UserId"].unique()
)

# --------------------------------------------------
# PREDICTION + RECOMMENDER
# --------------------------------------------------

features = [
    "VisitYear",
    "VisitMonth",
    "attraction_popularity",
    "attr_avg_rating",
    "user_total_visits",
    "user_avg_rating"
]

user_rows = df[df["UserId"] == user_id]

if user_rows.empty:

    st.warning("No data available for this user.")

else:

    user_data = user_rows[features].mean()

    prediction = clf.predict([user_data])[0]
    predicted_label = visit_mode_map.get(prediction, prediction)

    # Prediction card
    st.markdown(
        f"""
        <div class="card">
            <h3>🧠 Predicted Travel Style</h3>
            <h2 style="color:#FFD700;">{predicted_label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # HYBRID RECOMMENDER
    # --------------------------------------------------

    current_user_history = user_rows

    liked_attractions = current_user_history[
        current_user_history["Rating"] >= 4
    ]["AttractionId"].unique()

    similar_users = df[
        df["AttractionId"].isin(liked_attractions)
    ]["UserId"].unique()

    candidate_recommendations = df[
        (df["UserId"].isin(similar_users)) &
        (~df["AttractionId"].isin(current_user_history["AttractionId"]))
    ].copy()

    recommendations = candidate_recommendations.sort_values(
        by=["attr_avg_rating","attraction_popularity"],
        ascending=False
    )[["AttractionId","AttractionType","attr_avg_rating"]].drop_duplicates().head(10)

    # --------------------------------------------------
    # NETFLIX CAROUSEL UI
    # --------------------------------------------------
import streamlit.components.v1 as components

# --------------------------------------------------
# HERO SECTION (BIG AI PREDICTION BANNER)
# --------------------------------------------------

st.markdown(f"""
<div style="
background:linear-gradient(90deg,#161b22,#0e1117);
padding:30px;
border-radius:14px;
margin-bottom:25px;
">

<h1 style="color:white;">🧠 Your AI Travel Style: {predicted_label}</h1>
<p style="color:#c9d1d9;">
Our AI analyzed your behavior and predicts your ideal travel experience.
Explore recommendations curated just for you.
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NETFLIX CARD STYLE FUNCTION
# --------------------------------------------------

def build_carousel(title, dataframe):

    html = """
    <style>
    .carousel {display:flex;overflow-x:auto;gap:20px;padding:15px;}
    .card {
        position:relative;
        min-width:260px;
        height:360px;
        border-radius:14px;
        overflow:hidden;
        transition:0.4s;
    }
    .card:hover {transform:scale(1.08);}
    .card img {width:100%;height:100%;object-fit:cover;}
    .overlay {
        position:absolute;
        bottom:0;
        width:100%;
        padding:15px;
        background:linear-gradient(to top,rgba(0,0,0,0.9),transparent);
        color:white;
        font-family:sans-serif;
    }
    </style>
    """

    html += f"<h2 style='color:white'>{title}</h2>"
    html += "<div class='carousel'>"

    image_url = "https://picsum.photos/400/600?random="

    for i, (_, row) in enumerate(dataframe.iterrows()):
        html += f"""
        <div class="card">
            <img src="{image_url}{i}">
            <div class="overlay">
                <h4>Attraction {row['AttractionId']}</h4>
                <p>{row['AttractionType']}</p>
                ⭐ {row['attr_avg_rating']:.2f}
            </div>
        </div>
        """

    html += "</div>"

    components.html(html, height=450, scrolling=True)

# --------------------------------------------------
# ROW 1 — AI PICKS
# --------------------------------------------------

build_carousel("⭐ Because You Might Love These", recommendations)

# --------------------------------------------------
# ROW 2 — TRENDING
# --------------------------------------------------

trending = df.sort_values("attraction_popularity", ascending=False)[
    ["AttractionId","AttractionType","attr_avg_rating"]
].drop_duplicates().head(10)

build_carousel("🔥 Trending Now", trending)

# --------------------------------------------------
# ROW 3 — CATEGORY BASED
# --------------------------------------------------

nature = df[df["AttractionType"].str.contains("Park|Nature", na=False)].head(10)

build_carousel("🌍 Explore Nature & Adventure", nature)

# --------------------------------------------------
# USER HISTORY
# --------------------------------------------------

st.markdown("## 📊 Your Recent Activity")

history = current_user_history[
    ["AttractionId","Rating","AttractionType"]
].head(10)

st.dataframe(history, use_container_width=True)


# --------------------------------------------------
# NETFLIX CAROUSEL UI
# --------------------------------------------------

import streamlit.components.v1 as components

st.markdown("## ⭐ AI Picks For You")

carousel_html = """
<style>

.carousel {
    display:flex;
    overflow-x:auto;
    gap:20px;
    padding:20px;
    background:#0e1117;
}

.card {
    position:relative;
    min-width:280px;
    height:380px;
    border-radius:14px;
    overflow:hidden;
    cursor:pointer;
    transition:transform 0.4s ease;
}

.card:hover {
    transform:scale(1.08);
}

.card img {
    width:100%;
    height:100%;
    object-fit:cover;
}

.overlay {
    position:absolute;
    bottom:0;
    width:100%;
    padding:15px;
    background:linear-gradient(to top, rgba(0,0,0,0.9), transparent);
    color:white;
    font-family:sans-serif;
}

.badge {
    background:#FFD700;
    color:black;
    padding:4px 8px;
    border-radius:6px;
    font-size:12px;
    margin-top:5px;
    display:inline-block;
}

</style>

<div class="carousel">
"""

# placeholder netflix-style images
image_url = "https://picsum.photos/400/600?random="

for i, (_, row) in enumerate(recommendations.iterrows()):

    carousel_html += f"""
    <div class="card">
        <img src="{image_url}{i}">
        <div class="overlay">
            <h4>🎯 Attraction {row['AttractionId']}</h4>
            <p>{row['AttractionType']}</p>
            <span class="badge">⭐ {row['attr_avg_rating']:.2f}</span>
        </div>
    </div>
    """

carousel_html += "</div>"

components.html(carousel_html, height=450, scrolling=True)


    # --------------------------------------------------
    # USER HISTORY
    # --------------------------------------------------
st.markdown("## 📊User History")
history = current_user_history[
    ["AttractionId","Rating","AttractionType"]
].head(10)

st.dataframe(history, use_container_width=True)
