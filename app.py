import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="AI Tourism Recommender", layout="wide")

st.markdown("""
<style>

body {
    background-color:#0e1117;
}

/* MAIN TEXT COLOR FIX */
html, body, [class*="css"] {
    color:#E6EDF3 !important;
}

/* HEADERS */
h1, h2, h3, h4 {
    color:#ffffff !important;
}

/* SUBTEXT */
p, span, label {
    color:#c9d1d9 !important;
}

/* CARD STYLE */
.card {
    background-color:#161b22;
    padding:20px;
    border-radius:14px;
    margin-bottom:20px;
    transition:0.3s;
}

.card:hover {
    transform:scale(1.03);
    box-shadow:0 0 15px rgba(255,255,255,0.1);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color:#11161c;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PREMIUM DARK THEME STYLE
# --------------------------------------------------

st.markdown("""
<style>

body {
    background-color:#0e1117;
}

[data-testid="stAppViewContainer"] {
    background-color:#0e1117;
}

.card {
    background-color:#161b22;
    padding:20px;
    border-radius:14px;
    margin-bottom:20px;
    transition:0.3s;
}

.card:hover {
    transform:scale(1.03);
    box-shadow:0 0 15px rgba(255,255,255,0.1);
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
# SIDEBAR USER INPUT
# --------------------------------------------------

st.sidebar.header("User Input")

user_id = st.sidebar.selectbox(
    "Select User ID",
    df["UserId"].unique()
)

# --------------------------------------------------
# PREDICTION
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

    # Premium prediction badge
    st.markdown(
        f"""
        <div class="card">
            <h3>Predicted Travel Style</h3>
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

    mode_preferences = {
        "Couples": ["Beaches", "Parks"],
        "Family": ["Museums", "Theme Parks"],
        "Friends": ["Adventure", "Nightlife"],
        "Business": ["Museums"],
        "Solo": ["Historic", "Nature"]
    }

    preferred_types = mode_preferences.get(predicted_label, [])

    candidate_recommendations["boost"] = (
        candidate_recommendations["AttractionType"]
        .isin(preferred_types)
        .astype(int)
    )

    recommendations = candidate_recommendations.sort_values(
        by=["boost", "attr_avg_rating", "attraction_popularity"],
        ascending=False
    )[["AttractionId","AttractionType","attr_avg_rating"]].drop_duplicates().head(6)

    # --------------------------------------------------
    # NETFLIX STYLE UI
    # --------------------------------------------------
st.markdown("##AI Picks For You")

carousel_html = """
<div style="
display:flex;
overflow-x:auto;
gap:20px;
padding:10px;
scroll-behavior:smooth;
">
"""

for _, row in recommendations.iterrows():

    card = f"""
    <div style="
        min-width:260px;
        background:#161b22;
        border-radius:14px;
        padding:20px;
        flex-shrink:0;
        transition:0.3s;
        box-shadow:0 0 10px rgba(0,0,0,0.4);
    ">

        <h4 style="color:white;">Attraction {row['AttractionId']}</h4>
        <p style="color:#c9d1d9;">Category: {row['AttractionType']}</p>
        <p style="color:#FFD700;">Rating: {row['attr_avg_rating']}</p>

        <small style="color:#58a6ff;">
        Recommended because similar users liked this
        </small>

    </div>
    """

    carousel_html += card

carousel_html += "</div>"

st.markdown(carousel_html, unsafe_allow_html=True)

    
    # --------------------------------------------------
    # USER HISTORY
    # --------------------------------------------------

     st.markdown("##User History")

    history = current_user_history[
        ["AttractionId","Rating","AttractionType"]
    ].head(10)

    st.dataframe(history, use_container_width=True)
