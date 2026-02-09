import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="AI Tourist App", layout="wide")
st.markdown("""
<style>

body {
    background-color:#0e1117;
}

[data-testid="stAppViewContainer"] {
    background-color:#0e1117;
}

h1, h2, h3, h4 {
    color:white;
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


st.title("AI Tourism Recommendation System")

st.markdown("""
# 🎬 AI Tourism Recommender
##### Personalized AI travel discovery powered by hybrid intelligence
""")

def load_data():
  df = pickle.load(open("data.pkl","rb"))
  return df

def load_model():
  clf = pickle.load(open("clf_model.pkl","rb"))
  return clf

df = load_data()
clf = load_model()

visit_mode_map = {
  0: "Business",
  1: "Couples",
  2: "Family",
  3: "Friends",
  4: "Solo"
}

st.sidebar.header("User Input")

user_id = st.sidebar.selectbox(
  "Select User ID",
  df["UserId"].unique()
)
st.write("Selected User:", user_id)

st.subheader("Predicted Visit Mode")
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
  st.warning("No data found for this user.") 
else:
  user_data = user_rows[features].mean()

st.write("Features values:", user_data)
prediction = clf.predict([user_data])[0]
st.markdown(
    f"""
    <div class="card">
        <h3>🧠 Predicted Travel Style</h3>
        <h2 style="color:#FFD700;">{visit_mode_map.get(prediction, prediction)}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## ⭐ AI Picks For You")

cols = st.columns(3)

for i, (_, row) in enumerate(recommendations.iterrows()):

    with cols[i % 3]:

        st.markdown(
            f"""
            <div class="card">
                <h4>🎯 Attraction {row['AttractionId']}</h4>
                <p>Category: {row['AttractionType']}</p>
                <p>⭐ Rating: {row['attr_avg_rating']}</p>
                <small style="color:#58a6ff;">
                Recommended because users with similar behaviour liked this.
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )


recommendations = candidate_recommendations.sort_values(
by=["boost", "attr_avg_rating", "attraction_popularity"],
    ascending=False
)[
    ["AttractionId", "AttractionType", "attr_avg_rating"]
].drop_duplicates().head(5)

st.markdown("###AI Picks For You")

cols = st.columns(3)

for i, (_, row) in enumerate(recommendations.iterrows()):
  with cols[i % 3]:
     st.markdown(
            f"""
            <div style="
                background-color:#111;
                padding:15px;
                border-radius:12px;
                margin-bottom:10px;
                box-shadow:0px 0px 10px rgba(0,0,0,0.3);
            ">
                <h4 style="color:white;">Attraction {row['AttractionId']}</h4>
                <p style="color:#aaa;">Type: {row['AttractionType']}</p>
                <p style="color:#FFD700;">⭐ Rating: {row['attr_avg_rating']}</p>
                <p style="color:#0f9d58;">
                    Recommended because users similar to you liked this.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    

st.info(
    f"Hybrid recommendations based on similar users + predicted travel style: **{predicted_label}**"
)

st.subheader("AI Recommend Attractions")
predicted_label = visit_mode_map.get(prediction, prediction)
mode_preferences = {
  "Couples": ["Beaches", "Parks", "Romantic"],
  "Family": ["Theme Parks", "Zoos", "Museums"],
  "Friends": ["Nightlife", "Adventure", "Sports"],
  "Busines": ["Museums", "City Tours"],
  "Solo": ["Historic", "Nature", "Temples"]
}
preferred_types = mode_preferences.get(predicted_label, [])

if preferred_types:
  recommendations = df[
  df["AttractionType"].isin(preferred_types)
  ].sort_values(
    by=["attr_avg_rating", "attraction_popularity"],
    ascending=False
  )
else:
  recommendations = df.sort_values(
    by=["attr_avg_rating", "attraction_popularity"],
    ascending=False
  )

recommendations = recommendations[
["AttractionId", "AttractionType", "attr_avg_rating"]
].drop_duplicates().head(5)

st.dataframe(recommendations, use_container_width=True)

st.info(f"Recommendations generated based on predicted travel style: **{predicted_label}**")

st.subheader("User History")

user_history = df[df["UserId"] == user_id][
["AttractionId", "Rating", "AttractionType"]
]

st.subheader("Recommended Attractions")
recommendations = (
  df.sort_values(by=["attr_avg_rating", "attraction_popularity"], ascending=False)
  [["AttractionId", "AttractionType", "attr_avg_rating"]]
  .drop_duplicates()
  .head(5)
)
st.dataframe(recommendations, use_container_width=True)
st.dataframe(user_history.head(10), use_container_width=True)
