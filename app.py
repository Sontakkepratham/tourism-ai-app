import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="AI Tourist App", layout="wide")

st.title("AI Tourism Recommendation System")

st.markdown(
  """
  ### AI-powered travel behavior prediction and recommendation system 
  Select a user to simulate personalized AI predictions.
  """
)

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
st.success(f"Predicted Visit Mode:{visit_mode_map.get(prediction, prediction)}")

st.subheader("AI Hybrid Recommendations")
current_user_history = df[df["UserId"] == user_id]
liked_attractions = current_user_history[
current_user_history["Rating"] >=4
]["AttractionId].unique()

similar_users = df[
df["AttractionId].isin(liked_attractions)
]["UserId"].unique()

candidate_recommendations = df[
(df["UserId"].isin(similar_users)) & 
(~df["AttractionId"].isin(current_user_history["AttractionId"]))

predicted_label = visit_mode_map.get(prediction, prediction)

mode_preferences = {
"Couples": ["Beaches", "Parks"],
"Family": ["Museums", "Theme Parks"],
"Friends": ["Adventure", "Nightlife"],
"Business": ["Museums"],
"Solo": ["Historic", "Nature"]
}

preferrred_types = mode_preferences.get(predicted_label, [])
if preferred_types:
candidate_recommendations["boost"] = candidate_recommendations[
"AttractionType"
].isin(preferred_types).astype(int)
else:
candidate_recommendations["boost"]=0

recommendations = candidate_recommendations.sort_values(
by=["boost", "attr_avg_rating", "attraction_popularity"],
    ascending=False
)[
    ["AttractionId", "AttractionType", "attr_avg_rating"]
].drop_duplicates().head(5)

st.dataframe(recommendations, use_container_width=True)

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
