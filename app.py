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

st.subheader("Predicted Visit Mode")

features = [
  "VisitYear",
  "VisitMonth",
  "attraction_popularity",
  "attr_avg_rating",
  "user_total_visits",
  "user_avg_rating"
]

user_data = df[df["UserId"] == user_id][features].iloc[0]
prediction = clf.predict([user_data])[0]
st.success(f"Predicted Visit Mode:{visit_mode_map.get(prediction, prediction)}")

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
