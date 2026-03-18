import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("ระบบแนะนำหนังด้วย Machine Learning")

# =========================
# โหลดข้อมูล
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("movie_data_used.csv")

data = load_data()

# =========================
# โหลดโมเดล
# =========================

@st.cache_resource
def load_model():
    return joblib.load("movie_model_small.pkl")

model = load_model()

# =========================
# Dataset Info
# =========================

st.subheader("📊 Dataset Information")

num_users = data["userId"].nunique()
num_movies = data["movieId"].nunique()

st.write("👤 Total Users:", num_users)
st.write("🎬 Total Movies:", num_movies)

# =========================
# Feature Engineering
# =========================

movie_stats = data.groupby("movieId").agg({
    "rating": ["mean","count"]
})

movie_stats.columns = ["avg_rating","num_rating"]
movie_stats = movie_stats.reset_index()

data = pd.merge(data, movie_stats, on="movieId")

movies_unique = data.drop_duplicates("movieId")

# =========================
# เลือกวิธีแนะนำ
# =========================

method = st.radio(
    "Choose recommendation method",
    ["Enter User ID", "Select Favorite Movie"]
)

# =========================
# METHOD 1 : USER ID
# =========================

if method == "Enter User ID":

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1
    )

    if st.button("Recommend Movies"):

        X_pred = movies_unique[[
            "movieId",
            "avg_rating",
            "num_rating"
        ]].copy()

        X_pred["userId"] = user_id

        X_pred = X_pred[[
            "userId",
            "movieId",
            "avg_rating",
            "num_rating"
        ]]

        pred = model.predict(X_pred)

        movies_unique["pred_rating"] = pred

        recommend = movies_unique.sort_values(
            "pred_rating",
            ascending=False
        )

        result = recommend[[
            "title",
            "genres",
            "pred_rating"
        ]].head(10)

        st.subheader("🎬 Top Recommended Movies")

        st.dataframe(result)

# =========================
# METHOD 2 : SELECT MOVIE
# =========================

else:

    movie_list = movies_unique["title"].tolist()

    selected_movie = st.selectbox(
        "Choose a movie you like",
        movie_list
    )

    if st.button("Recommend Similar Movies"):

        movie_genre = movies_unique[
            movies_unique["title"] == selected_movie
        ]["genres"].values[0]

        similar_movies = movies_unique[
            movies_unique["genres"] == movie_genre
        ]

        result = similar_movies[[
            "title",
            "genres"
        ]].head(10)

        st.subheader("🎬 Movies you might like")

        st.dataframe(result)
