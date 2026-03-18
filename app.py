import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")

st.write("ระบบแนะนำหนังด้วย Machine Learning")

# โหลดโมเดล
st.title("🎬 Movie Recommendation System")

data = pd.read_csv("movie_data_used.csv")

st.write("Choose a movie you like")

movie_list = data["title"].unique()

selected_movie = st.selectbox("Movie", movie_list)

if st.button("Recommend"):

    movie_genre = data[data["title"] == selected_movie]["genres"].values[0]

    recommend = data[data["genres"] == movie_genre]

    st.write("Recommended Movies")

    st.dataframe(recommend[["title","genres"]].head(10))

# โหลด dataset
@st.cache_data
def load_data():
    return pd.read_csv("movie_data_used.csv")

data = load_data()

# Feature engineering
movie_stats = data.groupby("movieId").agg({
    "rating": ["mean","count"]
})

movie_stats.columns = ["avg_rating","num_rating"]
movie_stats = movie_stats.reset_index()

data = pd.merge(data, movie_stats, on="movieId")

movies_unique = data.drop_duplicates("movieId")

# =========================
# วิธีเลือก recommendation
# =========================

method = st.radio(
    "Choose recommendation method",
    ["Enter User ID", "Select Favorite Movie"]
)

# =========================
# METHOD 1: USER ID
# =========================

if method == "Enter User ID":

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1
    )

    if st.button("Recommend Movies"):

        X_pred = movies_unique[[
            "userId",
            "movieId",
            "avg_rating",
            "num_rating"
        ]].copy()

        X_pred["userId"] = user_id

        pred = model.predict(X_pred)

        movies_unique["pred_rating"] = pred

        recommend = movies_unique.sort_values(
            "pred_rating",
            ascending=False
        )

        result = recommend[["title","genres","pred_rating"]].head(10)

        st.subheader("Top Recommended Movies")

        st.dataframe(result)


# =========================
# METHOD 2: SELECT MOVIE
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

        st.subheader("Movies you might like")

        st.dataframe(result)
