import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("ระบบแนะนำหนังด้วย Machine Learning")

# โหลด dataset
@st.cache_data
def load_data():
    return pd.read_csv("movie_data_used.csv")

data = load_data()

# =========================
# วิธีเลือก recommendation
# =========================

method = st.radio(
    "Choose recommendation method",
    ["Select Favorite Movie", "Enter User ID"]
)

# =========================
# METHOD 1: SELECT MOVIE
# =========================

if method == "Select Favorite Movie":

    movie_list = data["title"].unique()

    selected_movie = st.selectbox(
        "Choose a movie you like",
        movie_list
    )

    if st.button("Recommend"):

        movie_genre = data[
            data["title"] == selected_movie
        ]["genres"].values[0]

        similar_movies = data[
            data["genres"] == movie_genre
        ]

        st.subheader("Recommended Movies")

        st.dataframe(
            similar_movies[["title", "genres"]]
            .drop_duplicates()
            .head(10)
        )

# =========================
# METHOD 2: USER ID
# =========================

else:

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1
    )

    if st.button("Recommend Movies"):

        user_movies = data[data["userId"] == user_id]

        if len(user_movies) == 0:
            st.warning("User not found in dataset")
        else:

            favorite_genres = user_movies["genres"].mode()[0]

            recommend = data[data["genres"] == favorite_genres]

            st.subheader("Recommended Movies")

            st.dataframe(
                recommend[["title", "genres"]]
                .drop_duplicates()
                .head(10)
            )
