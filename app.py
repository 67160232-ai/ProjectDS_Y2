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

st.write("👤 Total Users:", data["userId"].nunique())
st.write("🎬 Total Movies:", data["movieId"].nunique())

# =========================
# Feature Engineering (สำคัญ)
# =========================

# สร้าง avg_rating + num_rating
movie_stats = data.groupby("movieId").agg(
    avg_rating=("rating", "mean"),
    num_rating=("rating", "count")
).reset_index()

# merge เข้า data
data = data.merge(movie_stats, on="movieId", how="left")

# เอา movie ไม่ซ้ำ
movies_unique = data.drop_duplicates("movieId").copy()

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

    user_id = st.number_input("Enter User ID", min_value=1, value=1)

    if st.button("Recommend Movies"):

        # เช็ค column ก่อน (กัน error)
        required_cols = ["movieId", "avg_rating", "num_rating"]
        if not all(col in movies_unique.columns for col in required_cols):
            st.error("Missing columns for prediction")
        else:
            # เตรียม input
            X_pred = movies_unique[required_cols].copy()
            X_pred["userId"] = user_id

            # เรียง column ให้ตรงตอน train
            X_pred = X_pred[[
                "userId",
                "movieId",
                "avg_rating",
                "num_rating"
            ]]

            # predict
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
