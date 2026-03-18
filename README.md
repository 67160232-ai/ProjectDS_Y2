# Movie Recommendation System (Machine Learning Project)

## Project Overview

This project builds a machine learning model to predict movie ratings and recommend movies to users.
The system is based on the MovieLens dataset and compares multiple regression models to determine the best performer.

The final model is used to recommend the top movies for a given user.

---

## Dataset

Dataset used in this project:

MovieLens 20M Dataset

The dataset contains:

* 20 million movie ratings
* 27,000+ movies
* 130,000+ users

Main files used:

* `rating.csv`
* `movie.csv`

---

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training
5. Model Comparison
6. Movie Recommendation

---

## Features Used

The following features were used for training:

* userId
* movieId
* avg_rating
* num_rating

Target variable:

* rating

---

## Models Compared

Three regression models were evaluated:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Evaluation metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

## Model Performance

| Model             | MAE       | RMSE      |
| ----------------- | --------- | --------- |
| Linear Regression | 0.717     | 0.926     |
| Random Forest     | **0.695** | **0.902** |
| Gradient Boosting | 0.708     | 0.915     |

Random Forest achieved the best performance and was selected as the final model.

---

## Recommendation System

The trained model predicts movie ratings for a specific user and recommends the top movies based on predicted ratings.

Example output:

Top 10 recommended movies for a user.

---

## Project Files

```
project_DS.ipynb                # Main notebook
movie_recommendation_model.pkl # Trained model
model_comparison.csv           # Model comparison results
top10_movies.csv               # Recommendation results
```

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Google Colab
* Machine Learning (Regression Models)

---

## Conclusion

Random Forest provided the best prediction performance for movie ratings in this dataset.
The model can be used as the core of a movie recommendation system.

---

## Author

Machine Learning Project – Data Science
