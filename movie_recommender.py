import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
movies = pd.read_csv("dataset/movies.csv")

# Replace | with space in genres
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found in dataset"]

    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i in similarity_scores[1:num_recommendations+1]:
        recommended_movies.append(movies.iloc[i[0]]['title'])

    return recommended_movies

# Example usage
movie_name = "Toy Story (1995)"
recommendations = recommend_movies(movie_name)

# Save output
os.makedirs("output", exist_ok=True)
with open("output/recommendations.txt", "w") as f:
    f.write(f"Recommendations for '{movie_name}':\n")
    for movie in recommendations:
        f.write(movie + "\n")

print("Recommendations saved to output/recommendations.txt")
