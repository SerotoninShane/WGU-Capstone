from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/books.csv")

# Fill missing values
df.fillna("", inplace=True)

# Remove duplicates (keep the first edition found)
df = df.drop_duplicates(subset=["Title"], keep="first")

# Combine features to create a more robust feature set
df["combined_features"] = df["Title"] + " " + df["Author"] + " " + df["genres"]

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined_features"])

# Train the KNN model using these features
knn_model = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
knn_model.fit(X)

# Train the Cosine Similarity model using the same features
cosine_sim_matrix = cosine_similarity(X)

# Function to fetch book details
def fetch_details(book_titles):
    details = df[df["Title"].isin(book_titles)].drop_duplicates("Title")
    return (
        details["Title"].tolist(),
        details["Author"].tolist(),
        details["publisher"].tolist(),
        details["average_rating"].tolist(),
        details["genres"].tolist()
    )

# KNN Recommendation Function
def recommend_knn(book_name):
    book_idx = df[df["Title"] == book_name].index[0]
    distances, indices = knn_model.kneighbors(X[book_idx], n_neighbors=20)
    
    recommended_books = df.iloc[indices[0]]["Title"].tolist()
    
    return fetch_details(recommended_books)

def recommend_cosine(book_name):
    book_idx = df[df["Title"] == book_name].index[0]
    scores = list(enumerate(cosine_sim_matrix[book_idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:20]
    recommended_books = [df.iloc[i[0]]["Title"] for i in sorted_scores]

    return fetch_details(recommended_books)

def compute_similarity(selected_book, recommended_titles, genre_weight=1.0, author_weight=1.0):
    # Get the index for the selected book
    selected_idx = df[df["Title"] == selected_book].index[0]
    
    # Ensure we only use valid indices
    similarities = []
    
    # Get the total number of books in the cosine similarity matrix (should match the number of rows/columns)
    max_idx = cosine_sim_matrix.shape[0]  # or cosine_sim_matrix.shape[1], they should be equal
    
    for title in recommended_titles:
        rec_idx_list = df[df["Title"] == title].index
        if len(rec_idx_list) > 0:
            rec_idx = rec_idx_list[0]

            # Check if rec_idx is within valid bounds of the cosine similarity matrix
            if rec_idx < max_idx:
                # Get the cosine similarity score
                cosine_sim_score = cosine_sim_matrix[selected_idx, rec_idx]

                # Boost similarity based on the same author
                if df.iloc[selected_idx]["Author"] == df.iloc[rec_idx]["Author"]:
                    cosine_sim_score *= author_weight  # Increase weight if the authors are the same

                # Boost similarity based on the same genres (split genres into lists and compare)
                selected_genres = set(df.iloc[selected_idx]["genres"].split(','))
                rec_genres = set(df.iloc[rec_idx]["genres"].split(','))
                common_genres = selected_genres & rec_genres  # Intersection of genres

                cosine_sim_score += genre_weight * len(common_genres)  # Add weight for shared genres

                # Ensure cosine_sim_score is between 0 and 1
                cosine_sim_score = min(1, max(0, cosine_sim_score))

                similarities.append({
                    "title": title,
                    "similarity": cosine_sim_score,
                })
            else:
                similarities.append({
                    "title": title,
                    "similarity": 0,  # Default to 0 if the index is out of bounds
                })
        else:
            similarities.append({
                "title": title,
                "similarity": 0,  # Default to 0 if the book was not found
            })
    
    # Sort by similarity score (in descending order)
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Extract sorted titles and similarities
    sorted_titles = [item["title"] for item in similarities]
    sorted_similarities = [item["similarity"] for item in similarities]

    return sorted_titles, sorted_similarities