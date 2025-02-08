import pandas as pd
import streamlit as st
from ml_models import recommend_knn, recommend_cosine, compute_similarity

# Load the dataset
df = pd.read_csv("data/books.csv")

def display_recommendations():
    selected_book = st.selectbox("Select a book", df["Title"].dropna().unique())
    method = st.radio("Choose recommendation technique", ["KNN", "Cosine Similarity"])

    if st.button("Show Recommendations"):
        if method == "KNN":
            titles, authors, publishers, ratings, genres = recommend_knn(selected_book)
        else:
            titles, authors, publishers, ratings, genres = recommend_cosine(selected_book)

        # Compute similarity scores with weight for genre and author
        titles, similarities = compute_similarity(selected_book, titles, genre_weight=1.5, author_weight=1.2)

        st.header(f"Books Similar to '{selected_book}'")
        st.write("----")

        for i in range(len(titles)):
            st.write(f"**{titles[i]}** by {authors[i]}")
            st.write(f"Publisher: {publishers[i]}")
            st.write(f"Average Rating: {ratings[i]}")
            st.write(f"🔍 **Similarity Score:** {similarities[i]:.2f}")

            # Format genres properly
            if isinstance(genres[i], str):
                cleaned_genres = genres[i].replace(';', ',').replace(' ,', ',').split(',')
                formatted_genres = [genre.strip() for genre in cleaned_genres if genre.strip()]
                genres_str = ', '.join(formatted_genres)
                st.write(f"*Genres:* {genres_str}")
                
            st.write("----") 