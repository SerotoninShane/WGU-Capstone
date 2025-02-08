import pandas as pd
import streamlit as st

# Load the dataset
df = pd.read_csv("data/books.csv")

# Calculate global average rating
global_avg = df['average_rating'].mean()

# Constant C to weight the ratings by the number of reviews
C = 1000  # You can adjust this constant to change the weight

# Function to calculate weighted rating
def weighted_rating(x, global_avg=global_avg, C=C):
    v = x['ratings_count']  # Number of ratings for the book
    R = x['average_rating']  # Average rating for the book
    return (R * v + global_avg * C) / (v + C)

# Apply weighted rating function to the DataFrame
df['weighted_rating'] = df.apply(weighted_rating, axis=1)

# Function to display popular books (you can call this in the main script)
def display_popular_books():
    st.header("Top 50 Highest Average Rated Stories")
    st.write("----")

    # Sort by weighted rating and get top 50
    for _, row in df.nlargest(50, 'weighted_rating').iterrows():
        st.write(f"**{row['Title']}** by {row['Author']}")

        # Display publisher
        st.write(f"Publisher: {row['publisher']}")

        # Display average rating
        st.write(f"Average Rating: {row['average_rating']:.2f}")

        # Add genres to the display
        if isinstance(row['genres'], str):  # Check if genres are in a string format
            # Replace commas and semicolons with a single delimiter (e.g., comma)
            cleaned_genres = row['genres'].replace(';', ',').replace(' ,', ',').split(',')

            # Strip leading/trailing spaces from each genre
            formatted_genres = [genre.strip() for genre in cleaned_genres if genre.strip()]

            # Join genres with commas for display
            genres_str = ', '.join(formatted_genres)
            st.write(f"*Genres:* {genres_str}")

        st.write("----")
