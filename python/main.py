import streamlit as st
import popular_books as popular_books  # Ensure this is correctly imported
import recommended_books as recommended_books  # Ensure this is correctly imported

st.title("Literature Recommendation")

# Sidebar navigation
st.sidebar.title("Menu")
option = st.sidebar.selectbox("Choose a section", ["Popular", "Recommended"])

if option == "Popular":
    popular_books.display_popular_books()  # Call the function in popular_books.py
elif option == "Recommended":
    recommended_books.display_recommendations()  # Call the function in recommended_books.py
