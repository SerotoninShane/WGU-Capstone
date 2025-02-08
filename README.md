# WGU Capstone - Book Recommendation System

## Description
This project aims to build a book recommendation system based on genres and other metadata from the Goodreads dataset. The system uses algorithms such as KNN (K-Nearest Neighbors) and Cosine Similarity to recommend books based on user input.

## Source Code Location
The source code is available on GitHub: [https://github.com/SerotoninShane/WGU-Capstone](https://github.com/SerotoninShane/WGU-Capstone)

### Main files:
- **`python/main.py`**: Streamlit application for displaying recommendations.
- **`python/recommend.py`**: Algorithm for generating book recommendations.
- **`python/data/`**: Folder containing the `books.csv` dataset.

## Dependencies
The following libraries are required to run the project:
- `streamlit`: For the app interface.
- `scikit-learn`: For machine learning algorithms.
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/SerotoninShane/WGU-Capstone.git
   cd WGU-Capstone/python
   ```

2. Install the required dependencies:
   ```bash
   pip install -r python/requirements.txt
   ```

3. Ensure the `books.csv` dataset is in the `python/data/` folder. If not, download it from [Kaggle: Goodreads Books with Genres](https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres?resource=download).

4. Run the application:
   ```bash
   streamlit run python/main.py
   ```

5. The app will open in your default web browser, and you can input book details to get similar book recommendations.

## Usage
- Input a book or author name to see a list of recommended books based on your input.
- The recommendation system leverages machine learning algorithms to find similar books based on shared genres and other metadata.

## License
This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.
