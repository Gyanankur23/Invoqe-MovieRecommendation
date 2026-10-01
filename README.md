# Task 4: AI-Powered Recommendation System - Movie Recommendations

## Overview
This project implements a movie recommendation system using collaborative filtering and content-based filtering techniques. It includes data generation, model training with cosine similarity, and an interactive Streamlit interface for personalized recommendations.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Cosine Similarity
- TF-IDF Vectorization

## Project Structure
```
task4/
├── data/
│   ├── movies.csv              # Movie dataset
│   └── ratings.csv             # User ratings dataset
├── models/
│   └── recommender_model.pkl   # Trained recommendation model
├── create_dataset.py           # Dataset creation script
├── recommendation.py           # Recommendation engine
├── app.py                      # Streamlit application
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Create Dataset
```bash
python create_dataset.py
```
This generates a synthetic movie dataset with 100 movies and 200 users with 5000+ ratings.

### Step 2: Train Recommendation Model
```bash
python recommendation.py
```
This will:
- Load the dataset
- Create user-movie rating matrix
- Compute item-item similarity using cosine similarity
- Compute genre-based similarity using TF-IDF
- Save the trained model to `models/` directory

### Step 3: Run Streamlit Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`

## Methodology

### Data Loading
- MovieLens Small Dataset (real public dataset from GroupLens Research)
- 9,742 movies with genre information
- 610 users with 100,836 ratings
- Automatically downloads from GroupLens if not available
- Falls back to sample dataset if download fails
- Performance optimization for large datasets (filters to top users/movies)

### Collaborative Filtering
- Creates user-movie rating matrix
- Computes item-item similarity using cosine similarity
- Predicts ratings for unrated movies based on similar items
- Recommends top N movies with highest predicted ratings

### Content-Based Filtering
- Uses TF-IDF vectorization on movie genres
- Computes genre similarity using cosine similarity
- Recommends movies similar to a selected movie
- Useful for new users with no rating history

### Popular Movies
- Calculates average rating and rating count for each movie
- Filters movies with minimum 5 ratings
- Sorts by average rating
- Recommends top-rated movies

## Features
- **Collaborative Filtering**: Personalized recommendations based on user preferences
- **Content-Based Filtering**: Genre-based similarity recommendations
- **Popular Movies**: Top-rated movies discovery
- **User History**: View user's rating history
- **Interactive Interface**: Beautiful Streamlit UI with movie cards
- **Multiple Views**: Dataframe and card-based displays
- **Genre Distribution**: Visual genre statistics

## Recommendation Algorithms

### Collaborative Filtering
1. Create user-movie rating matrix
2. Compute item-item similarity using cosine similarity
3. For each unrated movie, calculate predicted rating:
   - Weighted average of ratings from similar movies
   - Weight = similarity score
4. Recommend movies with highest predicted ratings

### Content-Based Filtering
1. Vectorize movie genres using TF-IDF
2. Compute genre similarity matrix using cosine similarity
3. For a selected movie, find most similar movies
4. Recommend top N similar movies

### Popular Movies
1. Calculate average rating and count for each movie
2. Filter movies with minimum rating threshold
3. Sort by average rating (descending)
4. Recommend top N movies

## Dataset Information
- **Movies**: 100
- **Users**: 200
- **Ratings**: 5000+
- **Genres**: Drama, Action, Sci-Fi, Comedy, Crime, Thriller, Romance, etc.
- **Rating Scale**: 1-5 stars

## Streamlit Interface Features

### Collaborative Filtering Tab
- Select user ID from dropdown
- View user's rating history
- Get personalized recommendations
- Adjust number of recommendations
- Display as dataframe and movie cards

### Content-Based Tab
- Select a movie from dropdown
- Get similar movies based on genre
- View similarity scores
- Display as dataframe and movie cards

### Popular Movies Tab
- View top-rated movies
- See average ratings and rating counts
- Adjust number of movies to display
- Display as dataframe and movie cards

### Sidebar Features
- Dataset statistics
- Genre distribution chart
- Navigation between recommendation types

## Deployment

### Deploy to Vercel
1. Create `requirements.txt`
2. Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
```
3. Push to GitHub
4. Connect to Vercel
5. Deploy as Streamlit app

### Alternative Deployment Options
- Streamlit Cloud
- Heroku
- Railway
- AWS/GCP/Azure

## Future Enhancements
- Hybrid recommendation system (combining collaborative and content-based)
- User profiles and preferences
- Recommendation ranking and filtering
- Search functionality
- Real dataset integration (MovieLens, IMDb)
- Matrix factorization (SVD, NMF)
- Deep learning-based recommendations
- Cold start problem solutions
- A/B testing framework

## License
This project is created for the Invoqe AI/ML Internship Program.
