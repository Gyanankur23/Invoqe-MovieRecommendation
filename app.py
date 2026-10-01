"""
Streamlit Application for Movie Recommendation System
Interactive interface for movie recommendations
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 AI-Powered Movie Recommendation System")
st.markdown("""
This application provides personalized movie recommendations using collaborative filtering and content-based filtering.
Built using machine learning and cosine similarity algorithms.
""")

# Load model
@st.cache_resource
def load_model():
    """Load the trained recommendation model or train if not found"""
    try:
        model_data = joblib.load('models/recommender_model.pkl')
        return model_data
    except:
        st.warning("Model not found. Preparing data and training model... This may take a moment.")
        try:
            # First create data if it doesn't exist
            from data.create_dataset import download_movielens_dataset
            try:
                download_movielens_dataset()
            except:
                # If download fails, create sample data
                from data.create_dataset import create_sample_dataset
                create_sample_dataset()
            
            # Then train model
            from recommendation import train_model
            model_data = train_model()
            return model_data
        except Exception as e:
            st.error(f"Error preparing model: {str(e)}")
            return None

model_data = load_model()

if model_data is not None:
    movies_df = model_data['movies_df']
    ratings_df = model_data['ratings_df']
    user_movie_matrix = model_data['user_movie_matrix']
    movie_similarity = model_data['movie_similarity']
    genre_similarity = model_data['genre_similarity']
    
    # Sidebar for navigation
    st.sidebar.header("Recommendation Type")
    
    recommendation_type = st.sidebar.radio(
        "Choose recommendation type:",
        ["Collaborative Filtering", "Content-Based", "Popular Movies"]
    )
    
    if recommendation_type == "Collaborative Filtering":
        st.header("👥 Collaborative Filtering Recommendations")
        st.markdown("Get personalized recommendations based on similar users' preferences.")
        
        # User selection
        user_ids = sorted(user_movie_matrix.index.tolist())
        selected_user = st.selectbox("Select User ID:", user_ids)
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("Get Recommendations"):
            # Get user's rating history
            from recommendation import MovieRecommender
            recommender = MovieRecommender()
            recommender.movies_df = movies_df
            recommender.ratings_df = ratings_df
            recommender.user_movie_matrix = user_movie_matrix
            recommender.movie_similarity = movie_similarity
            recommender.genre_similarity = genre_similarity
            
            # Show user history
            st.subheader(f"User {selected_user}'s Rating History")
            user_history = recommender.get_user_history(selected_user, 10)
            
            if user_history:
                history_df = pd.DataFrame(user_history)
                st.dataframe(history_df[['title', 'genre', 'year', 'rating']])
            else:
                st.info("No rating history found for this user.")
            
            # Get recommendations
            st.subheader(f"Recommended Movies for User {selected_user}")
            recommendations = recommender.get_collaborative_recommendations(selected_user, n_recs)
            
            if recommendations:
                recs_df = pd.DataFrame(recommendations)
                st.dataframe(recs_df[['title', 'genre', 'year', 'predicted_rating']])
                
                # Display as cards
                st.subheader("Movie Cards")
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    col = cols[i % 3]
                    with col:
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 10px;'>
                            <h3>{rec['title']}</h3>
                            <p><strong>Genre:</strong> {rec['genre']}</p>
                            <p><strong>Year:</strong> {rec['year']}</p>
                            <p><strong>Predicted Rating:</strong> {rec['predicted_rating']:.2f}/5</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No recommendations available for this user.")
    
    elif recommendation_type == "Content-Based":
        st.header("🎭 Content-Based Recommendations")
        st.markdown("Get recommendations based on movie genre similarity.")
        
        # Movie selection
        movie_titles = sorted(movies_df['title'].tolist())
        selected_movie = st.selectbox("Select a movie you like:", movie_titles)
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("Get Similar Movies"):
            from recommendation import MovieRecommender
            recommender = MovieRecommender()
            recommender.movies_df = movies_df
            recommender.ratings_df = ratings_df
            recommender.user_movie_matrix = user_movie_matrix
            recommender.movie_similarity = movie_similarity
            recommender.genre_similarity = genre_similarity
            
            # Get recommendations
            st.subheader(f"Movies Similar to '{selected_movie}'")
            recommendations = recommender.get_content_based_recommendations(selected_movie, n_recs)
            
            if recommendations:
                recs_df = pd.DataFrame(recommendations)
                st.dataframe(recs_df[['title', 'genre', 'year', 'similarity']])
                
                # Display as cards
                st.subheader("Movie Cards")
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    col = cols[i % 3]
                    with col:
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; margin-bottom: 10px;'>
                            <h3>{rec['title']}</h3>
                            <p><strong>Genre:</strong> {rec['genre']}</p>
                            <p><strong>Year:</strong> {rec['year']}</p>
                            <p><strong>Similarity:</strong> {rec['similarity']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No similar movies found.")
    
    elif recommendation_type == "Popular Movies":
        st.header("🔥 Popular Movies")
        st.markdown("Discover top-rated movies based on user ratings.")
        
        # Number of recommendations
        n_recs = st.slider("Number of movies:", 5, 30, 15)
        
        if st.button("Get Popular Movies"):
            from recommendation import MovieRecommender
            recommender = MovieRecommender()
            recommender.movies_df = movies_df
            recommender.ratings_df = ratings_df
            recommender.user_movie_matrix = user_movie_matrix
            recommender.movie_similarity = movie_similarity
            recommender.genre_similarity = genre_similarity
            
            # Get popular movies
            st.subheader("Top Rated Movies")
            recommendations = recommender.get_popular_movies(n_recs)
            
            if recommendations:
                recs_df = pd.DataFrame(recommendations)
                st.dataframe(recs_df[['title', 'genre', 'year', 'avg_rating', 'rating_count']])
                
                # Display as cards
                st.subheader("Movie Cards")
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    col = cols[i % 3]
                    with col:
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; margin-bottom: 10px;'>
                            <h3>{rec['title']}</h3>
                            <p><strong>Genre:</strong> {rec['genre']}</p>
                            <p><strong>Year:</strong> {rec['year']}</p>
                            <p><strong>Avg Rating:</strong> {rec['avg_rating']:.2f}/5</p>
                            <p><strong>Ratings:</strong> {rec['rating_count']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No popular movies found.")
    
    # Dataset statistics
    st.sidebar.header("Dataset Statistics")
    st.sidebar.metric("Total Movies", len(movies_df))
    st.sidebar.metric("Total Users", ratings_df['user_id'].nunique())
    st.sidebar.metric("Total Ratings", len(ratings_df))
    
    # Genre distribution
    st.sidebar.header("Genre Distribution")
    genre_counts = movies_df['genre'].value_counts()
    st.sidebar.bar_chart(genre_counts)
    
    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. **Collaborative Filtering**: Select a user ID to get personalized recommendations based on similar users' preferences.
    2. **Content-Based**: Select a movie you like to get similar movies based on genre.
    3. **Popular Movies**: View top-rated movies based on average ratings and number of ratings.
    
    The system uses cosine similarity to find similar movies and users.
    """)
    
    # About
    with st.expander("About the System"):
        st.markdown("""
        **Recommendation Algorithms:**
        
        - **Collaborative Filtering**: Recommends items based on similar users' preferences. Uses item-item collaborative filtering with cosine similarity.
        
        - **Content-Based Filtering**: Recommends items similar to those the user liked in the past. Uses TF-IDF vectorization on genres and cosine similarity.
        
        - **Popular Movies**: Shows top-rated movies based on average rating and minimum number of ratings.
        
        **Dataset:**
        - Synthetic movie dataset with 100 movies
        - 200 users with 5000+ ratings
        - Multiple genres including Drama, Action, Sci-Fi, Comedy, etc.
        """)
