"""
Movie Recommendation System
Implements collaborative filtering using cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class MovieRecommender:
    """Movie recommendation system using collaborative filtering"""
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.movie_similarity = None
        self.tfidf_vectorizer = None
        self.genre_similarity = None
    
    def load_data(self, movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
        """Load movies and ratings data"""
        print("Loading data...")
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Normalize column names (MovieLens uses camelCase, we use snake_case)
        if 'userId' in self.ratings_df.columns:
            self.ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
        if 'movieId' in self.movies_df.columns:
            self.movies_df.rename(columns={'movieId': 'movie_id'}, inplace=True)
        
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        print(f"Users: {self.ratings_df['user_id'].nunique()}")
    
    def create_user_movie_matrix(self):
        """Create user-movie rating matrix"""
        print("Creating user-movie matrix...")
        
        # For large datasets (MovieLens), limit to top users and movies for performance
        if len(self.ratings_df) > 10000:
            print("Large dataset detected. Using top users and movies for performance...")
            top_users = self.ratings_df['user_id'].value_counts().head(100).index
            top_movies = self.ratings_df['movie_id'].value_counts().head(500).index
            
            filtered_ratings = self.ratings_df[
                self.ratings_df['user_id'].isin(top_users) & 
                self.ratings_df['movie_id'].isin(top_movies)
            ]
            self.ratings_df = filtered_ratings
            print(f"Filtered to {len(self.ratings_df)} ratings")
        
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        print(f"Matrix shape: {self.user_movie_matrix.shape}")
    
    def compute_item_similarity(self):
        """Compute item-item similarity using cosine similarity"""
        print("Computing item-item similarity...")
        # Transpose to get movie-movie matrix
        movie_matrix = self.user_movie_matrix.T
        self.movie_similarity = cosine_similarity(movie_matrix)
        print("Item similarity computed")
    
    def compute_genre_similarity(self):
        """Compute genre-based similarity using TF-IDF"""
        print("Computing genre similarity...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # MovieLens uses pipe-separated genres, handle that
        if 'genres' in self.movies_df.columns:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'[^|]+')
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['genres'])
        elif 'genre' in self.movies_df.columns:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['genre'])
        else:
            print("No genre column found, skipping genre similarity")
            self.genre_similarity = None
            return
        
        self.genre_similarity = cosine_similarity(tfidf_matrix)
        print("Genre similarity computed")
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using collaborative filtering"""
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found. Returning popular movies.")
            return self.get_popular_movies(n_recommendations)
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Find movies not rated by user
        unrated_movies = user_ratings[user_ratings == 0].index
        
        if len(unrated_movies) == 0:
            print("User has rated all movies.")
            return []
        
        # Calculate predicted ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            movie_idx = list(self.user_movie_matrix.columns).index(movie_id)
            
            # Get similarity scores for this movie
            similarity_scores = self.movie_similarity[movie_idx]
            
            # Get rated movies by user
            rated_movies = user_ratings[user_ratings > 0]
            
            if len(rated_movies) == 0:
                continue
            
            # Calculate weighted average
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_movie_id, rating in rated_movies.items():
                rated_movie_idx = list(self.user_movie_matrix.columns).index(rated_movie_id)
                similarity = similarity_scores[rated_movie_idx]
                
                if similarity > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = predictions[:n_recommendations]
        
        # Get movie details
        recommendations = []
        for movie_id, predicted_rating in top_recommendations:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if len(movie_info) > 0:
                movie_info = movie_info.iloc[0]
                genre = movie_info.get('genres', movie_info.get('genre', 'Unknown'))
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'],
                    'genre': genre,
                    'year': movie_info.get('year', 'Unknown'),
                    'predicted_rating': predicted_rating
                })
        
        return recommendations
    
    def get_content_based_recommendations(self, movie_title, n_recommendations=10):
        """Get recommendations based on genre similarity"""
        if self.genre_similarity is None:
            print("Genre similarity not available.")
            return []
        
        # Find movie (case-insensitive search)
        movie = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()]
        if len(movie) == 0:
            print(f"Movie '{movie_title}' not found.")
            # Try partial match
            movie = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False, na=False)]
            if len(movie) == 0:
                return []
        
        movie_idx = movie.index[0]
        
        # Get similarity scores
        similarity_scores = self.genre_similarity[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = self.movies_df.iloc[idx]
            genre = movie_info.get('genres', movie_info.get('genre', 'Unknown'))
            recommendations.append({
                'movie_id': movie_info['movie_id'],
                'title': movie_info['title'],
                'genre': genre,
                'year': movie_info.get('year', 'Unknown'),
                'similarity': similarity_scores[idx]
            })
        
        return recommendations
    
    def get_popular_movies(self, n_recommendations=10):
        """Get popular movies based on average rating and number of ratings"""
        # Calculate average rating and count for each movie
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'avg_rating', 'rating_count']
        
        # Filter movies with at least 5 ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 5]
        
        # Sort by average rating
        popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
        
        # Get top N
        top_movies = popular_movies.head(n_recommendations)
        
        recommendations = []
        for _, row in top_movies.iterrows():
            movie_info = self.movies_df[self.movies_df['movie_id'] == row['movie_id']]
            if len(movie_info) > 0:
                movie_info = movie_info.iloc[0]
                genre = movie_info.get('genres', movie_info.get('genre', 'Unknown'))
                recommendations.append({
                    'movie_id': row['movie_id'],
                    'title': movie_info['title'],
                    'genre': genre,
                    'year': movie_info.get('year', 'Unknown'),
                    'avg_rating': row['avg_rating'],
                    'rating_count': row['rating_count']
                })
        
        return recommendations
    
    def get_user_history(self, user_id, n_movies=10):
        """Get user's rating history"""
        if user_id not in self.ratings_df['user_id'].values:
            return []
        
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        user_ratings = user_ratings.sort_values('rating', ascending=False)
        user_ratings = user_ratings.head(n_movies)
        
        history = []
        for _, row in user_ratings.iterrows():
            movie_info = self.movies_df[self.movies_df['movie_id'] == row['movie_id']]
            if len(movie_info) > 0:
                movie_info = movie_info.iloc[0]
                genre = movie_info.get('genres', movie_info.get('genre', 'Unknown'))
                history.append({
                    'movie_id': row['movie_id'],
                    'title': movie_info['title'],
                    'genre': genre,
                    'year': movie_info.get('year', 'Unknown'),
                    'rating': row['rating']
                })
        
        return history
    
    def save_model(self, path='models/recommender_model.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df,
            'user_movie_matrix': self.user_movie_matrix,
            'movie_similarity': self.movie_similarity,
            'genre_similarity': self.genre_similarity
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/recommender_model.pkl'):
        """Load the trained model"""
        model_data = joblib.load(path)
        
        self.movies_df = model_data['movies_df']
        self.ratings_df = model_data['ratings_df']
        self.user_movie_matrix = model_data['user_movie_matrix']
        self.movie_similarity = model_data['movie_similarity']
        self.genre_similarity = model_data['genre_similarity']
        
        print("Model loaded successfully")

def train_model():
    """Train the recommendation model"""
    print("=" * 50)
    print("TRAINING RECOMMENDATION MODEL")
    print("=" * 50)
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load data
    recommender.load_data()
    
    # Create matrices
    recommender.create_user_movie_matrix()
    
    # Compute similarities
    recommender.compute_item_similarity()
    recommender.compute_genre_similarity()
    
    # Save model
    recommender.save_model()
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)
    
    return recommender

def main():
    """Train and test the recommendation system"""
    # Train model
    recommender = train_model()
    
    # Test recommendations
    print("\n" + "=" * 50)
    print("TESTING RECOMMENDATIONS")
    print("=" * 50)
    
    # Test popular movies
    print("\nPopular movies:")
    popular_recs = recommender.get_popular_movies(5)
    for rec in popular_recs:
        print(f"  - {rec['title']} ({rec['genre']}) - Rating: {rec['avg_rating']:.2f} ({rec['rating_count']} ratings)")

if __name__ == "__main__":
    main()
