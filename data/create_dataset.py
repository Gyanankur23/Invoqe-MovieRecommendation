"""
Download MovieLens Dataset (Real Public Dataset)
Dataset: MovieLens Small Dataset (100k ratings)
Source: https://grouplens.org/datasets/movielens/
This is a real-world movie ratings dataset from GroupLens Research
"""

import pandas as pd
import os
import requests
import zipfile
from io import BytesIO

def download_movielens_dataset():
    """Download MovieLens Small Dataset (real public dataset)"""
    print("Downloading MovieLens Small Dataset (real public dataset)...")
    
    # MovieLens Small Dataset URL
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    try:
        # Download
        print("Downloading from GroupLens...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract
        print("Extracting files...")
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('data/temp')
        
        # Read the files
        movies = pd.read_csv('data/temp/ml-latest-small/movies.csv')
        ratings = pd.read_csv('data/temp/ml-latest-small/ratings.csv')
        
        # Clean up
        import shutil
        shutil.rmtree('data/temp')
        
        # Save to our data directory
        os.makedirs('data', exist_ok=True)
        movies.to_csv('data/movies.csv', index=False)
        ratings.to_csv('data/ratings.csv', index=False)
        
        print(f"Dataset saved successfully!")
        print(f"Movies: {len(movies)}")
        print(f"Ratings: {len(ratings)}")
        print(f"Users: {ratings['userId'].nunique()}")
        
        print("\nDataset Info:")
        print("- Source: MovieLens (GroupLens Research - Real public dataset)")
        print("- Version: ml-latest-small")
        print("- Samples: 100,836 ratings")
        print("- Movies: 9,742 movies")
        print("- Users: 610 users")
        print("- Rating scale: 0.5 to 5 stars")
        
        print("\nSample movies:")
        print(movies.head(10))
        
        print("\nSample ratings:")
        print(ratings.head(10))
        
        return movies, ratings
        
    except Exception as e:
        print(f"Error downloading MovieLens dataset: {e}")
        print("\nFalling back to built-in sample dataset...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset if download fails"""
    print("Creating sample movie dataset...")
    
    # Sample movie data
    movies_data = {
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)',
            'Heat (1995)', 'Sabrina (1995)', 'Tom and Huck (1995)',
            'Sudden Death (1995)', 'GoldenEye (1995)'
        ],
        'genres': [
            'Adventure|Animation|Children|Comedy|Fantasy',
            'Adventure|Children|Fantasy', 'Comedy|Drama|Romance',
            'Comedy|Drama|Romance', 'Comedy',
            'Action|Crime|Thriller', 'Comedy|Romance',
            'Adventure|Children', 'Action', 'Action|Adventure|Thriller'
        ]
    }
    
    # Sample ratings data
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        'movieId': [1, 2, 3, 1, 4, 2, 5, 3, 6, 1],
        'rating': [4.0, 5.0, 3.5, 5.0, 4.0, 3.0, 4.5, 5.0, 3.5, 4.0],
        'timestamp': [964982703, 964981247, 964982224, 964983815, 964982931, 
                     964981179, 964982125, 964983653, 964982400, 964982176]
    }
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    # Save datasets
    os.makedirs('data', exist_ok=True)
    movies_df.to_csv('data/movies.csv', index=False)
    ratings_df.to_csv('data/ratings.csv', index=False)
    
    print("Sample dataset created.")
    print(f"Movies: {len(movies_df)}")
    print(f"Ratings: {len(ratings_df)}")
    
    return movies_df, ratings_df

if __name__ == "__main__":
    download_movielens_dataset()
