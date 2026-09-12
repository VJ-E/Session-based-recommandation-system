import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global variables to cache the model in memory
_product_df = None
_cosine_sim_matrix = None
_indices = None

def initialize_content_engine():
    global _product_df, _cosine_sim_matrix, _indices
    print("Initializing Real-Time Content Engine...")
    
    load_dotenv('.env.local')
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DB_NAME', 'test')]
    
    # Fetch products with text fields
    products_cursor = db.products.find({}, {"_id": 1, "name": 1, "category": 1, "brandName": 1, "features": 1})
    _product_df = pd.DataFrame(list(products_cursor))
    
    if _product_df.empty:
        print("No products found for content engine.")
        return
        
    _product_df['_id'] = _product_df['_id'].astype(str)
    
    # Combine relevant text fields into a single 'soup' string for NLP
    def create_soup(row):
        features = ' '.join(row.get('features', [])) if isinstance(row.get('features'), list) else str(row.get('features', ''))
        category = str(row.get('category', ''))
        brand = str(row.get('brandName', ''))
        return f"{category} {brand} {features}"
        
    _product_df['soup'] = _product_df.apply(create_soup, axis=1)
    
    # Calculate TF-IDF Matrix (converting words to numbers)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(_product_df['soup'])
    
    # Calculate cosine similarity between all products
    _cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a reverse mapping of indices and product IDs
    _indices = pd.Series(_product_df.index, index=_product_df['_id']).drop_duplicates()
    print(f"Content Engine Initialized for {len(_product_df)} products.")

def get_content_recommendations(product_id, top_n=5):
    """Returns top N similar product IDs based on content"""
    global _product_df, _cosine_sim_matrix, _indices
    
    if _cosine_sim_matrix is None:
        initialize_content_engine()
        
    if product_id not in _indices:
        return []
        
    idx = _indices[product_id]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(_cosine_sim_matrix[idx]))
    
    # Sort products based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of the top N most similar products (skip index 0 because it's the product itself)
    sim_scores = sim_scores[1:top_n+1]
    
    product_indices = [i[0] for i in sim_scores]
    
    # Return the actual MongoDB Object IDs
    return _product_df.iloc[product_indices]['_id'].tolist()
