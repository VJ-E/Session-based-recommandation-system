import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.decomposition import TruncatedSVD

def run_batch_job():
    print("Starting Batch Job for Collaborative Filtering...")
    load_dotenv('.env.local')
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DB_NAME', 'ecommerce_db')]
    
    # 1. Fetch interactions
    # Map implicit feedback (clicks, add to cart, purchase) to a score
    # Fetch from 'analyticsevents' collection where userId and productId exist
    pipeline = [
        {
            "$match": {
                "userId": {"$exists": True, "$ne": None},
                "eventData.productId": {"$exists": True}
            }
        },
        {
            "$project": {
                "user_id": {"$toString": "$userId"},
                "product_id": {"$toString": "$eventData.productId"},
                "score": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$eventType", "purchase"]}, "then": 5},
                            {"case": {"$eq": ["$eventType", "add_to_cart"]}, "then": 3},
                            {"case": {"$eq": ["$eventType", "click_product"]}, "then": 1}
                        ],
                        "default": 0
                    }
                }
            }
        },
        {
            "$group": {
                "_id": {"user_id": "$user_id", "product_id": "$product_id"},
                "score": {"$sum": "$score"} # Sum scores if a user interacted multiple times
            }
        }
    ]
    
    interactions_cursor = db.analyticsevents.aggregate(pipeline)
    
    # Flatten the grouped data
    data = []
    for doc in interactions_cursor:
        if doc["score"] > 0:
            data.append({
                "user_id": doc["_id"]["user_id"],
                "product_id": doc["_id"]["product_id"],
                "score": doc["score"]
            })
            
    df = pd.DataFrame(data)

    if df.empty:
        print("No interaction data found. Please interact with the app (click, add to cart, purchase) to generate data.")
        return
        
    print(f"Fetched {len(df)} user-product interactions. Building User-Item Matrix...")
    
    # 2. Build User-Item Matrix
    # We create a grid where rows are users, columns are products, and values are the interaction scores
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='score').fillna(0)
    
    # 3. Apply Collaborative Filtering (SVD Matrix Factorization)
    n_components = min(20, user_item_matrix.shape[1] - 1)
    if n_components <= 0:
        print("Not enough unique products to run SVD.")
        return
        
    svd = TruncatedSVD(n_components=n_components)
    matrix_svd = svd.fit_transform(user_item_matrix)
    predicted_matrix = svd.inverse_transform(matrix_svd)
    
    predicted_df = pd.DataFrame(predicted_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    # 4. Save Top 10 Recommendations per User to MongoDB
    precomputed_recs = db.precomputed_recs
    precomputed_recs.delete_many({}) # Clear old batch data
    
    recs_to_insert = []
    
    for user_id in predicted_df.index:
        user_scores = predicted_df.loc[user_id]
        
        # We don't want to recommend items the user has already interacted with
        already_interacted = df[df['user_id'] == user_id]['product_id'].tolist()
        user_scores = user_scores.drop(labels=already_interacted, errors='ignore')
        
        # Get the top 10 highest-scored products
        top_products = user_scores.nlargest(10).index.tolist()
        
        recs_to_insert.append({
            "user_id": user_id,
            "recommended_product_ids": top_products
        })
        
    if recs_to_insert:
        precomputed_recs.insert_many(recs_to_insert)
        print(f"Batch Job Complete! Saved precomputed recommendations for {len(recs_to_insert)} users.")

if __name__ == "__main__":
    run_batch_job()
