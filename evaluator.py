import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.decomposition import TruncatedSVD
import warnings

# Suppress some pandas warnings for clean output
warnings.filterwarnings('ignore')

def load_data_from_db():
    load_dotenv('.env.local')
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DB_NAME', 'ecommerce_db')]
    
    # Fetch events sorted by timestamp
    cursor = db.analyticsevents.find(
        {"userId": {"$exists": True, "$ne": None}, "eventData.productId": {"$exists": True}},
        {"userId": 1, "eventData.productId": 1, "timestamp": 1, "eventType": 1}
    ).sort("timestamp", 1)
    
    data = []
    for doc in cursor:
        user_id = str(doc["userId"])
        product_id = str(doc["eventData"]["productId"])
        event_type = doc.get("eventType", "")
        # Score mapping for SVD
        if event_type == "purchase":
            score = 5
        elif event_type == "add_to_cart":
            score = 3
        elif event_type == "click_product":
            score = 1
        else:
            continue
            
        data.append({
            "user_id": user_id,
            "product_id": product_id,
            "score": score,
            "timestamp": doc.get("timestamp")
        })
        
    return pd.DataFrame(data)

def time_based_split(df, min_interactions=3):
    """
    Leave-one-out time-based split.
    For users with >= min_interactions, the last interaction becomes the test target.
    All previous interactions become the training set.
    """
    # Group by user and count interactions
    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    df_valid = df[df['user_id'].isin(valid_users)]
    
    if df_valid.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Sort just to be sure
    df_valid = df_valid.sort_values(by=['user_id', 'timestamp'])
    
    # Test set: the last item for each user
    test_df = df_valid.groupby('user_id').tail(1)
    
    # Train set: everything else
    train_df = df_valid.drop(test_df.index)
    
    return train_df, test_df

def evaluate_hit_rate(predictions_dict, test_df, k=10):
    """
    Calculates Hit Rate @ K.
    predictions_dict: {user_id: [ranked_list_of_product_ids]}
    """
    hits = 0
    total = len(test_df)
    
    for _, row in test_df.iterrows():
        user = row['user_id']
        target = row['product_id']
        
        preds = predictions_dict.get(user, [])[:k]
        if target in preds:
            hits += 1
            
    return hits / total if total > 0 else 0.0

def evaluate_ndcg(predictions_dict, test_df, k=10):
    """
    Calculates NDCG @ K.
    """
    ndcg_sum = 0
    total = len(test_df)
    
    for _, row in test_df.iterrows():
        user = row['user_id']
        target = row['product_id']
        
        preds = predictions_dict.get(user, [])[:k]
        if target in preds:
            rank = preds.index(target) + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)
            
    return ndcg_sum / total if total > 0 else 0.0

# ==========================================
# Models
# ==========================================

class PopularityRecommender:
    def __init__(self):
        self.popular_items = []
        
    def fit(self, train_df):
        # Count frequency of each product
        item_counts = train_df['product_id'].value_counts()
        self.popular_items = item_counts.index.tolist()
        
    def predict(self, user_id, k=10):
        return self.popular_items[:k]


class SVDRecommender:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.user_item_matrix = None
        self.predicted_df = None
        
    def fit(self, train_df):
        # Aggregate scores like in batch_job.py
        grouped = train_df.groupby(['user_id', 'product_id'])['score'].sum().reset_index()
        # Clip max score to 5 to prevent high-frequency clicks from overshadowing purchases
        grouped['score'] = grouped['score'].clip(upper=5.0)
        
        self.user_item_matrix = grouped.pivot_table(index='user_id', columns='product_id', values='score').fillna(0)
        
        n_comp = min(self.n_components, self.user_item_matrix.shape[1] - 1)
        if n_comp <= 0:
            return
            
        svd = TruncatedSVD(n_components=n_comp)
        matrix_svd = svd.fit_transform(self.user_item_matrix)
        predicted_matrix = svd.inverse_transform(matrix_svd)
        
        self.predicted_df = pd.DataFrame(predicted_matrix, index=self.user_item_matrix.index, columns=self.user_item_matrix.columns)
        
    def predict(self, user_id, train_df, k=10):
        if self.predicted_df is None or user_id not in self.predicted_df.index:
            return []
            
        user_scores = self.predicted_df.loc[user_id]
        # Remove items already interacted with in the training set
        already_interacted = train_df[train_df['user_id'] == user_id]['product_id'].tolist()
        user_scores = user_scores.drop(labels=already_interacted, errors='ignore')
        
        return user_scores.nlargest(k).index.tolist()

class MarkovChainRecommender:
    def __init__(self):
        self.transition_matrix = {} # dict of dicts: {item_A: {item_B: count}}
        
    def fit(self, train_df):
        # Build sequences for each user
        user_sequences = train_df.sort_values(by=['user_id', 'timestamp']).groupby('user_id')['product_id'].apply(list)
        
        for seq in user_sequences:
            for i in range(len(seq) - 1):
                current_item = seq[i]
                next_item = seq[i+1]
                
                if current_item not in self.transition_matrix:
                    self.transition_matrix[current_item] = {}
                if next_item not in self.transition_matrix[current_item]:
                    self.transition_matrix[current_item][next_item] = 0
                    
                self.transition_matrix[current_item][next_item] += 1
                
        # We don't necessarily need to normalize to probabilities just for ranking, counts work fine
        
    def predict(self, user_id, train_df, k=10):
        # Get the user's very last item in the training set
        user_history = train_df[train_df['user_id'] == user_id].sort_values(by='timestamp')
        if user_history.empty:
            return []
            
        last_item = user_history.iloc[-1]['product_id']
        
        if last_item not in self.transition_matrix:
            return []
            
        # Get transitions from the last item
        transitions = self.transition_matrix[last_item]
        # Sort by count descending
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        return [item for item, count in sorted_transitions][:k]


# ==========================================
# Main Evaluation Loop
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data_from_db()
    
    if df.empty:
        print("No interaction data found. Please run seed_dummy_data.py first.")
        exit()
        
    print(f"Total interactions loaded: {len(df)}")
    
    train_df, test_df = time_based_split(df, min_interactions=3)
    
    if test_df.empty:
        print("Not enough users with >=3 interactions to form a test set. Generate more data.")
        exit()
        
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    users_to_test = test_df['user_id'].unique()
    
    # 1. Evaluate Popularity
    print("\n--- Training Popularity Baseline ---")
    pop_model = PopularityRecommender()
    pop_model.fit(train_df)
    
    pop_preds = {u: pop_model.predict(u) for u in users_to_test}
    print(f"Popularity HR@10:   {evaluate_hit_rate(pop_preds, test_df, k=10):.4f}")
    print(f"Popularity NDCG@10: {evaluate_ndcg(pop_preds, test_df, k=10):.4f}")
    
    # 2. Evaluate SVD
    print("\n--- Training SVD Collaborative Filtering ---")
    svd_model = SVDRecommender(n_components=20)
    svd_model.fit(train_df)
    
    svd_preds = {u: svd_model.predict(u, train_df) for u in users_to_test}
    print(f"SVD HR@10:          {evaluate_hit_rate(svd_preds, test_df, k=10):.4f}")
    print(f"SVD NDCG@10:        {evaluate_ndcg(svd_preds, test_df, k=10):.4f}")
    
    # 3. Evaluate Markov Chain
    print("\n--- Training Sequential Markov Chain ---")
    mc_model = MarkovChainRecommender()
    mc_model.fit(train_df)
    
    mc_preds = {u: mc_model.predict(u, train_df) for u in users_to_test}
    print(f"Markov Chain HR@10:   {evaluate_hit_rate(mc_preds, test_df, k=10):.4f}")
    print(f"Markov Chain NDCG@10: {evaluate_ndcg(mc_preds, test_df, k=10):.4f}")

    print("\nEvaluation Complete: Target Task was Next Interaction Prediction.")
