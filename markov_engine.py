import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

def run_markov_job():
    print("Starting Batch Job for Markov Chain Transitions...")
    load_dotenv('.env.local')
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DB_NAME', 'ecommerce_db')]
    
    # 1. Fetch interactions sorted by timestamp
    cursor = db.analyticsevents.find(
        {"userId": {"$exists": True, "$ne": None}, "eventData.productId": {"$exists": True}, "timestamp": {"$exists": True}},
        {"userId": 1, "eventData.productId": 1, "timestamp": 1}
    ).sort("timestamp", 1)
    
    data = []
    for doc in cursor:
        data.append({
            "user_id": str(doc["userId"]),
            "product_id": str(doc["eventData"]["productId"]),
            "timestamp": doc["timestamp"]
        })
        
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No interaction data found for Markov job.")
        return
        
    print(f"Fetched {len(df)} chronological interactions. Building Transition Matrix...")
    
    # 2. Build Transitions
    transition_matrix = {} # {item_A: {item_B: count}}
    
    user_sequences = df.sort_values(by=['user_id', 'timestamp']).groupby('user_id')['product_id'].apply(list)
    
    for seq in user_sequences:
        for i in range(len(seq) - 1):
            current_item = seq[i]
            next_item = seq[i+1]
            
            if current_item not in transition_matrix:
                transition_matrix[current_item] = {}
            if next_item not in transition_matrix[current_item]:
                transition_matrix[current_item][next_item] = 0
                
            transition_matrix[current_item][next_item] += 1
            
    # 3. Save to MongoDB
    markov_coll = db.markov_transitions
    markov_coll.delete_many({}) # Clear old batch data
    
    docs_to_insert = []
    for current_item, transitions in transition_matrix.items():
        # Sort transitions by highest count first
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        # We only need top ~20 transitions per item to save space
        top_transitions = [{"next_product_id": p_id, "count": count} for p_id, count in sorted_transitions[:20]]
        
        docs_to_insert.append({
            "product_id": current_item,
            "transitions": top_transitions
        })
        
    if docs_to_insert:
        markov_coll.insert_many(docs_to_insert)
        print(f"Markov Job Complete! Saved transitions for {len(docs_to_insert)} unique products.")

if __name__ == "__main__":
    run_markov_job()
