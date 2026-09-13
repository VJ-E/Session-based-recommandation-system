import os
import random
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime, timedelta
import numpy as np

def seed_realistic_data():
    print("Initializing Database Connection...")
    load_dotenv('.env.local')
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DB_NAME', 'ecommerce_db')]
    
    print("Clearing old synthetic data...")
    db.analyticsevents.delete_many({"is_dummy_data": True})
    
    products = list(db.products.find({}, {"_id": 1}))
    if not products:
        print("ERROR: No products found.")
        return
        
    product_ids = [p['_id'] for p in products]
    num_products = len(product_ids)
    
    # 1. Popularity Skew (Zipf/Pareto distribution)
    # A small percentage of products will be highly popular.
    popularity_weights = np.random.pareto(a=1.5, size=num_products)
    popularity_weights /= popularity_weights.sum() # Normalize to sum to 1
    
    # 2. Artificial Categories (Affinities)
    # Group products randomly into 5 'clusters' or 'categories'
    num_clusters = 5
    product_clusters = {}
    cluster_catalogs = {i: [] for i in range(num_clusters)}
    
    for pid in product_ids:
        cluster = random.randint(0, num_clusters - 1)
        product_clusters[pid] = cluster
        cluster_catalogs[cluster].append(pid)
        
    print(f"Grouped {num_products} products into {num_clusters} latent clusters.")
    
    # 3. Generate Users & Sequential Interactions
    num_users = 200 # More users for a better dataset
    dummy_events = []
    event_types = ['click_product', 'add_to_cart', 'purchase']
    event_weights = [0.80, 0.15, 0.05] 
    
    base_time = datetime.utcnow() - timedelta(days=30)
    
    print(f"Generating realistic sequential interactions for {num_users} users...")
    
    for user_idx in range(num_users):
        user_id = ObjectId()
        # Assign a primary cluster affinity to the user
        preferred_cluster = random.randint(0, num_clusters - 1)
        
        # Determine sequence length (some short sessions, some long)
        seq_length = int(np.clip(np.random.normal(loc=8, scale=4), a_min=3, a_max=20))
        
        user_time = base_time + timedelta(days=random.randint(0, 25))
        
        current_item = None
        
        for step in range(seq_length):
            # Sequential Transition Logic
            # 70% chance to stay in the preferred cluster, 30% chance to pick purely by popularity
            if random.random() < 0.70 and len(cluster_catalogs[preferred_cluster]) > 0:
                # Pick an item from the preferred cluster
                catalog = cluster_catalogs[preferred_cluster]
                # To add Markov logic, we can also make them repeat items occasionally (e.g. click -> cart)
                if current_item and random.random() < 0.3:
                    chosen_product = current_item
                else:
                    chosen_product = random.choice(catalog)
            else:
                # Pick globally based on popularity weights
                chosen_product = np.random.choice(product_ids, p=popularity_weights)
                
            current_item = chosen_product
            
            # Action type (click, cart, purchase)
            # If they just clicked it, they might cart/buy it next step, but for simplicity we keep it random per step
            e_type = random.choices(event_types, weights=event_weights)[0]
            
            # Increment time slightly (e.g. 10 to 120 seconds between clicks)
            user_time += timedelta(seconds=random.randint(10, 120))
            
            event = {
                "userId": user_id,
                "sessionId": f"simulated_session_{user_idx}",
                "eventType": e_type,
                "eventData": {
                    "productId": chosen_product
                },
                "timestamp": user_time,
                "is_dummy_data": True
            }
            dummy_events.append(event)

    print(f"Generated {len(dummy_events)} events. Inserting into MongoDB...")
    db.analyticsevents.insert_many(dummy_events)
    print("Realistic synthetic data seeded successfully! You can now run evaluator.py.")

if __name__ == "__main__":
    seed_realistic_data()
