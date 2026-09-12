import os
import random
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId

def seed_data():
    print("Initializing Database Connection...")
    load_dotenv('.env.local')
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DB_NAME', 'ecommerce_db')]
    
    print("Fetching real products from database...")
    # Fetch all product IDs
    products = list(db.products.find({}, {"_id": 1}))
    
    if not products:
        print("ERROR: No products found in your database. Please add some products to your app first.")
        return
        
    product_ids = [p['_id'] for p in products]
    print(f"Found {len(product_ids)} products. Generating simulated users and interactions...")
    
    # We will simulate 50 users, each making a few random interactions
    dummy_events = []
    event_types = ['click_product', 'add_to_cart', 'purchase']
    # Weighted probabilities: Clicks are very common, purchases are rare
    event_weights = [0.70, 0.20, 0.10] 
    
    for _ in range(50):
        # Create a fake User ID for this simulation
        user_id = ObjectId()
        
        # Each user interacts with 3 to 15 random products
        num_interactions = random.randint(3, 15)
        
        # Pick some random products for this user to interact with
        interacted_products = random.choices(product_ids, k=num_interactions)
        
        for p_id in interacted_products:
            e_type = random.choices(event_types, weights=event_weights)[0]
            
            # Construct the AnalyticsEvent matching your schema
            event = {
                "userId": user_id,
                "sessionId": f"simulated_session_{ObjectId()}",
                "eventType": e_type,
                "eventData": {
                    "productId": p_id
                },
                "is_dummy_data": True # Helpful flag in case you want to delete these later
            }
            dummy_events.append(event)
            
    print(f"Generated {len(dummy_events)} interaction events. Inserting into 'analyticsevents' collection...")
    db.analyticsevents.insert_many(dummy_events)
    print("Data seeded successfully! You can now run the batch_job.py script.")

if __name__ == "__main__":
    seed_data()
