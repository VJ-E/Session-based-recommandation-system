import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from content_engine import get_content_recommendations, initialize_content_engine

load_dotenv('.env.local')

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for the Next.js frontend

# Fetch MongoDB URI from environment variables
MONGO_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME', 'test')

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"Successfully connected to MongoDB: {DB_NAME}")
    
    # Initialize the NLP content engine on startup
    initialize_content_engine()
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Hybrid Recommendation Engine is running!"}), 200

@app.route('/api/recommend/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    # Check if frontend passed the product the user is currently looking at
    current_product_id = request.args.get('current_product_id')
    
    final_recommendations = []
    
    # 1. Real-Time Context (Content-Based)
    if current_product_id:
        print(f"Real-time context detected for product: {current_product_id}")
        content_recs = get_content_recommendations(current_product_id, top_n=4)
        final_recommendations.extend(content_recs)
        
    # 2. Historical Batch Job (Collaborative Filtering)
    batch_record = db.precomputed_recs.find_one({"user_id": user_id})
    if batch_record and "recommended_product_ids" in batch_record:
        batch_recs = batch_record["recommended_product_ids"]
        
        # Add batch recs to final list, avoiding duplicates
        for p_id in batch_recs:
            if p_id not in final_recommendations:
                final_recommendations.append(p_id)
                
    # 3. Fallback (If user has no history and isn't looking at a product)
    if not final_recommendations:
        # Just return the most popular products (or random ones) as a cold-start fallback
        # Here we just fetch 5 random products for simplicity
        random_products = list(db.products.aggregate([{"$sample": {"size": 5}}]))
        final_recommendations = [str(p['_id']) for p in random_products]

    # Limit to top 10 overall
    final_recommendations = final_recommendations[:10]

    return jsonify({
        "user_id": user_id,
        "recommendations": final_recommendations
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
