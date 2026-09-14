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
DB_NAME = os.getenv('DB_NAME', 'ecommerce_db')

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

@app.route('/api/recommend/v2/<user_id>', methods=['GET'])
def get_recommendations_v2(user_id):
    """
    V2 Recommendation Engine: Sequential (Markov Chain) + Hybrid Fallback
    """
    current_product_id = request.args.get('current_product_id')
    final_recommendations = []
    
    # 1. Primary Engine: Sequential Markov Chain
    if current_product_id:
        print(f"[V2 Engine] Looking up Markov transitions for product: {current_product_id}")
        markov_record = db.markov_transitions.find_one({"product_id": current_product_id})
        
        if markov_record and "transitions" in markov_record:
            # Add top sequential transitions
            for transition in markov_record["transitions"]:
                final_recommendations.append(transition["next_product_id"])
        else:
            print(f"[V2 Engine] Cold Start: No Markov transitions found. Falling back to Content similarity.")
            
    # 2. Context Fallback: Content-Based Engine
    # If Markov didn't find enough items (e.g., brand new item), use NLP similarity
    if current_product_id and len(final_recommendations) < 4:
        content_recs = get_content_recommendations(current_product_id, top_n=6)
        for p_id in content_recs:
            if p_id not in final_recommendations:
                final_recommendations.append(p_id)
                
    # 3. Global History Fallback: SVD Collaborative Filtering
    # Backfill the rest with the user's general static preferences
    batch_record = db.precomputed_recs.find_one({"user_id": user_id})
    if batch_record and "recommended_product_ids" in batch_record:
        batch_recs = batch_record["recommended_product_ids"]
        for p_id in batch_recs:
            if p_id not in final_recommendations:
                final_recommendations.append(p_id)
                
    # 4. Final Fallback: Popularity / Random
    if not final_recommendations:
        random_products = list(db.products.aggregate([{"$sample": {"size": 10}}]))
        final_recommendations = [str(p['_id']) for p in random_products]

    # Limit to top 10 overall
    final_recommendations = final_recommendations[:10]

    return jsonify({
        "user_id": user_id,
        "engine": "v2_markov_hybrid",
        "recommendations": final_recommendations
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
