# E-Commerce Recommendation Engine

A hybrid machine learning backend for an e-commerce platform. It combines offline collaborative filtering, real-time content-based analysis, and sequential Markov Chain modeling to suggest products based on a user's long-term history, their immediate browsing context, and their sequential navigation behavior.

<!-- Insert preview GIF or screenshot of the recommendations UI here -->
![Preview Placeholder](https://via.placeholder.com/800x400?text=Preview+Image+Placeholder)

## Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **Machine Learning:** scikit-learn, pandas, numpy
- **Database:** MongoDB (PyMongo)
- **Infrastructure:** Render (API hosting), GitHub Actions (Cron scheduling)

## Features

- **Multi-Algorithm Hybrid:** Merges historical collaborative data (SVD), real-time session context (TF-IDF), and sequential navigation probability (Markov Chains).
- **Offline Processing:** Runs heavy matrix and transition calculations during low-traffic hours to save API compute costs while ensuring lightning-fast inference.
- **A/B Testing Architecture:** Provides versioned endpoints (V1 and V2) designed for side-by-side frontend A/B testing.
- **Automated Pipeline:** GitHub Actions handles the daily batch job, bypassing paid scheduling limits on hosting platforms.
- **Evaluation Framework:** Includes a robust evaluation script to benchmark algorithm Hit Rate (HR@K) and NDCG against chronological hold-out validation sets.

## Machine Learning Details

This system handles the cold-start problem, extreme data sparsity, and sequential intent by layering multiple algorithms into a strict cascading fallback logic.

### 1. Collaborative Filtering (Batch Job)
We use **TruncatedSVD (Matrix Factorization)** to find hidden patterns in long-term user behavior. 

**Data used:** The `analyticsevents` MongoDB collection, assigning weighted scores to actions (e.g., Purchase = 5, Cart = 3, Click = 1).
**How it works:** We build a user-item interaction matrix. SVD reduces the dimensions of this matrix, smoothing out the missing data and identifying latent similarities between users with shared tastes. 
**Why this algorithm:** E-commerce interaction matrices are heavily sparse. SVD handles this well, but it generally predicts a user's *overall* taste rather than their immediate intent.

### 2. Sequential Pattern Learning (Batch Job)
We use a **Markov Chain** to predict the very next action a user will take based on historical transition probabilities.

**Data used:** Chronological sorting of `analyticsevents`.
**How it works:** We calculate the conditional probability of navigating from Product A directly to Product B ($P(B|A)$). We store the top 20 most probable next clicks for every item in the catalog.
**Why this algorithm:** SVD struggles with "Next-Item" prediction because it ignores time. A Markov Chain specifically models the chronological journey (e.g., viewing a camera body, then immediately viewing a compatible lens), resulting in vastly superior Hit Rates for active sessions.

### 3. Content-Based Filtering (Real-Time)
We use **TF-IDF Vectorization** paired with **Cosine Similarity** to match products based on their descriptions and features.

**Data used:** The `products` MongoDB collection.
**How it works:** When the server starts, it combines `category`, `brandName`, and `features` into a single text document per product. TF-IDF turns this text into numerical vectors. The API calculates cosine similarity between these vectors in memory.
**Why this algorithm:** It is extremely fast and provides immediate context. If a user with no history clicks a brand-new item, the collaborative algorithms have nothing to work with, but the content filter immediately suggests compatible products.

## Architecture

The system serves requests from a Next.js frontend via versioned endpoints:

- `GET /api/recommend/<user_id>` (V1 Engine): Interleaves the SVD collaborative filter with the real-time TF-IDF engine.
- `GET /api/recommend/v2/<user_id>` (V2 Engine): Implements a strict cascading logic for maximum relevance: 
  1. Looks for the next probable step via **Markov Chain**.
  2. Backfills with **NLP Content Similarity**.
  3. Falls back to **SVD History**.
  4. Finally fills remaining slots with global **Popularity**.

## Setup

1. Clone the repository and navigate to the directory.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env.local` file with your database credentials:
   ```env
   MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/
   DB_NAME=ecommerce_db
   ```
4. Populate the database with initial, statistically sound interaction data:
   ```bash
   python seed_dummy_data.py
   ```
5. Run the offline batch job to compute the SVD and Markov matrices:
   ```bash
   python batch_job.py
   ```
6. Start the Flask API:
   ```bash
   python app.py
   ```

## Future Improvements

- **Recurrent Neural Networks (RNNs):** Upgrade the sequential logic from a first-order Markov Chain to an RNN (e.g., GRU4Rec) to capture deeper temporal dependencies across an entire session rather than just the single previous click.
- **Attention Mechanisms (SASRec):** Implement Self-Attentive Sequential Recommendation to dynamically weigh which historical interactions in a session are most relevant to the next click.
- **Real-Time Model Updates:** Move away from nightly batch jobs to incremental matrix updates (streaming architecture) for immediate personalization.
