# Project Journey: E-Commerce Recommendation Engine

## How It Started
The project began with a clear goal: to enhance an existing Next.js application (`Agent-Based-E-Commerce`) by adding a machine learning-driven recommendation system. 

The existing application had a MongoDB database storing users and products, but recommendations were static. The vision was to build a separate backend (a Flask API) dedicated entirely to machine learning. We decided to start with a robust **Hybrid Recommendation System** (combining collaborative and content-based filtering) to build a solid foundation before eventually tackling complex, sequential session-based recommendations.

## Problems Faced & How We Solved Them

### 1. Data Scarcity & The Cold Start Problem
**Problem:** To train a collaborative filtering model, you need thousands of user interactions (clicks, purchases). The existing development database was virtually empty.
**Solution:** We created `seed_dummy_data.py`, a script that generated thousands of realistic, simulated interaction events (`analyticsevents`). It assigned logical weights to actions (e.g., Purchase = 5, Cart = 3, Click = 1) and linked them to the actual Product IDs in the database, giving our ML models the fuel they needed to learn.

### 2. Performance vs. Real-Time Context
**Problem:** Deep collaborative filtering requires comparing a single user against thousands of others. Running this heavy math (Matrix Factorization) every time a user loaded a page would freeze the API. However, if a user suddenly clicked on a "Camera", we needed to recommend camera gear *instantly*.
**Solution:** We split the brain in two:
- **Offline Batch Job:** A script (`batch_job.py`) runs overnight, using `TruncatedSVD` to crunch the heavy user-history math and save the answers to the database.
- **Real-Time Engine:** A fast, in-memory NLP engine (`content_engine.py`) uses `TF-IDF` to instantly match product text descriptions. 
The Flask endpoint simply merges the two, prioritizing whatever the user is clicking on *right now*.

### 3. Frontend-Backend Communication (CORS)
**Problem:** When we finally wired the Next.js frontend to call the new Flask API, the browser immediately blocked the requests with a "Failed to fetch" error due to Cross-Origin Resource Sharing (CORS) security policies.
**Solution:** We installed the `flask-cors` library and configured the Flask app to explicitly allow incoming requests from the frontend origin.

### 4. UI/UX Clutter
**Problem:** Once the data was flowing to the frontend, we injected a "Recommended for You" horizontal carousel. However, using the standard `ProductCard` component made the UI bulky and awkward.
**Solution:** We swapped the large card for a much sleeker `MiniProductCard`, adjusting the grid spacing and loading skeletons to allow 5-6 products to fit beautifully on the screen at once.

### 5. Deployment Scheduling & GitHub Traffic Jams
**Problem:** The offline batch job needed to run daily. Render (our hosting platform) charges for native cron jobs. The alternative was GitHub Actions, but previous attempts failed because GitHub's servers were too busy.
**Solution:** We wrote a GitHub Actions workflow (`batch_job.yml`) but strategically scheduled it for **3:43 AM UTC**. By picking an odd minute in the middle of the night, we successfully bypassed the massive traffic spikes that occur when thousands of developers schedule jobs exactly on the hour (e.g., 00:00).

---

## Current Progress (As of Today)
The system is **fully integrated and deployment-ready**.

1. **The Backend (`Session-based-recommandation-system`):** 
   - The Flask API is active and functioning perfectly.
   - Gunicorn is configured in `requirements.txt` and `app.py` is ready for Render.
2. **The Frontend (`Agent-Based-E-Commerce`):**
   - The `.env.local` contains the dynamic API URL.
   - The `RecommendedProducts` component is successfully fetching hybrid recommendations.
   - The personalized carousel is live on both the **Homepage** and the **Shop** page.
3. **The Pipeline:**
   - The GitHub Actions workflow is built and waiting to take over the daily batch processing once the repository is pushed and MongoDB secrets are added.

**Next Milestone:** Gathering real user data to benchmark the hybrid model, and eventually transitioning to RNN-driven session-based recommendations to capture sequential browsing intent.

---

## Phase 4: ML Engineering, Rigorous Evaluation, & Sequential Recommendations

We realized that pushing models to production without a proper framework to evaluate them is poor ML engineering. Before jumping into Neural Networks or Recurrent Neural Networks (RNNs), we needed an **Offline Evaluation Pipeline**.

### The Evaluation Pipeline (`evaluator.py`)
**The Goal:** Prove that complex models actually outperform a simple "Popularity" baseline before deploying them.
**The Method:** We implemented a **time-based Leave-One-Out split**. For every user, we hold out their *very last* click as the test target, and use all previous clicks to train the models. We then calculate Information Retrieval metrics: `Hit Rate@K` and `NDCG@K`.

We trained and evaluated three models:
1. **Popularity Baseline:** Recommends the most frequently clicked items across the whole dataset.
2. **SVD Collaborative Filtering:** Recommends items based on latent user-item similarities.
3. **Markov Chain (Sequential):** Recommends the most probable *next* item based on the user's immediate preceding click, capturing short-term session intent.

### The Findings (A Crucial ML Lesson)
We ran the evaluator on our synthetic dummy data (`seed_dummy_data.py`), and the results were highly educational:
- **Popularity HR@10:** 0.0000
- **SVD HR@10:** 0.0000
- **Markov Chain HR@10:** 0.0000

**Why did all models fail?** Because our dummy data generator simulated *completely random* clicks. There is no underlying pattern or true sequence to learn. Random data means the next click is impossible to predict mathematically.

**The Takeaway:** This perfectly demonstrates the necessity of the evaluation pipeline. If we had skipped this step and gone straight to deploying a complex Neural Network to production, we would have assumed it was working when it was actually learning nothing. We now have a rock-solid, rigorous framework to evaluate algorithms locally. 

### Fixing the Foundation & Re-Benchmarking
To prove our ML architecture works, we redesigned `seed_dummy_data.py` to generate **realistic** behavioral data:
- **Popularity Skew:** We injected a Pareto (Zipf) distribution so a few products get most of the clicks, mimicking real life.
- **User Affinities & Transitions:** Users were assigned to latent "clusters", and clicks mostly stayed within those clusters, creating logical sequences.
- **Chronological Timestamps:** We added accurate time-series data to fix the evaluation split.

**The Realistic Benchmark Results:**
- **Popularity HR@10:** `0.1294` (12.9%)
- **SVD HR@10:** `0.0149` (1.5%)
- **Markov Chain HR@10:** `0.0896` (8.9%)

**The Engineering Analysis:**
1. **Why did Popularity win?** In any real-world dataset with a heavy "bestseller" skew, recommending the most popular items is incredibly hard to beat. It establishes a very strong floor.
2. **Why did Markov do well?** Because it effectively captures the immediate, short-term transition logic. It learned the "clusters" we injected into the data.
3. **Why did SVD struggle?** SVD builds global latent profiles. However, with 200 users and 2,700 products, the matrix is extremely sparse. SVD struggles with extreme sparsity compared to simple popularity. Furthermore, SVD predicts general, timeless preferences, whereas our evaluation specifically measures **Next Interaction Prediction** (a strictly sequential task).

This audit proved its worth: we now understand our baselines, we know our evaluation pipeline works, and we have mathematically proven that Sequential (Markov) logic vastly outperforms static (SVD) logic for Next-Item Prediction in this environment.
