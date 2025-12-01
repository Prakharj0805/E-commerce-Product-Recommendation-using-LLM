# E-commerce-Product-Recommendation-using-LLM
These project generally recommend the e commerce products with AI insights

Project Structure




product_recommender/
│
├── app.py
├── model/
│   └── recommender.py
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   └── style.css
└── clean_data.csv



DEMO VIDEO::https://drive.google.com/file/d/1UTxjeUzyk-sIasYGYQFM3Ydj2w-LQOwA/view?usp=sharing


One of the most challenging problems I solved recently was integrating a traditional machine-learning recommendation engine with a modern LLM-based explanation system inside a Flask application. The biggest issue occurred when the project’s SQLAlchemy models did not match the underlying SQLite database schema, causing errors like “no such column: product.product_id”. Debugging this required carefully analyzing every model field, query, and CSV column, then fully rebuilding the database to ensure schema consistency. Another major challenge was designing a hybrid recommender system that combines category matching, star-rating similarity, TF-IDF review similarity, and helpful-vote weighting. After computing recommendations, I integrated the Groq LLM API to generate natural-language explanations for each recommendation, which required stable prompt design and fallback logic if the API failed. The final solution provides not just accurate product recommendations but also clear, AI-generated reasoning, improving transparency and user trust. This was a deeply technical problem but also the most rewarding one, as it brought together database engineering, NLP, ML ranking, and LLM reasoning into one cohesive system.
