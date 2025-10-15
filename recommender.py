import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with better error handling
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY not found in environment variables")
    print("AI features will be disabled. Please add GROQ_API_KEY to your .env file")
    groq_client = None
else:
    groq_client = Groq(api_key=api_key)
    print("✓ Groq API initialized successfully")

# Load dataset
products = pd.read_csv("clean_data.csv")

# Fill NaN values
for col in ['Category', 'Brand', 'Name', 'Description', 'Tags']:
    products[col] = products[col].fillna('')

# Combine important text fields
products['combined'] = (
        products['Name'] + ' ' +
        products['Category'] + ' ' +
        products['Brand'] + ' ' +
        products['Description'] + ' ' +
        products['Tags']
)

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(products['combined'])
similarity_matrix = cosine_similarity(tfidf_matrix)


def recommend(product_name, top_n=5):
    """Get product recommendations based on similarity"""
    if product_name not in products['Name'].values:
        return pd.DataFrame()

    idx = products[products['Name'] == product_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    rec_indices = [i[0] for i in scores]

    return products.iloc[rec_indices][['Name', 'Category', 'Brand', 'Rating', 'ImageURL']]


def smart_search(query, top_n=5):
    """Use Groq to understand natural language queries and find matching products"""
    if groq_client is None:
        print("Groq client not initialized, falling back to basic search")
        # Fallback: search in product names
        matches = products[products['Name'].str.contains(query, case=False, na=False)]
        if not matches.empty:
            return recommend(matches.iloc[0]['Name'], top_n)
        return pd.DataFrame()

    try:
        # Get product list for context
        product_list = products[['Name', 'Category', 'Brand']].head(100).to_string()

        prompt = f"""Given this user query: "{query}"

Here are some available products:
{product_list}

Based on the query, identify the most relevant product name that matches what the user is looking for.
Return ONLY the exact product name from the list, nothing else."""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=100
        )

        suggested_product = chat_completion.choices[0].message.content.strip()

        # Try to find the product
        if suggested_product in products['Name'].values:
            return recommend(suggested_product, top_n)

        # Fallback: search in product names
        matches = products[products['Name'].str.contains(query, case=False, na=False)]
        if not matches.empty:
            return recommend(matches.iloc[0]['Name'], top_n)

        return pd.DataFrame()

    except Exception as e:
        print(f"Groq API error: {e}")
        # Fallback to basic search
        matches = products[products['Name'].str.contains(query, case=False, na=False)]
        if not matches.empty:
            return recommend(matches.iloc[0]['Name'], top_n)
        return pd.DataFrame()


def generate_comparison(product_names):
    """Generate AI-powered comparison of recommended products"""
    if groq_client is None:
        return ""

    try:
        # Get product details
        product_details = []
        for name in product_names:
            prod = products[products['Name'] == name]
            if not prod.empty:
                prod_info = prod.iloc[0]
                product_details.append(
                    f"- {prod_info['Name']}: {prod_info['Category']}, Brand: {prod_info['Brand']}, Rating: {prod_info['Rating']}")

        if not product_details:
            return ""

        prompt = f"""Compare these products and provide a brief, helpful summary (2-3 sentences) highlighting key differences:

{chr(10).join(product_details)}

Keep it concise and consumer-friendly."""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=200
        )

        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq API error: {e}")
        return ""


def explain_recommendation(original_product, recommended_products):
    """Generate explanation for why products were recommended"""
    if groq_client is None:
        return ""

    try:
        orig = products[products['Name'] == original_product].iloc[0]
        rec_names = recommended_products['Name'].tolist()[:3]  # Top 3

        prompt = f"""Briefly explain (1-2 sentences) why these products are recommended for someone interested in "{original_product}" ({orig['Category']}, {orig['Brand']}):

Recommendations: {', '.join(rec_names)}

Be specific and helpful."""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=150
        )

        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq API error: {e}")
        return ""