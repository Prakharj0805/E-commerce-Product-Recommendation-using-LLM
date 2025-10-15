from flask import Flask, render_template, request, jsonify
from model.recommender import (
    recommend,
    products,
    smart_search,
    generate_comparison,
    explain_recommendation
)

app = Flask(__name__)


@app.route('/')
def home():
    product_names = products['Name'].tolist()
    return render_template('index.html', product_names=product_names)


@app.route('/recommend', methods=['POST'])
def recommend_products():
    product_name = request.form['product_name']
    use_ai_search = request.form.get('use_ai_search', 'off') == 'on'

    # Use smart search if enabled, otherwise use exact match
    if use_ai_search:
        results = smart_search(product_name)
    else:
        results = recommend(product_name)

    if results.empty:
        return render_template('results.html',
                               product_name=product_name,
                               recommendations=None,
                               explanation="",
                               comparison="")

    # Generate AI insights
    explanation = explain_recommendation(product_name, results)
    comparison = generate_comparison(results['Name'].tolist())

    return render_template('results.html',
                           product_name=product_name,
                           recommendations=results.to_dict(orient='records'),
                           explanation=explanation,
                           comparison=comparison)


@app.route('/api/smart-search', methods=['POST'])
def api_smart_search():
    """API endpoint for AJAX smart search"""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    results = smart_search(query)

    if results.empty:
        return jsonify({'results': [], 'message': 'No products found'})

    return jsonify({'results': results.to_dict(orient='records')})


if __name__ == '__main__':
    app.run(debug=True)