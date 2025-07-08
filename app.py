from flask import Flask, request, jsonify, render_template
from price_fetche import get_product_links, get_location_name
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    product = request.form.get('product')
    location = request.form.get('location')

    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return jsonify({"error": "SERPAPI_KEY not set in environment variables"}), 500

    try:
        results = get_product_links(product, location, serpapi_key, use_llm=False)
        # results = [item for item in results if item["price"] is not None]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
