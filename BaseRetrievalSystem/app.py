from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import argparse
from pathlib import Path

app = Flask(__name__)

# Global variables to store data
query_data = []
article_urls = {}
image_to_article = {}
database_dir = ""
query_dir = ""

def load_data(results_file, article_url_file, image_article_file, db_dir, q_dir):
    """Load all required data"""
    global query_data, article_urls, image_to_article, database_dir, query_dir
    
    database_dir = db_dir
    query_dir = q_dir
    
    # Load retrieval results
    with open(results_file, 'r') as f:
        query_data = json.load(f)
    
    # Load article URLs
    with open(article_url_file, 'r') as f:
        article_urls = json.load(f)
    
    # Load image to article mapping
    with open(image_article_file, 'r') as f:
        image_to_article = json.load(f)
    
    print(f"✓ Loaded {len(query_data)} queries")
    print(f"✓ Loaded {len(article_urls)} article URLs")
    print(f"✓ Loaded {len(image_to_article)} image mappings")

@app.route('/')
def index():
    """Main page with query navigation"""
    if not query_data:
        return "No data loaded. Please run with --results_file argument."
    
    return render_template('index.html', 
                         total_queries=len(query_data),
                         query_dir=query_dir)

@app.route('/api/query/<int:query_index>')
def get_query_data(query_index):
    """Get data for specific query"""
    if query_index >= len(query_data):
        return jsonify({"error": "Query index out of range"})
    
    query = query_data[query_index]
    
    # Add URLs to results
    results_with_urls = []
    for result in query['results']:
        article_id = result['article_id']
        url = article_urls.get(article_id, "#")
        
        result_data = {
            **result,
            'url': url,
            'image_path': f"/static/database/{result['image_id']}.jpg"
        }
        results_with_urls.append(result_data)
    
    return jsonify({
        'query_id': query['query_id'],
        'query_image': f"/static/query/{query['query_id']}.jpg",
        'results': results_with_urls,
        'expected_articles': query.get('expected_articles', [])
    })

@app.route('/api/queries')
def get_all_queries():
    """Get list of all query IDs"""
    queries = [{'index': i, 'query_id': q['query_id']} for i, q in enumerate(query_data)]
    return jsonify(queries)

@app.route('/static/database/<path:filename>')
def serve_database_image(filename):
    """Serve database images"""
    return send_from_directory(database_dir, filename)

@app.route('/static/query/<path:filename>')
def serve_query_image(filename):
    """Serve query images"""
    return send_from_directory(query_dir, filename)

def main():
    parser = argparse.ArgumentParser(description='Retrieval Results Visualization Web App')
    parser.add_argument('--database_dir', type=str, default='./data/database', 
                       help='Directory containing database images')
    parser.add_argument('--query_dir', type=str, default='./data/query', 
                       help='Directory containing query images')
    parser.add_argument('--results_file', type=str, default='retrieval_results.json',
                       help='JSON file with retrieval results')
    parser.add_argument('--article_url_file', type=str, default='database_article_to_url.json',
                       help='JSON file mapping article IDs to URLs')
    parser.add_argument('--image_article_file', type=str, default='database_images_to_article_v.0.1.json',
                       help='JSON file mapping image IDs to article IDs')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Check if files exist
    required_files = [args.results_file, args.article_url_file, args.image_article_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return
    
    # Setup static directories
    os.makedirs('static/database', exist_ok=True)
    os.makedirs('static/query', exist_ok=True)
    
    # Create symbolic links to image directories
    if os.path.exists(args.database_dir):
        print(f"Database images: {args.database_dir}")
    if os.path.exists(args.query_dir):
        print(f"Query images: {args.query_dir}")
    
    # Load data
    load_data(args.results_file, args.article_url_file, args.image_article_file, 
              args.database_dir, args.query_dir)
    
    print(f"\n🚀 Starting web app at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 