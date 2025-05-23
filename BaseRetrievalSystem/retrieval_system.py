import os
import json
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
import time
import argparse
import uuid
import hashlib

def load_vector_from_npy(file_path):
    """Load vector from .npy file"""
    return np.load(file_path).astype(np.float32)

def get_image_id_from_filename(filename):
    """Extract image ID from filename by removing .jpg.npy or .npy extension"""
    basename = os.path.basename(filename)
    
    # Remove .jpg.npy first, then .npy if not found
    if basename.endswith('.jpg.npy'):
        return basename.replace('.jpg.npy', '')
    elif basename.endswith('.npy'):
        return basename.replace('.npy', '')
    else:
        return basename

def load_image_to_article_mapping(mapping_file):
    """Load image to article ID mapping from CSV"""
    df = pd.read_csv(mapping_file)
    
    # The CSV has query_id and multiple article_id columns (article_id_1 to article_id_10)
    # We'll create a mapping from query_id to a list of article_ids
    mapping = {}
    
    for _, row in df.iterrows():
        query_id = row['query_id']
        article_ids = []
        
        # Collect all article_id columns
        for i in range(1, 11):  # article_id_1 to article_id_10
            col_name = f'article_id_{i}'
            if col_name in df.columns and pd.notna(row[col_name]):
                article_ids.append(row[col_name])
        
        mapping[query_id] = article_ids
    
    return mapping

def load_image_to_article_mapping_json(json_file):
    """Load image to article ID mapping from JSON file"""
    with open(json_file, 'r') as f:
        mapping = json.load(f)
    return mapping

def validate_directories(vectors_dir, query_dir):
    """Validate that required directories exist"""
    if not os.path.exists(vectors_dir):
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")
    if not os.path.exists(query_dir):
        raise FileNotFoundError(f"Query directory not found: {query_dir}")
    
    # Check if directories contain .npy files
    vector_files = [f for f in os.listdir(vectors_dir) if f.endswith('.npy')]
    query_files = [f for f in os.listdir(query_dir) if f.endswith('.npy')]
    
    if not vector_files:
        raise ValueError(f"No .npy files found in vectors directory: {vectors_dir}")
    if not query_files:
        raise ValueError(f"No .npy files found in query directory: {query_dir}")
    
    print(f"✓ Found {len(vector_files)} vector files in {vectors_dir}")
    print(f"✓ Found {len(query_files)} query files in {query_dir}")

def setup_qdrant_collection(client, collection_name, vector_size=512):
    """Setup Qdrant collection"""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            print(f"Collection {collection_name} already exists")
            return True
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,  # CLIP B/16 vector size
                distance=models.Distance.COSINE  # Use cosine similarity
            )
        )
        print(f"Created collection {collection_name}")
        return True
    except Exception as e:
        print(f"Error setting up collection: {e}")
        return False

def upload_vectors_to_qdrant(client, collection_name, vectors_dir):
    """Upload vectors to Qdrant"""
    files = [f for f in os.listdir(vectors_dir) if f.endswith('.npy')]
    
    # Process in batches
    batch_size = 1000
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        ids = []
        vectors = []
        payloads = []
        
        for file in batch_files:
            try:
                file_path = os.path.join(vectors_dir, file)
                vector = load_vector_from_npy(file_path)
                
                # Extract image ID from filename
                image_id = get_image_id_from_filename(file)
                
                # Convert image ID to UUID
                uuid_id = str(uuid.uuid5(uuid.NAMESPACE_URL, image_id))
                
                ids.append(uuid_id)
                vectors.append(vector)
                payloads.append({"image_id": image_id})
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if vectors:
            # Upload batch
            client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
            )
            
            print(f"Uploaded batch {i//batch_size + 1}, {len(vectors)} vectors")
    
    # Wait for optimization
    time.sleep(5)

def search_top_k(client, collection_name, query_vector, k=10):
    """Search for the top k matching vectors"""
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k
    )
    return results

def main():
    parser = argparse.ArgumentParser(description='Qdrant Vector Database Setup and Query')
    parser.add_argument('--host', type=str, default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--vector_size', type=int, default=None, help='Vector dimensions (auto-detect if not specified)')
    parser.add_argument('--vectors_dir', type=str, default=None, help='Directory containing vector files')
    parser.add_argument('--query_dir', type=str, required=True, help='Directory containing query vector files')
    parser.add_argument('--mapping_file', type=str, default='Rezero_Slavery_EVENTA2025_Track1.csv', help='File with image to article mapping')
    parser.add_argument('--json_mapping_file', type=str, default='database_images_to_article_v.0.1.json', help='JSON file with image to article mapping')
    parser.add_argument('--caption_file', type=str, default=None, help='CSV file containing captions to add to output')
    parser.add_argument('--collection', type=str, default='clip_vectors', help='Qdrant collection name')
    parser.add_argument('--skip_upload', action='store_true', help='Skip uploading vectors if collection already exists')
    parser.add_argument('--action', type=str, choices=['setup', 'query', 'all'], default='all', help='Action to perform')
    
    args = parser.parse_args()
    
    # Validate directories
    validate_directories(args.vectors_dir, args.query_dir)
    
    # Initialize Qdrant client
    client = QdrantClient(host=args.host, port=args.port)
    
    # Setup collection
    if args.action in ['setup', 'all']:
        # Determine vector size
        if args.vector_size:
            vector_size = args.vector_size
            print(f"Using specified vector size: {vector_size}")
        else:
            # Auto-detect from first file
            vector_sample = load_vector_from_npy(os.path.join(args.vectors_dir, os.listdir(args.vectors_dir)[0]))
            vector_size = vector_sample.shape[0]
            print(f"Auto-detected vector size: {vector_size}")
        
        if setup_qdrant_collection(client, args.collection, vector_size):
            # Check if we should skip upload
            if args.skip_upload:
                try:
                    # Check if collection has data
                    collection_info = client.get_collection(args.collection)
                    if collection_info.points_count > 0:
                        print(f"✓ Skipping upload - Collection '{args.collection}' already has {collection_info.points_count} vectors")
                    else:
                        print("Collection exists but is empty, uploading vectors...")
                        upload_vectors_to_qdrant(client, args.collection, args.vectors_dir)
                except Exception as e:
                    print(f"Error checking collection info: {e}")
                    print("Uploading vectors...")
                    upload_vectors_to_qdrant(client, args.collection, args.vectors_dir)
            else:
                # Upload vectors
                print("Uploading vectors...")
                upload_vectors_to_qdrant(client, args.collection, args.vectors_dir)
    
    # Query
    if args.action in ['query', 'all']:
        # Load query to article mapping (CSV)
        try:
            query_mapping = load_image_to_article_mapping(args.mapping_file)
            print(f"Loaded query mapping with {len(query_mapping)} entries")
        except Exception as e:
            print(f"Error loading query mapping: {e}")
            query_mapping = {}
        
        # Load image to article mapping (JSON)
        try:
            image_mapping = load_image_to_article_mapping_json(args.json_mapping_file)
            print(f"Loaded image mapping with {len(image_mapping)} entries")
        except Exception as e:
            print(f"Error loading image mapping: {e}")
            image_mapping = {}
        
        # Process queries
        query_files = [f for f in os.listdir(args.query_dir) if f.endswith('.npy')]
        
        results_list = []
        
        for query_file in query_files:
            query_path = os.path.join(args.query_dir, query_file)
            query_vector = load_vector_from_npy(query_path)
            query_id = get_image_id_from_filename(query_file)
            
            print(f"\nProcessing query {query_id}")
            
            search_results = search_top_k(client, args.collection, query_vector, k=10)
            
            # Format results
            query_results = {
                "query_id": query_id,
                "results": []
            }
            
            for hit in search_results:
                image_id = hit.payload.get("image_id").split(".")[00]
                
                # Get article ID from image mapping
                article_id = image_mapping.get(image_id, "unknown")
                
                result = {
                    "image_id": image_id,
                    "article_id": article_id,
                    "score": hit.score
                }
                query_results["results"].append(result)
                
                print(f"Match: {image_id} → Article: {article_id}, Score: {hit.score:.4f}")
            
            # Add expected relevant articles for this query if available
            if query_id in query_mapping:
                query_results["expected_articles"] = query_mapping[query_id]
            
            results_list.append(query_results)
        
        # Save results
        with open('retrieval_results.json', 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"\n✓ Saved results to retrieval_results.json")
        
        # Convert to CSV format
        print("Converting to CSV submission format...")
        try:
            # Load captions if provided
            caption_mapping = {}
            if args.caption_file and os.path.exists(args.caption_file):
                print(f"Loading captions from {args.caption_file}")
                caption_df = pd.read_csv(args.caption_file)
                
                # Try to find caption column
                caption_col = None
                for col in ['generated_caption', 'caption', 'text', 'description']:
                    if col in caption_df.columns:
                        caption_col = col
                        break
                
                if caption_col:
                    query_id_col = 'query_id' if 'query_id' in caption_df.columns else caption_df.columns[0]
                    for _, row in caption_df.iterrows():
                        query_id = row[query_id_col]
                        caption = row[caption_col] if pd.notna(row[caption_col]) else ""
                        caption_mapping[query_id] = caption
                    print(f"✓ Loaded {len(caption_mapping)} captions")
            
            csv_data = []
            for query_result in results_list:
                query_id = query_result['query_id']
                row = {'query_id': query_id}
                
                # Add article_id_1 to article_id_10
                for i in range(1, 11):
                    col_name = f'article_id_{i}'
                    if i-1 < len(query_result['results']):
                        article_id = query_result['results'][i-1]['article_id']
                        row[col_name] = article_id if article_id != "unknown" else ""
                    else:
                        row[col_name] = ""
                
                # Add generated_caption
                if caption_mapping and query_id in caption_mapping:
                    row['generated_caption'] = caption_mapping[query_id]
                else:
                    row['generated_caption'] = ""
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            columns = ['query_id'] + [f'article_id_{i}' for i in range(1, 11)] + ['generated_caption']
            df = df[columns]
            df.to_csv('submission.csv', index=False)
            
            # Show statistics
            queries_with_captions = sum(1 for row in csv_data if row['generated_caption'] != "")
            print(f"✓ Saved submission format to submission.csv")
            if args.caption_file:
                print(f"✓ Added captions to {queries_with_captions}/{len(csv_data)} queries")
            
        except Exception as e:
            print(f"Warning: Could not create CSV format: {e}")

if __name__ == "__main__":
    main() 