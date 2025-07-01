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
    return np.load(file_path).astype(np.float32)

def get_image_id_from_filename(filename):
    basename = os.path.basename(filename)
    
    if basename.endswith('.jpg.npy'):
        return basename.replace('.jpg.npy', '')
    elif basename.endswith('.npy'):
        return basename.replace('.npy', '')
    else:
        return basename



def load_image_to_article_mapping_json(json_file):
    with open(json_file, 'r') as f:
        mapping = json.load(f)
    return mapping

def validate_directories(vectors_dir, query_dir):
    if not os.path.exists(vectors_dir):
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")
    if not os.path.exists(query_dir):
        raise FileNotFoundError(f"Query directory not found: {query_dir}")
    
    vector_files = [f for f in os.listdir(vectors_dir) if f.endswith('.npy')]
    query_files = [f for f in os.listdir(query_dir) if f.endswith('.npy')]
    
    if not vector_files:
        raise ValueError(f"No .npy files found in vectors directory: {vectors_dir}")
    if not query_files:
        raise ValueError(f"No .npy files found in query directory: {query_dir}")
    
    print(f"Found {len(vector_files)} vector files, {len(query_files)} query files")

def setup_qdrant_collection(client, collection_name, vector_size=512):
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            print(f"Collection {collection_name} exists")
            return True
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection {collection_name}")
        return True
    except Exception as e:
        print(f"Error setting up collection: {e}")
        return False

def upload_vectors_to_qdrant(client, collection_name, vectors_dir):
    files = [f for f in os.listdir(vectors_dir) if f.endswith('.npy')]
    
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
                image_id = get_image_id_from_filename(file)
                uuid_id = str(uuid.uuid5(uuid.NAMESPACE_URL, image_id))
                
                ids.append(uuid_id)
                vectors.append(vector)
                payloads.append({"image_id": image_id})
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if vectors:
            client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
            )
            
            print(f"Uploaded batch {i//batch_size + 1}: {len(vectors)} vectors")
    
    time.sleep(5)

def search_top_k(client, collection_name, query_vector, k=20):
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
    parser.add_argument('--vector_size', type=int, default=None, help='Vector dimensions')
    parser.add_argument('--vectors_dir', type=str, default=None, help='Directory containing vector files')
    parser.add_argument('--query_dir', type=str, default=None, help='Directory containing query vector files')
    parser.add_argument('--json_mapping_file', type=str, default='database_images_to_article_v.0.1.json', help='JSON file with image to article mapping')
    parser.add_argument('--caption_file', type=str, default=None, help='CSV file containing captions to add to output')
    parser.add_argument('--collection', type=str, default='my_vectors', help='Qdrant collection name')
    parser.add_argument('--skip_upload', action='store_true', help='Skip uploading vectors if collection already exists')
    parser.add_argument('--action', type=str, choices=['setup', 'query', 'all'], default='all', help='Action to perform')
    
    args = parser.parse_args()
    
    validate_directories(args.vectors_dir, args.query_dir)
    
    client = QdrantClient(host=args.host, port=args.port, timeout=30)
    
    if args.action in ['setup', 'all']:
        if args.vector_size:
            vector_size = args.vector_size
            print(f"Using vector size: {vector_size}")
        else:
            vector_sample = load_vector_from_npy(os.path.join(args.vectors_dir, os.listdir(args.vectors_dir)[0]))
            vector_size = vector_sample.shape[0]
            print(f"Auto-detected vector size: {vector_size}")
        
        if setup_qdrant_collection(client, args.collection, vector_size):
            if args.skip_upload:
                try:
                    collection_info = client.get_collection(args.collection)
                    if collection_info.points_count > 0:
                        print(f"Collection has {collection_info.points_count} vectors, skipping upload")
                    else:
                        print("Collection empty, uploading vectors")
                        upload_vectors_to_qdrant(client, args.collection, args.vectors_dir)
                except Exception as e:
                    print(f"Error checking collection: {e}")
                    upload_vectors_to_qdrant(client, args.collection, args.vectors_dir)
            else:
                print("Uploading vectors")
                upload_vectors_to_qdrant(client, args.collection, args.vectors_dir)
    
    if args.action in ['query', 'all']:
        try:
            image_mapping = load_image_to_article_mapping_json(args.json_mapping_file)
            print(f"Loaded image mapping: {len(image_mapping)} entries")
        except Exception as e:
            print(f"Error loading image mapping: {e}")
            image_mapping = {}
        
        query_files = [f for f in os.listdir(args.query_dir) if f.endswith('.npy')]
        results_list = []
        
        for query_file in query_files:
            query_path = os.path.join(args.query_dir, query_file)
            query_vector = load_vector_from_npy(query_path)
            query_id = get_image_id_from_filename(query_file)
            
            print(f"Processing query {query_id}")
            
            search_results = search_top_k(client, args.collection, query_vector, k=10)
            
            query_results = {
                "query_id": query_id,
                "results": []
            }
            
            for hit in search_results:
                image_id = hit.payload.get("image_id")
                article_id = image_mapping.get(image_id.split("_")[0], "unknown")
                
                result = {
                    "image_id": image_id,
                    "article_id": article_id,
                    "score": hit.score
                }
                query_results["results"].append(result)
                
                print(f"Match: {image_id} -> Article: {article_id}, Score: {hit.score:.4f}")
            
            results_list.append(query_results)
        
        with open('retrieval_results.json', 'w') as f:
            json.dump(results_list, f, indent=2)
        print("Saved results to retrieval_results.json")
        
        print("Converting to CSV submission format")
        try:
            caption_mapping = {}
            if args.caption_file and os.path.exists(args.caption_file):
                print(f"Loading captions from {args.caption_file}")
                caption_df = pd.read_csv(args.caption_file)
                
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
                    print(f"Loaded {len(caption_mapping)} captions")
            
            csv_data = []
            for query_result in results_list:
                query_id = query_result['query_id']
                row = {'query_id': query_id}
                
                for i in range(1, 11):
                    col_name = f'article_id_{i}'
                    if i-1 < len(query_result['results']):
                        article_id = query_result['results'][i-1]['article_id']
                        row[col_name] = article_id if article_id != "unknown" else ""
                    else:
                        row[col_name] = ""
                
                if caption_mapping and query_id in caption_mapping:
                    row['generated_caption'] = caption_mapping[query_id]
                else:
                    row['generated_caption'] = ""
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            columns = ['query_id'] + [f'article_id_{i}' for i in range(1, 11)] + ['generated_caption']
            df = df[columns]
            df.to_csv('submission.csv', index=False)
            
            queries_with_captions = sum(1 for row in csv_data if row['generated_caption'] != "")
            print(f"Saved submission format to submission.csv")
            if args.caption_file:
                print(f"Added captions to {queries_with_captions}/{len(csv_data)} queries")
            
        except Exception as e:
            print(f"Warning: Could not create CSV format: {e}")

if __name__ == "__main__":
    main()