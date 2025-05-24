# Vector Retrieval System with Qdrant/FAISS

This project implements a vector retrieval system using either Qdrant vector database or FAISS for CLIP vectors extracted from images.

## Prerequisites

- Docker and Docker Compose (for Qdrant version)
- Python 3.8+
- pip

## Two Versions Available

### 1. **FAISS Version** (Recommended - Faster & Simpler)
- File: `retrieval_system_faiss.py`
- No Docker required
- Faster search performance
- Local file storage

### 2. **Qdrant Version** (Database approach)
- File: `retrieval_system.py` 
- Requires Docker
- Persistent database storage
- Web UI monitoring

## Setup Instructions

### 1. Start Qdrant Database

First, start the Qdrant vector database using Docker Compose:

```bash
docker-compose up -d
```

This will:
- Start Qdrant on port 6333 (HTTP API) and 6334 (gRPC)
- Create a persistent volume for data storage in `./qdrant_data`
- Enable CORS for web access

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Ensure you have:
- `clip_b16/` directory containing `.jpg.npy` files (database vectors)
- `query_b16_laion/clip_b16_query/` directory containing `.jpg.npy` files (query vectors)
- `Rezero_Slavery_EVENTA2025_Track1.csv` file with query to article mappings
- `database_images_to_article_v.0.1.json` file with image to article mappings

## Usage

### FAISS Version (Recommended)

#### First time setup (build index + query):
```bash
python retrieval_system_faiss.py --vector_size 640 --vectors_dir "path/to/vectors" --query_dir "path/to/queries"
```

#### Skip building index if already exists:
```bash
python retrieval_system_faiss.py --skip_build --action all --vector_size 640 --vectors_dir "path/to/vectors" --query_dir "path/to/queries"
```

#### Only build index:
```bash
python retrieval_system_faiss.py --action setup --vector_size 640 --vectors_dir "path/to/vectors" --query_dir "path/to/queries"
```

#### Only query (requires existing index):
```bash
python retrieval_system_faiss.py --action query --query_dir "path/to/queries"
```

### Qdrant Version

#### Run Complete Pipeline (Setup + Query)

```bash
python retrieval_system.py
```

### Run Individual Steps

#### Setup Database Only
```bash
python retrieval_system.py --action setup
```

#### Query Only (assumes database is already set up)
```bash
python retrieval_system.py --action query
```

### Custom Configuration

```bash
python retrieval_system.py \
    --host localhost \
    --port 6333 \
    --vector_size 512 \
    --vectors_dir clip_b16 \
    --query_dir query_b16_laion/clip_b16_query \
    --mapping_file Rezero_Slavery_EVENTA2025_Track1.csv \
    --json_mapping_file database_images_to_article_v.0.1.json \
    --collection clip_vectors
```

#### Examples for Different Models

**CLIP B/16** (512 dimensions):
```bash
python retrieval_system.py --vector_size 512 --vectors_dir clip_b16 --query_dir query_b16
```

**CLIP L/14** (768 dimensions):
```bash
python retrieval_system.py --vector_size 768 --vectors_dir clip_l14 --query_dir query_l14
```

**Custom Model**:
```bash
python retrieval_system.py --vector_size 1024 --vectors_dir custom_vectors --query_dir custom_queries
```

## What the System Does

1. **Database Setup**: 
   - Creates a Qdrant collection with cosine similarity
   - Uploads all vectors from `clip_b16/` directory
   - Removes `.jpg` extension from filenames (handles `<id>.jpg.npy` format)

2. **Querying**:
   - Processes each query vector from the query directory
   - Finds top 10 most similar vectors in the database
   - Outputs results with similarity scores
   - Saves results to `retrieval_results.json`

3. **Output Format**:
   ```json
   [
     {
       "query_id": "68d7ca1aaf634cd4",
       "results": [
         {
           "image_id": "5fd78beb7fe34584",
           "article_id": "f8097c7d27a8aac6",
           "score": 0.8543
         }
       ],
       "expected_articles": ["article_id_1", "article_id_2", ...]
     }
   ]
   ```

## File Structure

```
.
├── docker-compose.yml          # Qdrant Docker setup
├── requirements.txt            # Python dependencies
├── retrieval_system.py         # Main script
├── README.md                   # This file
├── clip_b16/                   # Database vectors (.jpg.npy files)
├── query_b16_laion/
│   └── clip_b16_query/         # Query vectors (.jpg.npy files)
├── Rezero_Slavery_EVENTA2025_Track1.csv  # Query-to-article mapping
├── database_images_to_article_v.0.1.json # Image-to-article mapping
└── qdrant_data/                # Qdrant data storage (created by Docker)
```

## Key Features

- **Automatic Extension Handling**: Removes `.jpg` from `.jpg.npy` filenames
- **Flexible Vector Dimensions**: Support any vector size (auto-detect or manual specify)
- **Configurable Directories**: Easy to switch between different model outputs
- **Batch Processing**: Uploads vectors in batches for efficiency
- **Cosine Similarity**: Uses cosine distance for CLIP vector comparison
- **Persistent Storage**: Database data persists between Docker restarts
- **Dual Mapping Support**: Uses both CSV (query expectations) and JSON (image-to-article) mappings
- **Flexible Configuration**: Command-line arguments for all parameters

## Troubleshooting

### Qdrant Connection Issues
- Ensure Docker is running and container is started
- Check if port 6333 is available
- Wait a few seconds after starting Docker before running the script

### Vector Size Mismatch
- The script auto-detects vector size from the first file
- Ensure all vectors have the same dimensions

### File Format Issues
- Ensure all `.npy` files contain valid numpy arrays
- Check that vectors are in float32 format

## Monitoring

You can access the Qdrant web UI at `http://localhost:6333/dashboard` to monitor your collections and data.

## Stopping the System

To stop the Qdrant database:

```bash
docker-compose down
```

To stop and remove all data:

```bash
docker-compose down -v
``` 