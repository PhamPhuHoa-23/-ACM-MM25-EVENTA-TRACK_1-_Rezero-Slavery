# Image-to-Image Retrieval System

A multi-stage image retrieval system using vector similarity search with Qdrant database and advanced computer vision-based re-ranking.

## Overview

This system performs image-to-image retrieval in two main phases:
1. **Vector Retrieval**: Uses Qdrant vector database for initial similarity search
2. **Multi-Stage Re-ranking**: Applies computer vision techniques to improve retrieval accuracy

## Features

- Vector similarity search using Qdrant database
- 4-stage re-ranking pipeline with feature matching
- Support for multiple embedding models (BEIT3, SigLIP, etc.)
- Docker-based Qdrant deployment
- Automatic CSV submission format generation

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- OpenCV 4.4.0+ (with SIFT support)
- Required Python packages (see Installation)

## Installation

1. **Clone the repository and install dependencies:**
```bash
pip install opencv-python numpy pandas qdrant-client tqdm
```

2. **Start Qdrant database:**
```bash
docker-compose up -d
```

3. **Verify Qdrant is running:**
```bash
curl http://localhost:6333/health
```

## File Structure

```
├── retrieval_system.py           # Vector database setup and retrieval
├── rerank_code.py                # Multi-stage re-ranking pipeline
├── docker-compose.yml            # Qdrant database configuration
├── database_images_to_article_v.0.1.json  # Image to article mapping
├── database_article_to_url.json  # Article to URL mapping
└── retrieval_results/             # Model results directory
    ├── beit3_base_ft.json
    ├── beit3_coco_base.json
    └── siglipso.json
```

## Usage

### Step 1: Vector Database Setup and Initial Retrieval

```bash
python retrieval_system.py \
    --vectors_dir /path/to/database/vectors \
    --query_dir /path/to/query/vectors \
    --json_mapping_file database_images_to_article_v.0.1.json \
    --collection my_collection \
    --vector_size 512
```

**Parameters:**
- `--vectors_dir`: Directory containing database image vectors (.npy files)
- `--query_dir`: Directory containing query image vectors (.npy files)
- `--json_mapping_file`: JSON file mapping image IDs to article IDs
- `--collection`: Qdrant collection name
- `--vector_size`: Vector dimensions (auto-detected if not specified)
- `--skip_upload`: Skip uploading if collection already exists
- `--action`: Choose 'setup', 'query', or 'all'

### Step 2: Multi-Stage Re-ranking

```bash
python rerank_code.py \
    --model1 retrieval_results/beit3_coco_base.json \
    --model2 retrieval_results/siglipso.json \
    --model3 retrieval_results/beit3_base_ft.json \
    --query_images_path /path/to/query/images \
    --database_images_path /path/to/database/images
```

**Parameters:**
- `--model1`, `--model2`, `--model3`: Input JSON files with retrieval results
- `--model_names`: Custom names for models (optional)
- `--weights`: Custom weights for ensemble (optional)
- `--query_images_path`: Directory containing query images (.jpg)
- `--database_images_path`: Directory containing database images (.jpg)
- `--stage1_output`: Stage 1 output filename (default: reranked_stage1.json)
- `--stage2_output`: Stage 2 output filename (default: reranked_stage2.json)
- `--stage3_output`: Stage 3 output filename (default: reranked_stage3.json)
- `--final_output`: Final output filename (default: reranked_final.json)

### Running Individual Stages

**Stage 1 only (Ensemble):**
```bash
python rerank_code.py --only_stage 1
```

**Stage 2 only (Top-2 Re-ranking):**
```bash
python rerank_code.py --only_stage 2 --skip_stage1
```

**Stage 3 only (Top-3 Optimization):**
```bash
python rerank_code.py --only_stage 3 --skip_stage1 --skip_stage2
```

**Stage 4 only (Top1 vs Top4):**
```bash
python rerank_code.py --only_stage 4 --skip_stage1 --skip_stage2 --skip_stage3
```

## Re-ranking Pipeline Stages

### Stage 1: Model Ensemble
- Combines results from multiple models
- Uses top-2 pooling with position weighting
- Applies appearance bonus for consistent results

### Stage 2: Enhanced Top-2 Re-ranking
- ORB and SIFT feature matching
- Homography validation
- Spatial consistency analysis

### Stage 3: Top-3 Optimization
- Enhanced feature scoring
- Multiple validation methods
- Adaptive preprocessing

### Stage 4: Top1 vs Top4 Comparison
- Compares top result with 4th position
- Promotes better matches to top position

## Output Formats

### JSON Format
```json
[
  {
    "query_id": "query_001",
    "results": [
      {
        "image_id": "img_001",
        "article_id": "article_123",
        "score": 0.95
      }
    ]
  }
]
```

### CSV Submission Format
```csv
query_id,article_id_1,article_id_2,...,article_id_10,generated_caption
query_001,article_123,article_456,...,article_789,""
```

## Configuration

### Qdrant Settings
Modify `docker-compose.yml` to adjust:
- Port mappings
- Storage volumes
- Memory limits

### Re-ranking Parameters
Key parameters in `rerank_code.py`:
- `confidence_threshold`: Minimum confidence for re-ranking
- `feature_quality_threshold`: Minimum feature score
- `min_inlier_ratio_diff`: Minimum difference for position swapping

## Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB+ RAM, 8+ CPU cores, SSD storage
- **GPU**: Not required (CPU-based feature matching)
