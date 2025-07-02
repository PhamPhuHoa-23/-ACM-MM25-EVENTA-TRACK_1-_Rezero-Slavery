# 📰 News Caption Generator

This project generates enriched news captions by combining base visual descriptions with contextual news article content using fine-tuned language models.

## 🎯 Overview

The caption generation process consists of 4 main stages:
1. **📷 Base Caption Creation** - Generate initial visual descriptions using InstructBLIP
2. **🚀 Model Fine-tuning** - Train a LoRA adapter on DeepSeek-Qwen3 for news caption generation
3. **✨ Enriched Caption Generation** - Create context-aware captions using the fine-tuned model
4. **🔧 Post-processing** - Fix errors and optimize CIDER scores

## ⚙️ Prerequisites

- 🐍 Python 3.8+
- 🖥️ CUDA-capable GPU (recommended)
- 📦 Required packages: `transformers`, `unsloth`, `sentence-transformers`, `pandas`, `PIL`, `torch`

## 📷 Stage 1: Base Caption Creation

**📄 File**: `create-base-captions.ipynb`

### 🎯 Purpose
Generate initial visual descriptions of news images using InstructBLIP model.

### 📋 Input Requirements
- **🖼️ Image Database**: Folder containing matched images (`.jpg` format)
- **🔗 Link File**: CSV file with columns `query_id` and `matched_image_id` linking queries to images

### 📝 Instructions
1. **Adjust paths in the notebook**:
   ```python
   # Update these paths according to your setup
   IMAGE_DIR = "/path/to/your/image/database/"  # Folder with images
   df = pd.read_csv('/path/to/link_images.csv')  # Your link file
   ```

2. **Configure processing**:
   ```python
   PROCESS_HALF = "first"  # Process "first" or "second" half
   IMAGE_EXT = ".jpg"      # Image file extension
   ```

3. **Run the notebook** to generate base captions
4. **📤 Output**: `captions.csv` with columns `query_id`, `matched_image_id`, `caption`

### 📌 Notes
- The notebook processes images in batches to manage memory
- Uses InstructBLIP-Vicuna-7B for detailed visual descriptions
- Focuses on news-relevant visual elements (people, objects, settings, activities)

## 🚀 Stage 2: Model Fine-tuning

**📄 File**: `Fine_Tune_Deepseek_Qwen3.ipynb`

### 🎯 Purpose
Fine-tune DeepSeek-Qwen3-8B model with LoRA for news caption generation.

### 📋 Input Requirements
- **📊 Training Dataset**: `output_merged.json` (download link provided in notebook)
- **☁️ Google Drive**: Mount for saving model checkpoints

### 📝 Instructions
1. **Download training data**:
   ```bash
   !gdown 1F0EMvGBm-l4iXV11Im3zaGOPbifsWxfQ  # Training dataset
   ```

2. **Configure training parameters**:
   ```python
   TRAINING_CONFIG = {
       "output_dir": "./sft_results_upgraded_prompt",
       "per_device_train_batch_size": 8,
       "num_train_epochs": 2,
       "learning_rate": 5e-5,
       # ... other parameters
   }
   ```

3. **Run training** - The notebook will:
   - Load and process the training dataset
   - Apply advanced prompting techniques with semantic search
   - Train LoRA adapter with optimal settings
   - Save model to Google Drive

4. **📤 Output**: Trained LoRA model saved to `/content/drive/MyDrive/lora_model_512`

### ⭐ Key Features
- **🔍 Semantic Search**: Extracts relevant article content using sentence transformers
- **📝 Advanced Prompting**: Prioritizes news content (70%) over visual description (30%)
- **🏷️ Information Extraction**: Automatically identifies people, organizations, locations, events, dates, and key terms

## ✨ Stage 3: Enriched Caption Generation

**📄 File**: `run-inference-private.ipynb`

### 🎯 Purpose
Generate enriched captions using the fine-tuned model and news article context.

### 📋 Input Requirements
- **📷 Base Captions**: From Stage 1 (`base_captions_private.csv`)
- **📰 Top Articles**: `top10_articles.csv` - ranked articles for each query
- **🗃️ Article Database**: `database_preprocessed.json` - full article content
- **🤖 Trained Model**: LoRA model from Stage 2 or use provided model

### 📝 Instructions
1. **Download required files**:
   ```bash
   !gdown 1vqmL82yiPhgAOCF43gSBJCZ2nlrZRZrL  # database
   !gdown 1h7FTRHfl2KJm8k8mfMsCFSN6nTdo-H1W  # top10_articles  
   !gdown 1GqxM2uf5y_bMrCzIMMMAMyenRHPA6Q6p  # base_captions
   ```

2. **Configure model path**:
   ```python
   MODEL_PATH = "checkpoint-200"  # Adjust to your model path
   # Or use provided model from link_model_lora.txt
   ```

3. **Run inference** - The notebook will:
   - Load the fine-tuned model and base captions
   - Extract key information from news articles
   - Generate enriched captions with news context
   - Save results with checkpoint support

4. **📤 Output**: 
   - `ReZeroSlavery.json` and `ReZeroSlavery.csv` with enriched captions
   - Checkpoint files for resume capability

### ⭐ Key Features
- **💾 Checkpoint System**: Resume interrupted processing
- **🔍 Semantic Article Extraction**: Focus on relevant article content
- **📊 Progress Tracking**: ETA and memory monitoring
- **🛡️ Error Handling**: Robust processing with fallbacks

## 🔧 Stage 4: Post-processing

### 🩹 Error Fixing

**📄 File**: `Post_process_fix_error_captions.py`

Some generated captions may contain formatting errors or include the original prompt.

**💻 Usage**:
```python
# Adjust file paths
RESULTS_FILE = "your_generated_captions.csv"           # Main results file
REZERO_FILE = "REZERO_SLAVERY_backup.csv"             # Backup captions
OUTPUT_FILE = "captions_fixed.csv"                    # Output file

# Run the script
python Post_process_fix_error_captions.py
```

**✅ What it does**:
- Identifies captions starting with "user" (error indicator)
- Extracts clean captions from "YOUR CAPTION:" headers
- Falls back to backup captions when available
- Provides default captions for unfixable errors

### 📈 CIDER Score Optimization

**📄 File**: `post_process_optimize_cider.py`

Optimize CIDER scores by merging generated captions with base caption phrases.

**💻 Usage**:
```python
# Adjust file paths
input_path = "your_generated_captions.csv"            # Input captions
base_json_path = "base_captions.json"                 # Base captions
output_path = "captions_optimized.csv"                # Optimized output

# Run the script
python post_process_optimize_cider.py
```

**✅ What it does**:
- Extracts useful n-gram phrases from base captions
- Merges them with generated captions
- Truncates to 106 words for optimal CIDER scoring
- Maintains semantic coherence

## 📁 File Structure

```
📂 project/
├── 📓 create-base-captions.ipynb           # Stage 1: Base caption generation
├── 📓 Fine_Tune_Deepseek_Qwen3.ipynb       # Stage 2: Model fine-tuning
├── 📓 run-inference-private.ipynb          # Stage 3: Enriched caption generation
├── 🐍 Post_process_fix_error_captions.py   # Stage 4a: Error fixing
├── 🐍 post_process_optimize_cider.py       # Stage 4b: CIDER optimization
├── 🔗 link_model_lora.txt                  # Pre-trained model link
└── 📊 top10_articles.csv                   # Article rankings (2000 queries)
```

## ⚠️ Important Notes

### 🤖 Model Resources
- 🔗 Pre-trained LoRA model available at: [Google Drive Link](https://drive.google.com/file/d/1pF6IQ-nOJf5sp6gYTSrEqr2nmZ0uurCx/view?usp=sharing) (from `link_model_lora.txt`)
- 🔄 You can either retrain using Stage 2 or download the pre-trained model

### 📝 File Path Configuration
**⚠️ Always adjust file paths** in each script according to your directory structure:
- 🖼️ Image directories in base caption generation
- 🤖 Model paths in inference
- 📁 Input/output files in post-processing scripts

### 🚀 Performance Tips
- 🖥️ Use GPU for faster processing in all stages
- 🔍 Enable semantic search for better article content extraction
- 💾 Monitor memory usage during batch processing
- ✅ Use checkpoints to resume interrupted processes

### 🔄 Data Flow
```
🖼️ Images → 📷 Base Captions → 🤖 Fine-tuned Model → ✨ Enriched Captions → 🩹 Fixed Captions → 📈 Optimized Captions
```

This pipeline ensures high-quality news captions that combine visual accuracy with rich contextual information from news articles.

## 🚀 Quick Start Guide

1. **📋 Prepare your data**: Organize images and create the link CSV file
2. **📷 Generate base captions**: Run `create-base-captions.ipynb`
3. **🚀 Train model** (optional): Run `Fine_Tune_Deepseek_Qwen3.ipynb` or download pre-trained model
4. **✨ Generate enriched captions**: Run `run-inference-private.ipynb`
5. **🔧 Post-process**: Run error fixing and CIDER optimization scripts
6. **🎉 Result**: High-quality news captions ready for evaluation

## 🛠️ Troubleshooting

- **💾 Memory issues**: Reduce batch sizes in training/inference
- **📁 Path errors**: Double-check all file paths in configuration sections
- **🤖 Model loading errors**: Ensure CUDA compatibility and sufficient GPU memory
- **📝 Caption errors**: Use post-processing scripts to clean up outputs

## 📄 License

This project is provided as-is for research purposes. Please ensure compliance with model licenses (DeepSeek, InstructBLIP) when using for commercial applications.