# VideoCrawler

A Python tool for extracting video URLs from CNN news articles using web scraping techniques.

## Features

- Extract video URLs from CNN news articles
- Process multiple URLs from JSON database  
- Concurrent processing support
- Export results to CSV and JSON formats
- Selenium-based web scraping with Chrome driver

## Requirements

- Python 3.7+
- Chrome browser installed on your system

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. The Chrome driver will be automatically downloaded by webdriver-manager when you first run the code.

## Usage

### Basic Usage

Run the main script to process CNN URLs from the database:

```bash
python process_json_with_cnn_extractor.py
```

The script will prompt you for:
- Number of URLs to process (enter a number or 'all')
- Number of concurrent workers (recommended: 1)

### Input Data

The script reads from `database_article_to_url.json` which should contain URLs to CNN articles.

### Output

The script generates:
- `cnn_video_urls.csv` - CSV file with extracted video URLs
- `cnn_video_urls.json` - JSON file with the same data

### Using the CNN Extractor Directly

You can also import and use the CNN extractor in your own code:

```python
from cnn_specific_extractor import get_cnn_video

# Extract video from a single CNN URL
has_video, video_url = get_cnn_video("https://www.cnn.com/your-article-url")

if has_video:
    print(f"Found video: {video_url}")
else:
    print("No video found")
```

## Configuration

The crawler is specifically designed for CNN articles. To support other news sites, you would need to modify the extraction logic in `cnn_specific_extractor.py`.

## Limitations

- Currently only supports CNN articles
- Requires Chrome browser
- May be slow due to browser automation
- Some videos may not be accessible due to anti-bot measures

## Files

- `cnn_specific_extractor.py` - Core CNN video extraction logic
- `process_json_with_cnn_extractor.py` - Main processing script
- `database_article_to_url.json` - Input URL database
- `requirements.txt` - Python dependencies 