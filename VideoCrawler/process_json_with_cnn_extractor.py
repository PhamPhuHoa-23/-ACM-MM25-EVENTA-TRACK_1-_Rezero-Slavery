import json
import csv
import os
import concurrent.futures
from urllib.parse import urlparse
from cnn_specific_extractor import get_cnn_video

def extract_domain(url):
    """Extract the domain name from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain

def process_url(args):
    """Process a single URL to extract video."""
    index, url, total = args
    print(f"\nProcessing URL {index+1}/{total}: {url}")
    
    # Check if URL is from CNN
    if "cnn.com" in url.lower():
        # Use CNN-specific extractor
        has_video, video_url = get_cnn_video(url)
    else:
        # For non-CNN URLs, you could use a different extractor here
        # For now, we'll just say no video found
        has_video, video_url = False, None
    
    if has_video:
        domain = extract_domain(url)
        result = {
            'source_url': url,
            'domain': domain,
            'video_url': video_url
        }
        print(f"✓ Found video URL: {video_url}")
        return result
    else:
        print(f"× No video found at {url}")
        return None

def process_urls_from_json(json_file, output_file, limit=None, max_workers=1):
    """Process URLs from JSON file and extract video URLs."""
    try:
        # Load JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract URLs based on structure
        urls = []
        if isinstance(data, list):
            # If it's a list of URLs or objects with URLs
            for item in data:
                if isinstance(item, str) and (item.startswith('http') or item.startswith('www')):
                    urls.append(item)
                elif isinstance(item, dict):
                    for key in ['url', 'link', 'href', 'src']:
                        if key in item and isinstance(item[key], str):
                            urls.append(item[key])
                            break
        elif isinstance(data, dict):
            # If it's a dictionary with URLs as values or in nested structures
            for key, value in data.items():
                if isinstance(value, str) and (value.startswith('http') or value.startswith('www')):
                    urls.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and (item.startswith('http') or item.startswith('www')):
                            urls.append(item)
                        elif isinstance(item, dict):
                            for url_key in ['url', 'link', 'href', 'src']:
                                if url_key in item and isinstance(item[url_key], str):
                                    urls.append(item[url_key])
                                    break
                elif isinstance(value, dict):
                    for url_key in ['url', 'link', 'href', 'src']:
                        if url_key in value and isinstance(value[url_key], str):
                            urls.append(value[url_key])
                            break
        
        # Filter valid URLs and keep only CNN URLs for now
        valid_urls = [url for url in urls if url.startswith('http') and 'cnn.com' in url.lower()]
        print(f"Found {len(valid_urls)} valid CNN URLs")
        
        # Apply limit if specified
        if limit and limit < len(valid_urls):
            valid_urls = valid_urls[:limit]
            print(f"Processing first {limit} URLs")
            
        total = len(valid_urls)
        
        # Process URLs (in parallel if max_workers > 1)
        results = []
        if max_workers > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create arguments for each URL (index, url, total)
                args_list = [(i, url, total) for i, url in enumerate(valid_urls)]
                
                # Process URLs in parallel
                for result in executor.map(process_url, args_list):
                    if result:
                        results.append(result)
        else:
            # Sequential processing
            for i, url in enumerate(valid_urls):
                result = process_url((i, url, total))
                if result:
                    results.append(result)
        
        # Write results to CSV
        print(f"\nWriting results to {output_file}...")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Source URL', 'Domain', 'Video URL'])
            for result in results:
                writer.writerow([
                    result['source_url'],
                    result['domain'],
                    result['video_url']
                ])
        
        # Also save to JSON for better readability
        json_output = f"{os.path.splitext(output_file)[0]}.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        print(f"Done! Found videos on {len(results)} pages out of {len(valid_urls)} checked.")
        print(f"Results saved to {output_file} and {json_output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    json_file = "database_article_to_url.json"
    output_file = "cnn_video_urls.csv"
    
    # Ask for number of URLs to process
    try:
        limit_input = input("How many CNN URLs do you want to process? (Enter a number or 'all' for all URLs): ")
        limit = None if limit_input.lower() == 'all' else int(limit_input)
    except ValueError:
        print("Invalid input. Processing 5 URLs by default.")
        limit = 5
    
    # Ask for number of concurrent workers
    try:
        workers_input = input("How many concurrent workers? (recommended: 1, can be slow with more due to browser instances): ")
        max_workers = int(workers_input)
        if max_workers < 1:
            print("Invalid number. Using 1 worker.")
            max_workers = 1
    except ValueError:
        print("Invalid input. Using 1 worker.")
        max_workers = 1
    
    process_urls_from_json(json_file, output_file, limit, max_workers)

if __name__ == "__main__":
    main() 