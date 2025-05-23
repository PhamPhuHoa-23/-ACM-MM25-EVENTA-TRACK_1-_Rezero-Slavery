import json
import pandas as pd
import argparse

def load_captions_from_csv(caption_file):
    """Load captions from CSV file"""
    try:
        df = pd.read_csv(caption_file)
        
        # Try to detect caption column name
        caption_col = None
        possible_caption_cols = ['generated_caption', 'caption', 'text', 'description']
        
        for col in possible_caption_cols:
            if col in df.columns:
                caption_col = col
                break
        
        if caption_col is None:
            print(f"Available columns in {caption_file}: {list(df.columns)}")
            caption_col = input("Enter the column name for captions: ")
        
        # Create mapping from query_id to caption
        caption_mapping = {}
        query_id_col = 'query_id' if 'query_id' in df.columns else df.columns[0]
        
        for _, row in df.iterrows():
            query_id = row[query_id_col]
            caption = row[caption_col] if pd.notna(row[caption_col]) else ""
            caption_mapping[query_id] = caption
        
        print(f"✓ Loaded {len(caption_mapping)} captions from {caption_file}")
        print(f"✓ Using column '{caption_col}' for captions")
        return caption_mapping
        
    except Exception as e:
        print(f"Error loading captions: {e}")
        return {}

def convert_json_to_csv(json_file, output_csv, caption_file=None):
    """Convert retrieval results JSON to CSV format"""
    
    # Load JSON results
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Load captions if caption file is provided
    caption_mapping = {}
    if caption_file:
        caption_mapping = load_captions_from_csv(caption_file)
    
    # Prepare data for CSV
    csv_data = []
    
    for query_result in results:
        query_id = query_result['query_id']
        
        # Initialize row with query_id
        row = {'query_id': query_id}
        
        # Add article_id_1 to article_id_10 (fill with empty if not enough results)
        for i in range(1, 11):
            col_name = f'article_id_{i}'
            if i-1 < len(query_result['results']):
                # Get article_id from results
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
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    
    # Ensure column order
    columns = ['query_id'] + [f'article_id_{i}' for i in range(1, 11)] + ['generated_caption']
    df = df[columns]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    # Show statistics
    total_queries = len(csv_data)
    queries_with_captions = sum(1 for row in csv_data if row['generated_caption'] != "")
    
    print(f"✓ Converted {total_queries} queries to {output_csv}")
    if caption_file:
        print(f"✓ Added captions to {queries_with_captions}/{total_queries} queries")
    
    # Show sample
    print(f"\nSample output:")
    print(df.head(3).to_string(index=False))
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Convert retrieval results JSON to CSV submission format')
    parser.add_argument('--input', type=str, default='retrieval_results.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output CSV file')
    parser.add_argument('--captions', type=str, default=None, help='CSV file containing captions to copy')
    
    args = parser.parse_args()
    
    try:
        convert_json_to_csv(args.input, args.output, args.captions)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        print("Please run the retrieval system first to generate results")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 