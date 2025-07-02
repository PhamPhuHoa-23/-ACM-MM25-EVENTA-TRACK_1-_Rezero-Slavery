
import pandas as pd
import re
from tqdm import tqdm


RESULTS_FILE       = "top10_articles_with_caption.csv" #The main file cointaining error captions need to be fix, you should adjust the file path here
REZERO_FILE        = "REZERO_SLAVERY_EVENTA2025_Track1.csv" #The backup file
OUTPUT_FILE        = "top10_articles_with_caption_fixed.csv"#The output file
VERIFY_FILE        = "top10_articles_with_caption_fixed.csv"  #Ensure that there is no error in the output file.

def get_error_query_ids(df):
    
    error_mask   = df['generated_caption'].astype(str).str.startswith('user')
    error_rows   = df[error_mask]
    return error_rows['query_id'].tolist()

def load_rezero_captions():
    
    try:
        rezero_df = pd.read_csv(REZERO_FILE, dtype={'query_id': str})
        rezero_dict = dict(
            zip(
                rezero_df['query_id'],
                rezero_df['generated_caption']
            )
        )
        return rezero_dict
    except Exception as e:
        return {}

def extract_after_specific_header(text):
    header_pattern = r"(?mi)^YOUR CAPTION\s*:?\s*([\s\S]*)"
    match = re.search(header_pattern, text)
    if match:
        return match.group(1).strip()
    return None

def fix_error_captions():
    try:
        df = pd.read_csv(RESULTS_FILE, dtype={'query_id': str})
    except Exception as e:
        return

    error_query_ids = get_error_query_ids(df)
    if not error_query_ids:
        return

    rezero_captions = load_rezero_captions()


    count_from_header = 0
    count_from_rezero = 0
    count_default     = 0

    for query_id in tqdm(error_query_ids, desc="Fixing captions"):
        idx_mask = df['query_id'] == query_id
        if not idx_mask.any():
            continue
        idx = df[idx_mask].index[0]
        orig_text = str(df.loc[idx, 'generated_caption'])

        new_cap = extract_after_specific_header(orig_text)
        if new_cap:
            df.loc[idx, 'generated_caption'] = new_cap
            count_from_header += 1
            continue

        if query_id in rezero_captions:
            df.loc[idx, 'generated_caption'] = rezero_captions[query_id]
            count_from_rezero += 1
        else:
            default_caption = "This is a news image. Generated caption not available in reference file."
            df.loc[idx, 'generated_caption'] = default_caption
            count_default += 1


    df.to_csv(OUTPUT_FILE, index=False)

    for i, qid in enumerate(error_query_ids[:3], start=1):
        mask_q = df['query_id'] == qid
        if mask_q.any():
            new_cap = df[mask_q]['generated_caption'].iloc[0]
            print(f"  {i}. Query ID: {qid}")
            print(f"     New caption: {new_cap[:150]}...\n")

def verify_fix():
    try:
        df = pd.read_csv(VERIFY_FILE, dtype={'query_id': str})
        error_mask = df['generated_caption'].astype(str).str.startswith('user')
        remaining_errors = df[error_mask]
    except Exception as e:
        print(f"Error when checking: {e}")

def compare_files():
    try:
        results_df = pd.read_csv(RESULTS_FILE, dtype={'query_id': str})

        error_mask       = results_df['generated_caption'].astype(str).str.startswith('user')
        error_query_ids  = set(results_df[error_mask]['query_id'])

        rezero_df        = pd.read_csv(REZERO_FILE, dtype={'query_id': str})
        rezero_query_ids = set(rezero_df['query_id'])

        common_ids       = error_query_ids.intersection(rezero_query_ids)
        missing_ids      = error_query_ids - rezero_query_ids

        if missing_ids:
            print("\nSome query cannot be found")
            for i, qid in enumerate(list(missing_ids)[:5], start=1):
                print(f"    {i}. {qid}")
            if len(missing_ids) > 5:
                print(f"    ... and {len(missing_ids) - 5} different query_id")
    except Exception as e:
        print(f"Error when comparing: {e}")

if __name__ == "__main__":
    

    
    compare_files()

    
    choice = input("\nContinue fix? (y/N): ").strip().lower()
    if choice != 'y':
        print("Stop.")
    else:
        
        fix_error_captions()
        
        verify_fix()
        print("\nComplete!")
