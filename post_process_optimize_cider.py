import pandas as pd
import json
from nltk.util import ngrams


input_path = r"E:\Di_cu\test\YOURCAPTION.csv" #Adjust your path
base_json_path = r"E:\Di_cu\test\captions_backup.json" #Adujst your path (File json containing base captions)
output_path = r"E:\Di_cu\test\YOURCAPTION_trunc106.csv" #Adjust your path


with open(base_json_path, 'r', encoding='utf-8') as f:
    base_captions = json.load(f)


def truncate_to_106_words(caption):
    words = str(caption).split()
    return " ".join(words[:106]) + " . " if len(words) > 106 else caption


def extract_useful_phrases(base_caption, gen_caption, n=2, max_phrases=2):
    base_ngrams = list(ngrams(base_caption.split(), n))
    useful = []
    for ng in base_ngrams:
        phrase = " ".join(ng)
        if phrase.lower() not in gen_caption.lower():
            useful.append(phrase)
        if len(useful) >= max_phrases:
            break
    return useful


def merge_caption(gen_caption, base_caption):
    phrases = extract_useful_phrases(base_caption, gen_caption)
    if phrases:
        return gen_caption.strip().rstrip('.') + '. ' + '; '.join(phrases)
    return gen_caption


df = pd.read_csv(input_path)


def process_caption(row):
    qid = row['query_id']
    gen = row['generated_caption']
    base = base_captions.get(str(qid), "")
    
    enhanced = merge_caption(gen, base)
    
    return truncate_to_106_words(enhanced)

df['generated_caption'] = df.apply(process_caption, axis=1)


df.to_csv(output_path, index=False)


