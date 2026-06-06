from openai import OpenAI
from pathlib import Path
import os
import re
import json
import glob
import tqdm
from multiprocessing import Pool
from functools import partial

API_KEY = ""
BASE_URL = "https://api.deepseek.com"
MODEL_GPT = "deepseek-chat"

INPUT_MD_DIR = r"E:\a\dou\SAM\LLM\mds"  
OUTPUT_CHUNKS_DIR = r"E:\a\dou\SAM\LLM\task-1\task1-chunks"  
OUTPUT_MOLECULES_DIR = r"E:\a\dou\SAM\LLM\task-1\task1-paper-info"  

def comfirm_json_string_gpt(json_string):
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        Fix the following string into a valid JSON format. Output only the JSON string.
        String: {json_string}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": "You are a JSON formatting expert."},
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error in comfirm_json_string_gpt: {e}")
        return json_string

def comfirm_json_string(response):
    
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    json_str = json_match.group(1) if json_match else response.strip()
    json_str = re.sub(r'[\\"\']', '', json_str)
    return json_str

def split_by_heading(markdown_text, heading_level='#'):
    
    pattern = r'(?=\n{})'.format(re.escape(heading_level))
    split_texts = re.split(pattern, markdown_text)
    return [block.strip() for block in split_texts if block.strip() and len(block) > 50]

def segment_classification(text_split):
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        Classify this text segment into one category: 
        Abstract, Introduction, Materials and methods, Results and discussion, Conclusions, References.
        Output only the category name.
        Text: {text_split[:2000]}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": "You are an academic paper classifier."},
                {"role": "user", "content": prompt}
            ],
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Classification API Error: {e}")
        return "Unknown"

def process_file(md_path, output_dir):
    
    try:
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as file:
            md_content = file.read()
        
        content_splits = []
        for level in ['#', '##', '###']:
            splits = split_by_heading(md_content, level)
            content_splits.extend(splits)
        
        chunks = []
        for idx, split_text in enumerate(content_splits):
            category = segment_classification(split_text)
            chunks.append({
                "id": idx + 1,
                "chunk": split_text,
                "category": category
            })
        
        output_path = os.path.join(output_dir, f"{Path(md_path).stem}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error processing {md_path}: {e}")

def get_niox_related_molecules(text):

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = f"""
        You are an expert in perovskite solar cells. Extract ANY mention of molecules or functional groups used to modify the NiOx hole transport layer. Focus on:
        1. Molecule/group names (e.g., "2PACz", "phosphonic acid", "SAMs").
        2. Brief description of their application (e.g., "passivates interface defects").
        Requirements:
        - Extract ONLY information explicitly related to NiOx modification.
        - If no relevant information, return an empty JSON {{}}.
        - Output a valid JSON: {{"molecule": "description", ...}}.
        Text: {text[:3000]}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": "You extract molecular data for NiOx interface engineering."},
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Molecule extraction API Error: {e}")
        return "{}"

def extract_molecules_from_chunks(chunks_path):

    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        molecules_dict = {}
        for chunk in chunks:
            if chunk['category'] in ['Abstract', 'Introduction', 'Materials and methods', 'Results and discussion', 'Conclusions']:
                result_text = get_niox_related_molecules(chunk['chunk'])
                cleaned_json = comfirm_json_string(result_text)
                try:
                    mol_data = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    fixed_json = comfirm_json_string_gpt(cleaned_json)
                    mol_data = json.loads(fixed_json) if fixed_json else {}
                
                if isinstance(mol_data, dict) and mol_data:
                    molecules_dict.update(mol_data)
        
        output_path = os.path.join(OUTPUT_MOLECULES_DIR, f"niox_{Path(chunks_path).stem}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(molecules_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error in {chunks_path}: {e}")

def md_segment():

    os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)

    md_pattern = os.path.join(INPUT_MD_DIR, "**", "*.md")
    md_paths = glob.glob(md_pattern, recursive=True)
    print(f"Found {len(md_paths)} .md files in {INPUT_MD_DIR}")

    processed_files = set([f.replace('.json', '') for f in os.listdir(OUTPUT_CHUNKS_DIR)])
    md_paths = [p for p in md_paths if Path(p).stem not in processed_files]
    
    if not md_paths:
        print("All files already processed.")
        return
    
    print(f"Processing {len(md_paths)} new files...")
    for path in tqdm.tqdm(md_paths, desc="Segmenting MD files"):
        process_file(path, OUTPUT_CHUNKS_DIR)

def extract_niox_molecules():

    os.makedirs(OUTPUT_MOLECULES_DIR, exist_ok=True)
    
    chunk_pattern = os.path.join(OUTPUT_CHUNKS_DIR, "*.json")
    chunk_paths = glob.glob(chunk_pattern)
    print(f"Found {len(chunk_paths)} chunk files in {OUTPUT_CHUNKS_DIR}")
    
    processed_files = set([f.replace('niox_', '') for f in os.listdir(OUTPUT_MOLECULES_DIR)])
    chunk_paths = [p for p in chunk_paths if f"niox_{Path(p).stem}.json" not in processed_files]
    
    if not chunk_paths:
        print("All chunk files already processed for molecule extraction.")
        return
    
    print(f"Extracting molecules from {len(chunk_paths)} chunk files...")
    
    with Pool(processes=min(4, len(chunk_paths))) as pool:
        list(tqdm.tqdm(
            pool.imap_unordered(extract_molecules_from_chunks, chunk_paths), 
            total=len(chunk_paths),
            desc="Extracting molecules"
        ))

def main():

    print("Starting NiOx molecule extraction pipeline...")
    print(f"Input directory: {INPUT_MD_DIR}")
    print(f"Chunks output: {OUTPUT_CHUNKS_DIR}")
    print(f"Molecules output: {OUTPUT_MOLECULES_DIR}")
    
    print("\n" + "="*50)
    print("STEP 1: Text Segmentation and Classification")
    print("="*50)
    md_segment()
    
    print("\n" + "="*50)
    print("STEP 2: NiOx-related Molecule Extraction")
    print("="*50)
    extract_niox_molecules()
    
    print("\nPipeline completed successfully!")

if __name__ == '__main__':
    main()